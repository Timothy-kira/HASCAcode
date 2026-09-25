# %% [markdown]
# # WEAR 2026 — Step C v2: fusion network, 6 epochs (v1 peaked at ep 2-5), 2-seed average
# IMU branch: 1D-CNN over one 50x3 sensor window + sensor-location embedding.
# Video branch: per-subject-centred 15x768 VideoMAE frames -> projection -> temporal conv -> pooling.
# Augmentations: random valid sensor per tile each epoch, random 3D rotation, left-arm mirroring,
# scaling/jitter, video noise, modality dropout. Subject-grouped 5-fold, same folds/eval sensor as the baseline.

# %%
import glob, os, time, math
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

def find(name): return os.path.dirname(glob.glob(f"/kaggle/input/**/{name}", recursive=True)[0])
P, B = find("tr_meta.csv"), find("eval_loc.npy")
OUT = "/kaggle/working"
LOCS = ["left_arm", "left_leg", "right_arm", "right_leg"]
torch.set_num_threads(os.cpu_count()); torch.manual_seed(0); rng = np.random.default_rng(0)

tr_meta = pd.read_csv(f"{P}/tr_meta.csv"); te_meta = pd.read_csv(f"{P}/te_meta.csv")
tr_imu = np.load(f"{P}/tr_imu.npy"); te_imu = np.load(f"{P}/te_imu.npy")
tr_valid = np.load(f"{P}/tr_valid.npy") if os.path.exists(f"{P}/tr_valid.npy") else np.ones((len(tr_meta), 4), bool)
eval_loc = np.load(f"{B}/eval_loc.npy")
te_loc = te_meta.sensor_location.map({l: i for i, l in enumerate(LOCS)}).to_numpy()
y = tr_meta.label.values; sbj = tr_meta.sbj.values; T, N = len(tr_meta), len(te_meta)

def centred_video(path, groups):
    V = np.load(path).astype(np.float32)
    for s in np.unique(groups):
        m = groups == s; V[m] -= V[m].mean((0, 1))
    V /= V.reshape(-1, 768).std(0) + 1e-6
    return V.astype(np.float16)
tr_vid = centred_video(f"{P}/tr_vid.npy", sbj); te_vid = centred_video(f"{P}/te_vid.npy", te_meta.sbj_id.values)
print("loaded", tr_vid.shape, te_vid.shape)

# %%
def prep_imu(X, loc):
    X = X.astype(np.float32).copy()
    X[loc == 0, :, 0] *= -1  # mirror left arm onto right arm
    return X

def rand_rot(n, deg=20):
    a = np.deg2rad(rng.uniform(-deg, deg, (n, 3)))
    cx, cy, cz, sx, sy, sz = np.cos(a[:, 0]), np.cos(a[:, 1]), np.cos(a[:, 2]), np.sin(a[:, 0]), np.sin(a[:, 1]), np.sin(a[:, 2])
    o, z = np.ones(n), np.zeros(n)
    Rx = np.stack([o, z, z, z, cx, -sx, z, sx, cx], 1).reshape(n, 3, 3)
    Ry = np.stack([cy, z, sy, z, o, z, -sy, z, cy], 1).reshape(n, 3, 3)
    Rz = np.stack([cz, -sz, z, sz, cz, z, z, z, o], 1).reshape(n, 3, 3)
    return (Rx @ Ry @ Rz).astype(np.float32)

def augment(X):
    X = np.einsum("nij,ntj->nti", rand_rot(len(X)), X)
    X *= rng.uniform(0.85, 1.15, (len(X), 1, 1)).astype(np.float32)
    return X + rng.normal(0, 0.02, X.shape).astype(np.float32)

def imu_input(X):  # (n,50,3) -> (n,4,50) with magnitude channel
    return np.concatenate([X, np.linalg.norm(X, axis=2, keepdims=True)], 2).transpose(0, 2, 1)

# %%
class Net(nn.Module):
    def __init__(self, d=128):
        super().__init__()
        c = lambda i, o, k: nn.Sequential(nn.Conv1d(i, o, k, padding=k // 2), nn.BatchNorm1d(o), nn.GELU())
        self.imu = nn.Sequential(c(4, 64, 7), c(64, 64, 5), nn.MaxPool1d(2), c(64, 128, 5), c(128, 128, 3), nn.MaxPool1d(2), c(128, d, 3))
        self.loc = nn.Embedding(4, d)
        self.vproj = nn.Sequential(nn.Dropout(0.2), nn.Linear(768, d), nn.GELU())
        self.vtemp = c(d, d, 3)
        self.head = nn.Sequential(nn.LayerNorm(4 * d), nn.Dropout(0.3), nn.Linear(4 * d, 256), nn.GELU(), nn.Dropout(0.3), nn.Linear(256, 19))

    def forward(self, imu, loc, vid, drop_imu=None, drop_vid=None):
        h = self.imu(imu) + self.loc(loc)[:, :, None]
        hi = torch.cat([h.mean(2), h.amax(2)], 1)
        v = self.vtemp(self.vproj(vid).transpose(1, 2))
        hv = torch.cat([v.mean(2), v.amax(2)], 1)
        if drop_imu is not None: hi = hi * (1 - drop_imu[:, None])
        if drop_vid is not None: hv = hv * (1 - drop_vid[:, None])
        return self.head(torch.cat([hi, hv], 1))

def predict(model, imu, loc, vid, bs=2048):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(imu), bs):
            out.append(F.softmax(model(torch.from_numpy(imu[i:i + bs]), torch.from_numpy(loc[i:i + bs]),
                                       torch.from_numpy(vid[i:i + bs].astype(np.float32))), 1).numpy())
    return np.concatenate(out)

# %%
EPOCHS, BS, SEEDS = 6, 256, 2
Xe = imu_input(prep_imu(tr_imu[np.arange(T), eval_loc], eval_loc)); Xte = imu_input(prep_imu(te_imu, te_loc))
oof = np.zeros((T, 19), np.float32); pte = np.zeros((N, 19), np.float32)
for k, (tri, vai) in enumerate(GroupKFold(5).split(tr_meta, y, sbj)):
  for seed in range(SEEDS):
      t0 = time.time(); torch.manual_seed(100 * seed + k)
      model = Net(); opt = torch.optim.AdamW(model.parameters(), 2e-3, weight_decay=0.05)
      steps = EPOCHS * math.ceil(len(tri) / BS); sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=steps)
      for ep in range(EPOCHS):
          model.train(); perm = rng.permutation(tri); tot = 0
          # one random valid sensor per tile per epoch
          loc = np.argsort(rng.random((T, 4)) + ~tr_valid, 1)[:, 0]
          for i in range(0, len(perm), BS):
              b = perm[i:i + BS]; lb = loc[b]
              xi = imu_input(augment(prep_imu(tr_imu[b, lb], lb)))
              xv = tr_vid[b].astype(np.float32); xv += rng.normal(0, 0.1, xv.shape).astype(np.float32)
              md = rng.random(len(b))
              logits = model(torch.from_numpy(xi), torch.from_numpy(lb), torch.from_numpy(xv),
                             torch.from_numpy((md < 0.1).astype(np.float32)), torch.from_numpy(((md > 0.1) & (md < 0.2)).astype(np.float32)))
              loss = F.cross_entropy(logits, torch.from_numpy(y[b]), label_smoothing=0.1)
              opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += loss.item() * len(b)
          if ep % 3 == 2 or ep == EPOCHS - 1:
              pv = predict(model, Xe[vai], eval_loc[vai], tr_vid[vai])
              print(f"fold {k} ep {ep} loss {tot/len(tri):.3f} F1 {f1_score(y[vai], pv.argmax(1), average='macro'):.4f} ({time.time()-t0:.0f}s)")
      oof[vai] += predict(model, Xe[vai], eval_loc[vai], tr_vid[vai]) / SEEDS
      pte += predict(model, Xte, te_loc, te_vid) / (5 * SEEDS)

# %%
def mf1(p, w0): q = p.copy(); q[:, 0] *= w0; return f1_score(y, q.argmax(1), average="macro")
print("OOF macro-F1:", round(mf1(oof, 1), 4))
scores = {w: round(mf1(oof, w), 4) for w in [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]}; print(scores)
best_w = max(scores, key=scores.get)
np.save(f"{OUT}/oof_nn2.npy", oof); np.save(f"{OUT}/te_nn2.npy", pte)
pt = pte.copy(); pt[:, 0] *= best_w
pd.DataFrame({"id": te_meta.id, "target_feature": pt.argmax(1)}).to_csv(f"{OUT}/submission.csv", index=False)
print("test pred dist:", np.bincount(pt.argmax(1), minlength=19) / N)
