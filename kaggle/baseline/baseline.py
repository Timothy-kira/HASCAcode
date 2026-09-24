# %% [markdown]
# # WEAR 2026 — Step B: independent-window LightGBM baseline
# Hand-crafted IMU features (single sensor, location-aware) + VideoMAE summary features (PCA, subject-centred).
# Subject-grouped 5-fold CV on tiles built exactly like the test set. Saves OOF / test probabilities for later steps.

# %%
import glob, os, time
import numpy as np, pandas as pd, lightgbm as lgb
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import GroupKFold

P = os.path.dirname(glob.glob("/kaggle/input/**/tr_meta.csv", recursive=True)[0])
ROOT = os.path.dirname(glob.glob("/kaggle/input/**/sample_submission.csv", recursive=True)[0])
OUT = "/kaggle/working"
LOCS = ["left_arm", "left_leg", "right_arm", "right_leg"]
rng = np.random.default_rng(0)
print("cpus", os.cpu_count(), "| prep dir", P)

tr_meta = pd.read_csv(f"{P}/tr_meta.csv"); te_meta = pd.read_csv(f"{P}/te_meta.csv")
tr_imu = np.load(f"{P}/tr_imu.npy"); tr_valid = np.load(f"{P}/tr_valid.npy"); te_imu = np.load(f"{P}/te_imu.npy")
tr_vid = np.load(f"{P}/tr_vid.npy", mmap_mode="r"); te_vid = np.load(f"{P}/te_vid.npy", mmap_mode="r")
T, N = len(tr_meta), len(te_meta)
te_loc = te_meta.sensor_location.map({l: i for i, l in enumerate(LOCS)}).to_numpy()

# %%
def imu_feats(X, loc):
    """X: (n,50,3) single-sensor windows, loc: (n,) index into LOCS -> (n, F) features."""
    X = X.astype(np.float32).copy()
    X[loc == 0, :, 0] *= -1  # mirror left arm onto right arm (x axis sign flips, see EDA)
    mag = np.linalg.norm(X, axis=2, keepdims=True)
    A = np.concatenate([X, mag], 2)  # (n,50,4): x,y,z,|a|
    d = np.diff(A, axis=1)
    F = [A.mean(1), A.std(1), A.min(1), A.max(1), *np.percentile(A, [10, 25, 50, 75, 90], axis=1),
         stats.skew(A, 1), stats.kurtosis(A, 1), np.abs(d).mean(1), d.std(1),
         ((A[:, 1:] - A.mean(1, keepdims=True)) * (A[:, :-1] - A.mean(1, keepdims=True)) < 0).mean(1)]
    spec = np.abs(np.fft.rfft(A - A.mean(1, keepdims=True), axis=1)) ** 2  # (n,26,4), 1 Hz bins
    bands = [(1, 2), (2, 3), (3, 4), (4, 6), (6, 9), (9, 14), (14, 26)]
    tot = spec[:, 1:].sum(1) + 1e-8
    F += [spec[:, a:b].sum(1) / tot for a, b in bands] + [np.log(tot), spec[:, 1:].argmax(1).astype(np.float32)]
    m = X.mean(1); g = m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-8)
    C = np.stack([np.nan_to_num([np.corrcoef(w.T)[i, j] for i, j in [(0, 1), (0, 2), (1, 2)]]) for w in X])
    F += [g, C, np.eye(4, dtype=np.float32)[loc]]
    return np.concatenate([f.reshape(len(X), -1) for f in F], 1).astype(np.float32)

# %%
def vid_summary(V):
    """(n,15,768) -> per-tile mean, std over frames and first-to-last frame change."""
    out = [], [], []
    for i in range(0, len(V), 20000):
        Vf = np.asarray(V[i:i + 20000], dtype=np.float32)
        out[0].append(Vf.mean(1)); out[1].append(Vf.std(1))
        out[2].append(np.linalg.norm(Vf[:, -1] - Vf[:, 0], axis=1, keepdims=True))
    return [np.concatenate(o) for o in out]

def centre(mean, sbj):  # per-subject centring removes location/lighting offsets
    c = mean.copy()
    for s in np.unique(sbj): c[sbj == s] -= mean[sbj == s].mean(0)
    return c

t0 = time.time()
tr_mean, tr_std, tr_fl = vid_summary(tr_vid); te_mean, te_std, te_fl = vid_summary(te_vid)
tr_cen = centre(tr_mean, tr_meta.sbj.values); te_cen = centre(te_mean, te_meta.sbj_id.values)
pca_m = PCA(64, random_state=0).fit(np.r_[tr_mean, te_mean])
pca_c = PCA(64, random_state=0).fit(np.r_[tr_cen, te_cen])
pca_s = PCA(16, random_state=0).fit(np.r_[tr_std, te_std])
def vfeat(mean, cen, std, fl):
    return np.concatenate([pca_m.transform(mean), pca_c.transform(cen), pca_s.transform(std), fl], 1).astype(np.float32)
trVF = vfeat(tr_mean, tr_cen, tr_std, tr_fl); teVF = vfeat(te_mean, te_cen, te_std, te_fl)
print("video feats", trVF.shape, f"{time.time()-t0:.0f}s", "explained var (mean/cen):",
      pca_m.explained_variance_ratio_.sum().round(3), pca_c.explained_variance_ratio_.sum().round(3))

# %%
# Evaluation view: one random sensor per tile (like test). Training view: 2 random sensors per tile.
# Only sample sensors that were actually recorded (tr_valid masks e.g. sbj_10's missing left arm).
order = np.argsort(rng.random((T, 4)) + ~tr_valid, axis=1)  # valid sensors first, random order
eval_loc = order[:, 0]
tr_loc2 = np.where(tr_valid[np.arange(T)[:, None], order[:, :2]], order[:, :2], order[:, :1])
t0 = time.time()
F_eval = imu_feats(tr_imu[np.arange(T), eval_loc], eval_loc)
F_tr = [imu_feats(tr_imu[np.arange(T), tr_loc2[:, k]], tr_loc2[:, k]) for k in range(2)]
F_te = imu_feats(te_imu, te_loc)
print("imu feats", F_eval.shape, f"{time.time()-t0:.0f}s")
Xe = np.c_[F_eval, trVF]; Xte = np.c_[F_te, teVF]
Xtr_all = np.r_[np.c_[F_tr[0], trVF], np.c_[F_tr[1], trVF]]
y = tr_meta.label.values; y2 = np.r_[y, y]; sbj = tr_meta.sbj.values

# %%
params = dict(objective="multiclass", num_class=19, learning_rate=0.08, num_leaves=63, min_data_in_leaf=40,
              feature_fraction=0.4, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=os.cpu_count())
oof = np.zeros((T, 19), np.float32); pte = np.zeros((N, 19), np.float32); iters = []
for k, (tri, vai) in enumerate(GroupKFold(5).split(Xe, y, sbj)):
    t0 = time.time()
    tri2 = np.r_[tri, tri + T]
    dtr = lgb.Dataset(Xtr_all[tri2], y2[tri2]); dva = lgb.Dataset(Xe[vai], y[vai])
    m = lgb.train(params, dtr, 1500, valid_sets=[dva], callbacks=[lgb.early_stopping(60, verbose=False)])
    oof[vai] = m.predict(Xe[vai], num_iteration=m.best_iteration)
    pte += m.predict(Xte, num_iteration=m.best_iteration) / 5
    iters.append(m.best_iteration)
    print(f"fold {k}: sbj {sorted(set(sbj[vai]))} iters {m.best_iteration} "
          f"F1 {f1_score(y[vai], oof[vai].argmax(1), average='macro'):.4f} ({time.time()-t0:.0f}s)")
print("OOF macro-F1:", round(f1_score(y, oof.argmax(1), average="macro"), 4))

# %%
# Null-class scaling (null is 40% of tiles; macro-F1 rewards trading a bit of null recall)
best = (1.0, f1_score(y, oof.argmax(1), average="macro"))
for w in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.1, 1.3]:
    p = oof.copy(); p[:, 0] *= w
    f = f1_score(y, p.argmax(1), average="macro"); print(f"null x{w}: {f:.4f}")
    if f > best[1]: best = (w, f)
print("best null weight", best)
pf = oof.copy(); pf[:, 0] *= best[0]
print("per-class F1:", dict(enumerate(f1_score(y, pf.argmax(1), average=None).round(3))))
print("per-location F1:", {LOCS[l]: round(f1_score(y[eval_loc == l], pf[eval_loc == l].argmax(1), average="macro"), 4) for l in range(4)})

# %%
np.save(f"{OUT}/oof_lgb.npy", oof); np.save(f"{OUT}/te_lgb.npy", pte); np.save(f"{OUT}/eval_loc.npy", eval_loc)
pt = pte.copy(); pt[:, 0] *= best[0]
sub = pd.DataFrame({"id": te_meta.id, "target_feature": pt.argmax(1).astype(int)})
sub.to_csv(f"{OUT}/submission.csv", index=False)
print(sub.target_feature.value_counts(normalize=True).sort_index().round(3).to_dict())
