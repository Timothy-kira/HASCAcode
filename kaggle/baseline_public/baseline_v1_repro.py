# %% [markdown]
# # WEAR 2026 (3rd WEAR Dataset Challenge @ HASCA) — baseline v1, fully reproducible
# Self-contained: reads only the competition data and regenerates the three files that the later stacking stages
# discover by name in `/kaggle/input`:
#
# | output | content |
# |---|---|
# | `eval_loc.npy` | (69326,) the one random sensor (0 left_arm, 1 left_leg, 2 right_arm, 3 right_leg) each 1 s train tile is evaluated with |
# | `oof_lgb.npy` | (69326, 19) subject-grouped 5-fold out-of-fold class probabilities |
# | `te_lgb.npy` | (12234, 19) test class probabilities (mean of the 5 fold models) |
#
# The `_lgb` suffix matters: a downstream stage pairs every `oof_<name>.npy` with `te_<name>.npy`.
# Part 1 cuts the training recordings into non-overlapping 1 s tiles that mirror the test construction
# (50 IMU rows per sensor, central 15 of the 30 VideoMAE frames); part 2 is the LightGBM baseline
# (102 hand-crafted single-sensor IMU features + VideoMAE PCA features, null probability x0.3).
# Original run: OOF macro-F1 0.634 (0.6343 after null x0.3), public LB 0.64641. The seed is fixed (`default_rng(0)`),
# so `eval_loc.npy` is identical on every run; LightGBM multi-threading can change the probabilities in the last digits.

# %% [markdown]
# ## Part 1 — tile preparation
# Cuts every training recording into non-overlapping 1 s tiles that mirror the test construction
# (50 IMU rows per sensor, central 15 of 30 VideoMAE frames) and re-packs the test set in the same layout.
# Tiles are kept in a scratch directory (`/kaggle/temp`) for part 2 and are not part of the notebook outputs:
# `tr_imu.npy (T,4,50,3)`, `tr_vid.npy (T,15,768) f16`, `tr_meta.csv`, `te_imu.npy (N,50,3)`, `te_vid.npy (N,15,768) f16`, `te_meta.csv`.

# %%
import glob, os, re, time
import numpy as np, pandas as pd

ROOT = os.path.dirname(glob.glob("/kaggle/input/**/sample_submission.csv", recursive=True)[0])
OUT = "/kaggle/temp/wear_prep"; os.makedirs(OUT, exist_ok=True)  # scratch only: not part of the notebook outputs
LOCS = ["left_arm", "left_leg", "right_arm", "right_leg"]
LABELS = ["null", "jogging", "jogging (rotating arms)", "jogging (skipping)", "jogging (sidesteps)", "jogging (butt-kicks)",
          "stretching (triceps)", "stretching (lunging)", "stretching (shoulders)", "stretching (hamstrings)",
          "stretching (lumbar rotation)", "push-ups", "push-ups (complex)", "sit-ups", "sit-ups (complex)",
          "burpees", "lunges", "lunges (complex)", "bench-dips"]
LAB2ID = {l: i for i, l in enumerate(LABELS)}
V0, V1 = 8, 23  # frames 8..22 of each 30-frame second: the clip (i-8..i+7) stays inside the second

# %%
files = sorted(glob.glob(f"{ROOT}/train/inertial_feat/*.csv"), key=lambda f: [int(x) for x in re.findall(r"\d+", os.path.basename(f))])
imus, vids, metas = [], [], []
for fid, f in enumerate(files):
    t0 = time.time()
    name = os.path.basename(f)[:-4]
    df = pd.read_csv(f, dtype={"label": str})
    cols = [f"{l}_acc_{a}" for l in LOCS for a in "xyz"]
    nan = df[cols].isna().sum()
    if nan.sum():
        print(name, "NaN per column:", nan[nan > 0].to_dict(), "longest NaN run:",
              int(df[cols[0]].isna().astype(int).groupby(df[cols[0]].notna().cumsum()).sum().max()))
        df[cols] = df[cols].interpolate(limit_direction="both")
    lab = df["label"].map(LAB2ID).fillna(0).astype(int).to_numpy()
    assert df["label"].dropna().isin(LAB2ID).all()
    v = np.load(f"{ROOT}/train/videomae_feat/{name}.npy", mmap_mode="r")
    T = min(len(df) // 50, v.shape[0] // 30)
    imu = df[cols].to_numpy(np.float32)[: T * 50].reshape(T, 50, 4, 3).transpose(0, 2, 1, 3)
    vid = np.asarray(v[: T * 30], dtype=np.float32).reshape(T, 30, 768)[:, V0:V1].astype(np.float16)
    lt = lab[: T * 50].reshape(T, 50)
    counts = np.stack([(lt == k).sum(1) for k in range(19)], 1)
    metas.append(pd.DataFrame(dict(file=name, file_id=fid, sbj=int(df.sbj_id.iloc[0]), t=np.arange(T),
                                   label=counts.argmax(1), label_frac=counts.max(1) / 50, label_center=lt[:, 25])))
    imus.append(imu); vids.append(vid)
    print(f"{name}: T={T} ({time.time()-t0:.0f}s)")

tr_meta = pd.concat(metas, ignore_index=True)
np.save(f"{OUT}/tr_imu.npy", np.concatenate(imus)); np.save(f"{OUT}/tr_vid.npy", np.concatenate(vids))
tr_meta.to_csv(f"{OUT}/tr_meta.csv", index=False)
del imus
print(tr_meta.shape, "pure tiles:", (tr_meta.label_frac == 1).mean().round(3),
      "| mode==center:", (tr_meta.label == tr_meta.label_center).mean().round(4))
print(tr_meta.label.value_counts().sort_index().to_dict())

# %%
te_meta = pd.read_csv(f"{ROOT}/test/test_meta_data.csv")
te_imu = np.load(f"{ROOT}/test/test_inertial_data.npy").astype(np.float32)
te_vid = np.load(f"{ROOT}/test/test_videomae_data.npy", mmap_mode="r")
assert te_vid.shape[1:] == (768, 15), te_vid.shape
te_vid = np.ascontiguousarray(np.asarray(te_vid).transpose(0, 2, 1)).astype(np.float16)
assert (te_meta.id.values == np.arange(len(te_meta))).all()
np.save(f"{OUT}/te_imu.npy", te_imu); np.save(f"{OUT}/te_vid.npy", te_vid)
te_meta.to_csv(f"{OUT}/te_meta.csv", index=False)
print("test", te_imu.shape, te_vid.shape, te_meta.sbj_id.value_counts().to_dict())

# %% [markdown]
# ## Part 2 — independent-window LightGBM baseline
# Hand-crafted IMU features (single sensor, location-aware) + VideoMAE summary features (PCA, subject-centred).
# Subject-grouped 5-fold CV on tiles built exactly like the test set. Saves `oof_lgb.npy`, `te_lgb.npy`, `eval_loc.npy` (and the baseline `submission.csv`).

# %%
import glob, os, time
import numpy as np, pandas as pd, lightgbm as lgb
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.model_selection import GroupKFold

P = "/kaggle/temp/wear_prep"  # tiles built in part 1
ROOT = os.path.dirname(glob.glob("/kaggle/input/**/sample_submission.csv", recursive=True)[0])
OUT = "/kaggle/working"
LOCS = ["left_arm", "left_leg", "right_arm", "right_leg"]
rng = np.random.default_rng(0)
print("cpus", os.cpu_count(), "| prep dir", P)

tr_meta = pd.read_csv(f"{P}/tr_meta.csv"); te_meta = pd.read_csv(f"{P}/te_meta.csv")
tr_imu = np.load(f"{P}/tr_imu.npy"); te_imu = np.load(f"{P}/te_imu.npy")
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
eval_loc = rng.integers(0, 4, T)
tr_loc2 = np.stack([rng.permutation(4)[:2] for _ in range(T)])
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
