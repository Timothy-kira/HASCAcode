# %% [markdown]
# # WEAR 2026 — Step D2 (v3): learned successor model -> global assignment
# 1. Per subject, whitened VideoMAE embeddings of each tile's first/last/mean frames.
# 2. A ridge "next-frame predictor" (trained on other subjects' consecutive tiles, subject-grouped folds) predicts
#    tile t+1's first frame from tile t; cos(prediction, first frame of j) is a dynamics-aware match score.
# 3. Candidates = union of top-K by plain last->first similarity and by predicted similarity.
# 4. Pair features -> LightGBM "is j the successor of i" (subject-grouped OOF).
# 5. Global one-to-one assignment per subject (Hungarian on -log p) -> hard successor edges.
# Outputs tr_pairs / te_pairs (gi, gj, p, assigned) consumed by stage2.

# %%
import glob, os, time
import numpy as np, pandas as pd, lightgbm as lgb
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import Ridge
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold

def find(name): return os.path.dirname(glob.glob(f"/kaggle/input/**/{name}", recursive=True)[0])
P, B = find("tr_meta.csv"), find("oof_lgb.npy")
OUT = "/kaggle/working"; LOCS = ["left_arm", "left_leg", "right_arm", "right_leg"]
K, NC = 40, 256
tr_meta = pd.read_csv(f"{P}/tr_meta.csv"); te_meta = pd.read_csv(f"{P}/te_meta.csv")
tr_imu = np.load(f"{P}/tr_imu.npy"); te_imu = np.load(f"{P}/te_imu.npy")
tr_vid = np.load(f"{P}/tr_vid.npy", mmap_mode="r"); te_vid = np.load(f"{P}/te_vid.npy", mmap_mode="r")
eval_loc = np.load(f"{B}/eval_loc.npy"); oof = np.load(f"{B}/oof_lgb.npy"); pte = np.load(f"{B}/te_lgb.npy")
te_loc = te_meta.sensor_location.map({l: i for i, l in enumerate(LOCS)}).to_numpy()
tr_x = tr_imu[np.arange(len(tr_meta)), eval_loc].astype(np.float32)
tr_sbj, te_sbj = tr_meta.sbj.values, te_meta.sbj_id.values
nxt_ok = np.r_[tr_meta.file_id.values[1:] == tr_meta.file_id.values[:-1], False]  # tile i has a true successor i+1

# %%
def norm(a): return a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)

def subject_emb(V):
    V = np.asarray(V, dtype=np.float32); mean = V.mean(1); mu = mean.mean(0)
    _, S, Vt = np.linalg.svd(mean - mu, full_matrices=False)
    W = Vt[:NC].T / (S[:NC] / np.sqrt(len(V)) + 1e-6)
    w = lambda x: (x - mu) @ W
    E = dict(first=w(V[:, 0]), last=w(V[:, -1]), f1=w(V[:, 1]), l1=w(V[:, -2]), mean=w(mean), l5=w(V[:, -5]))
    E["rf"], E["rl"] = norm(V[:, 0] - mu), norm(V[:, -1] - mu)
    return E

def ridge_in(E): return np.c_[E["last"], E["last"] - E["l5"], E["mean"]]

t0 = time.time()
tr_E = {s: subject_emb(tr_vid[tr_sbj == s]) for s in np.unique(tr_sbj)}
te_E = {s: subject_emb(te_vid[te_sbj == s]) for s in np.unique(te_sbj)}
print("embeddings", f"{time.time()-t0:.0f}s")

def fit_ridge(subjects):
    X, Y = [], []
    for s in subjects:
        E = tr_E[s]; ok = nxt_ok[tr_sbj == s]
        X.append(ridge_in(E)[ok]); Y.append(E["first"][np.r_[False, ok[:-1]]])
    return Ridge(alpha=10.0).fit(np.concatenate(X), np.concatenate(Y))

subjects = np.unique(tr_sbj)
fold_of = {s: k for k, (_, va) in enumerate(GroupKFold(5).split(subjects, groups=subjects)) for s in subjects[va]}
ridges = {k: fit_ridge([s for s in subjects if fold_of[s] != k]) for k in range(5)}
ridge_all = fit_ridge(subjects)

# %%
def subject_pairs(E, X, loc, P0, ridge):
    n = len(X)
    first, last = norm(E["first"]), norm(E["last"])
    pred = norm(ridge.predict(ridge_in(E)))
    S1 = last @ first.T; np.fill_diagonal(S1, -np.inf)
    S2 = pred @ first.T; np.fill_diagonal(S2, -np.inf)
    c1 = np.argpartition(-S1, K, 1)[:, :K]; c2 = np.argpartition(-S2, K, 1)[:, :K]
    cand = pd.DataFrame({"i": np.repeat(np.arange(n), 2 * K), "j": np.c_[c1, c2].ravel()}).drop_duplicates()
    i, j = cand.i.values, cand.j.values
    r1 = np.argsort(np.argsort(-S1, 1), 1); c1r = np.argsort(np.argsort(-S1, 0), 0)
    r2 = np.argsort(np.argsort(-S2, 1), 1); c2r = np.argsort(np.argsort(-S2, 0), 0)
    endx = X[:, -1] + (X[:, -1] - X[:, -3]) / 2; mag = np.linalg.norm(X, axis=2); same = (loc[i] == loc[j]).astype(np.float32)
    f1_, l1_, m = norm(E["f1"]), norm(E["l1"]), norm(E["mean"])
    feats = dict(
        s1=S1[i, j], s2=S2[i, j], r1_row=r1[i, j], r1_col=c1r[i, j], r2_row=r2[i, j], r2_col=c2r[i, j],
        gap1_row=S1[i].max(1) - S1[i, j], gap2_row=S2[i].max(1) - S2[i, j],
        gap1_col=np.where(np.isfinite(S1), S1, -1).max(0)[j] - S1[i, j], gap2_col=np.where(np.isfinite(S2), S2, -1).max(0)[j] - S2[i, j],
        s_inner=(l1_[i] * f1_[j]).sum(1), s_mean=(m[i] * m[j]).sum(1), s_raw=(E["rl"][i] * E["rf"][j]).sum(1),
        s_self_i=(first[i] * last[i]).sum(1), s_self_j=(first[j] * last[j]).sum(1),
        same_loc=same, loc_i=loc[i], loc_j=loc[j],
        imu_gap=np.where(same > 0, np.linalg.norm(endx[i] - X[j, 0], axis=1), np.nan),
        mag_mean_diff=np.abs(mag[i].mean(1) - mag[j].mean(1)), mag_std_diff=np.abs(mag[i].std(1) - mag[j].std(1)),
        mag_edge_diff=np.abs(mag[i, -5:].mean(1) - mag[j, :5].mean(1)),
        p_dot=(P0[i] * P0[j]).sum(1), p_null_i=P0[i, 0], p_null_j=P0[j, 0], p_l1=np.abs(P0[i] - P0[j]).sum(1),
    )
    return i, j, pd.DataFrame(feats).astype(np.float32), (S1, S2)

# %%
t0 = time.time(); rows = []; retr = []
for s in subjects:
    idx = np.where(tr_sbj == s)[0]
    i, j, Fdf, (S1, S2) = subject_pairs(tr_E[s], tr_x[idx], eval_loc[idx], oof[idx], ridges[fold_of[s]])
    ok = nxt_ok[idx][:-1]; ar = np.arange(len(idx) - 1)
    retr.append([(S1[ar, ar + 1][:, None] >= S1[ar]).all(1)[ok].mean(), (S2[ar, ar + 1][:, None] >= S2[ar]).all(1)[ok].mean()])
    gi, gj = idx[i], idx[j]
    Fdf["target"] = ((gj == gi + 1) & nxt_ok[gi]).astype(np.int8); Fdf["gi"], Fdf["gj"], Fdf["sbj"] = gi, gj, s
    rows.append(Fdf)
pairs = pd.concat(rows, ignore_index=True)
FEATS = [c for c in pairs.columns if c not in ("target", "gi", "gj", "sbj")]
print("raw top1: plain", np.mean(retr, 0)[0].round(3), "| ridge-predicted", np.mean(retr, 0)[1].round(3))
print("pairs", pairs.shape, "recall@cand", round(pairs.target.sum() / nxt_ok.sum(), 3), f"{time.time()-t0:.0f}s")

# %%
params = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=100, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=os.cpu_count())
pairs["p"] = 0.0
for k in range(5):
    va = pairs.sbj.map(fold_of).values == k
    m = lgb.train(params, lgb.Dataset(pairs.loc[~va, FEATS], pairs.target.values[~va]), 600)
    pairs.loc[va, "p"] = m.predict(pairs.loc[va, FEATS])
print("pair AUC", round(roc_auc_score(pairs.target, pairs.p), 4))
top = pairs.loc[pairs.groupby("gi").p.idxmax()]
print("reranked top1 successor acc (all tiles):", round(top.target.sum() / nxt_ok.sum(), 4))
print(pd.Series(m.feature_importance("gain"), FEATS).sort_values(ascending=False).round(0).to_dict())
final_pair_model = lgb.train(params, lgb.Dataset(pairs[FEATS], pairs.target.values), 600)

# %%
def assign(df, n_by_sbj):
    """Per subject Hungarian matching on -log p over candidate pairs (non-candidates: high cost)."""
    df = df.copy(); df["assigned"] = 0
    for s, g in df.groupby("sbj"):
        nodes = np.unique(np.r_[g.gi.values, g.gj.values]); pos = {v: k for k, v in enumerate(nodes)}; n = len(nodes)
        C = np.full((n, n), 12.0, np.float32)
        a = np.vectorize(pos.get)(g.gi.values); b = np.vectorize(pos.get)(g.gj.values)
        C[a, b] = -np.log(g.p.values + 1e-5)
        r, c = linear_sum_assignment(C)
        ok = C[r, c] < 12.0
        key = pd.MultiIndex.from_arrays([nodes[r[ok]], nodes[c[ok]]])
        hit = pd.MultiIndex.from_arrays([g.gi.values, g.gj.values]).isin(key)
        df.loc[g.index[hit], "assigned"] = 1
    return df

t0 = time.time()
pairs = assign(pairs, None)
A = pairs[pairs.assigned == 1]
print(f"assignment ({time.time()-t0:.0f}s): edges {len(A)} precision {A.target.mean():.4f} recall {A.target.sum()/nxt_ok.sum():.4f}")
for thr in [0.05, 0.1, 0.2, 0.3]:
    a = A[A.p >= thr]; print(f"  assigned & p>={thr}: edges {len(a)} precision {a.target.mean():.4f} recall {a.target.sum()/nxt_ok.sum():.4f}")

# %%
rows = []
for s in np.unique(te_sbj):
    idx = np.where(te_sbj == s)[0]
    i, j, Fdf, _ = subject_pairs(te_E[s], te_imu[idx].astype(np.float32), te_loc[idx], pte[idx], ridge_all)
    Fdf["gi"], Fdf["gj"], Fdf["sbj"] = idx[i], idx[j], s; rows.append(Fdf)
tp = pd.concat(rows, ignore_index=True); tp["p"] = final_pair_model.predict(tp[FEATS])
tp = assign(tp, None)
print("test pairs", tp.shape, "assigned", int(tp.assigned.sum()), "mean p of assigned", tp.p[tp.assigned == 1].mean().round(3),
      "| train mean p of assigned", A.p.mean().round(3))
pairs[["gi", "gj", "p", "assigned", "target"]].to_parquet(f"{OUT}/tr_pairs.parquet")
tp[["gi", "gj", "p", "assigned"]].to_parquet(f"{OUT}/te_pairs.parquet")
