# %% [markdown]
# # WEAR 2026 — Diagnostic v2: how much long-range (segment-level) context is worth, and can clustering deliver it?
# (A) ceiling: perfect successor graph with hops up to 64; (B) ceiling: oracle vote over the true activity segment;
# (C) realistic: per-subject spectral clustering of (pair-prob graph + video kNN), cluster-mean probs as stage2 features.
# (stage2 header follows)
#
# Each tile's neighbours in time (soft successor/predecessor graph from the learned pair model, plus a video kNN graph)
# usually carry a *different* sensor, so aggregating their IMU features and base-model probabilities gives a
# multi-limb, multi-second view of the activity. A stage-2 LightGBM learns from [own probs, own IMU feats,
# graph-aggregated probs & IMU feats (1 and 2 hops)]. Subject-grouped CV; base probs are already OOF.

# %%
import glob, os, time
import numpy as np, pandas as pd, lightgbm as lgb, scipy.sparse as sp
from scipy import stats
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold

def find(name): return os.path.dirname(glob.glob(f"/kaggle/input/**/{name}", recursive=True)[0])
P, B, C = find("tr_meta.csv"), find("oof_lgb.npy"), find("tr_pairs.parquet")
HOPS, ROUNDS = [1, 2, 4], 1  # v2 with [1..16] hops and 2 rounds was worse (0.783 vs 0.787)
OUT = "/kaggle/working"; LOCS = ["left_arm", "left_leg", "right_arm", "right_leg"]
tr_meta = pd.read_csv(f"{P}/tr_meta.csv"); te_meta = pd.read_csv(f"{P}/te_meta.csv")
tr_imu = np.load(f"{P}/tr_imu.npy"); te_imu = np.load(f"{P}/te_imu.npy")
tr_vid = np.load(f"{P}/tr_vid.npy", mmap_mode="r"); te_vid = np.load(f"{P}/te_vid.npy", mmap_mode="r")
eval_loc = np.load(f"{B}/eval_loc.npy")
te_loc = te_meta.sensor_location.map({l: i for i, l in enumerate(LOCS)}).to_numpy()
y = tr_meta.label.values; T, N = len(tr_meta), len(te_meta)
tr_sbj, te_sbj = tr_meta.sbj.values, te_meta.sbj_id.values

# base probabilities: every *oof_*.npy / te_*.npy pair found in inputs (lgb baseline, nn fusion, ...)
bases = []
for f in sorted(glob.glob("/kaggle/input/**/oof_*.npy", recursive=True)):
    name = os.path.basename(f)[4:-4]; tf = os.path.join(os.path.dirname(f), f"te_{name}.npy")
    if os.path.exists(tf): bases.append((name, np.load(f), np.load(tf)))
print("base models:", [b[0] for b in bases])
P0_tr = np.mean([b[1] for b in bases], 0); P0_te = np.mean([b[2] for b in bases], 0)
tr_pairs = pd.read_parquet(f"{C}/tr_pairs.parquet"); te_pairs = pd.read_parquet(f"{C}/te_pairs.parquet")

# %%
def imu_feats(X, loc):  # same as baseline
    X = X.astype(np.float32).copy(); X[loc == 0, :, 0] *= -1
    mag = np.linalg.norm(X, axis=2, keepdims=True); A = np.concatenate([X, mag], 2); d = np.diff(A, axis=1)
    F = [A.mean(1), A.std(1), A.min(1), A.max(1), *np.percentile(A, [10, 25, 50, 75, 90], axis=1),
         stats.skew(A, 1), stats.kurtosis(A, 1), np.abs(d).mean(1), d.std(1),
         ((A[:, 1:] - A.mean(1, keepdims=True)) * (A[:, :-1] - A.mean(1, keepdims=True)) < 0).mean(1)]
    spec = np.abs(np.fft.rfft(A - A.mean(1, keepdims=True), axis=1)) ** 2
    bands = [(1, 2), (2, 3), (3, 4), (4, 6), (6, 9), (9, 14), (14, 26)]; tot = spec[:, 1:].sum(1) + 1e-8
    F += [spec[:, a:b].sum(1) / tot for a, b in bands] + [np.log(tot), spec[:, 1:].argmax(1).astype(np.float32)]
    m = X.mean(1); g = m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-8)
    F += [g, np.eye(4, dtype=np.float32)[loc]]
    return np.nan_to_num(np.concatenate([f.reshape(len(X), -1) for f in F], 1).astype(np.float32))

t0 = time.time()
Ftr = imu_feats(tr_imu[np.arange(T), eval_loc], eval_loc); Fte = imu_feats(te_imu, te_loc)
print("imu feats", Ftr.shape, f"{time.time()-t0:.0f}s")

# %%
def norm(a): return a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)
def knn_graph(V, sbj, k=10):
    rows, cols = [], []
    for s in np.unique(sbj):
        idx = np.where(sbj == s)[0]; M = np.asarray(V[idx], dtype=np.float32).mean(1); mu = M.mean(0)
        _, S, Vt = np.linalg.svd(M - mu, full_matrices=False); W = Vt[:256].T / (S[:256] / np.sqrt(len(M)) + 1e-6)
        E = norm((M - mu) @ W); Sim = E @ E.T; np.fill_diagonal(Sim, -np.inf)
        nb = np.argpartition(-Sim, k, 1)[:, :k]
        rows.append(np.repeat(idx, k)); cols.append(idx[nb.ravel()])
    n = len(sbj); A = sp.csr_matrix((np.ones(sum(map(len, rows))), (np.concatenate(rows), np.concatenate(cols))), (n, n))
    return rownorm(A)

def rownorm(A):
    d = np.asarray(A.sum(1)).ravel(); d[d == 0] = 1
    return sp.diags(1 / d) @ A

def pair_graphs(pairs, n):
    W = sp.csr_matrix((pairs.p.values, (pairs.gi.values, pairs.gj.values)), (n, n))
    return rownorm(W), rownorm(W.T.tocsr()), rownorm(W + W.T)  # successor, predecessor, both

def hops(W, X, ks):
    out, cur, done = {}, X, 0
    for k in ks:
        while done < k: cur = W @ cur; done += 1
        out[k] = cur
    return out

def context(P0, Fimu, pairs, V, sbj):
    n = len(P0); Ws, Wp, Wb = pair_graphs(pairs, n); Wk = knn_graph(V, sbj)
    L = np.log(P0 + 1e-6); hb = hops(Wb, P0, HOPS)
    blocks = {"own_p": P0, "own_imu": Fimu,
              "succ_p": Ws @ P0, "pred_p": Wp @ P0, **{f"both{k}_p": hb[k] for k in HOPS},
              "knn_p": Wk @ P0, "knn2_p": Wk @ (Wk @ P0), "both_logp": Wb @ L,
              "both_imu": Wb @ Fimu, "both2_imu": Wb @ (Wb @ Fimu), "knn_imu": Wk @ Fimu,
              "deg": np.c_[np.asarray((Ws > 0).sum(1)).ravel(), np.asarray((Wp > 0).sum(1)).ravel()]}
    if "assigned" in pairs:  # hard one-to-one successor edges from the Hungarian assignment (chain v3+)
        a = pairs[pairs.assigned == 1]
        Wa = sp.csr_matrix((np.ones(len(a)), (a.gi.values, a.gj.values)), (n, n))
        ha = hops(rownorm(Wa + Wa.T), P0, [1, 2, 4, 8, 16])
        blocks.update({f"asg{k}_p": ha[k] for k in ha})
        blocks["asg_succ_p"] = rownorm(Wa) @ P0; blocks["asg_pred_p"] = rownorm(Wa.T.tocsr()) @ P0
    names = [f"{k}_{i}" for k, v in blocks.items() for i in range(v.shape[1])]
    return np.concatenate([np.asarray(v, dtype=np.float32) for v in blocks.values()], 1), names

# %%
from sklearn.cluster import SpectralClustering
def mf1(p, w0): q = p.copy(); q[:, 0] *= w0; return f1_score(y, q.argmax(1), average="macro")
def best_null(p): return max((round(mf1(p, w), 4), w) for w in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2])
params = dict(objective="multiclass", num_class=19, learning_rate=0.05, num_leaves=31, min_data_in_leaf=50,
              feature_fraction=0.3, bagging_fraction=0.8, bagging_freq=1, lambda_l2=2.0, verbose=-1, num_threads=os.cpu_count())
file_id = tr_meta.file_id.values
nxt_ok = np.r_[file_id[1:] == file_id[:-1], False]
seg = np.r_[0, np.cumsum((y[1:] != y[:-1]) | (file_id[1:] != file_id[:-1]))]  # true activity segments (label runs)
print("true segments:", seg.max() + 1, "| mean length (s):", round(T / (seg.max() + 1), 1))

def cv(X, tag):
    t0 = time.time(); oof2 = np.zeros((T, 19), np.float32)
    for k, (tri, vai) in enumerate(GroupKFold(5).split(X, y, tr_sbj)):
        m = lgb.train(params, lgb.Dataset(X[tri], y[tri]), 2000, valid_sets=[lgb.Dataset(X[vai], y[vai])],
                      callbacks=[lgb.early_stopping(80, verbose=False)])
        oof2[vai] = m.predict(X[vai], num_iteration=m.best_iteration)
    print(f"{tag}: stage2 OOF {best_null(oof2)} ({time.time()-t0:.0f}s)", flush=True)
    return oof2

def group_mean(P, groups):
    """Per-tile mean of P over its group, plus log group size."""
    g = pd.factorize(groups)[0]; cnt = np.bincount(g)
    S = np.zeros((cnt.size, P.shape[1])); np.add.at(S, g, P)
    return np.c_[S[g] / cnt[g, None], np.log(cnt[g])[:, None]].astype(np.float32)

print("base (no graph):", best_null(P0_tr))

# %%
# Real graph (chain v3 pairs, incl. assigned edges), hops 1-4 — reference
HOPS = [1, 2, 4]
X_real, _ = context(P0_tr, Ftr, tr_pairs, tr_vid, tr_sbj)
cv(X_real, "real graph, hops 1-4")

# %%
# (A) perfect successor graph, long hops
gi = np.where(nxt_ok)[0]
perfect = pd.DataFrame({"gi": gi, "gj": gi + 1, "p": 1.0})
HOPS = [1, 2, 4, 8, 16, 32, 64]
X_perf, _ = context(P0_tr, Ftr, perfect, tr_vid, tr_sbj)
cv(X_perf, "(A) perfect graph, hops 1-64")
HOPS = [1, 2, 4]

# %%
# (B) oracle: mean base probs over the true segment (uses labels -> ceiling only)
cv(np.c_[X_real, group_mean(P0_tr, seg)], "(B) real graph + oracle segment vote")

# %%
# (C) realistic: spectral clustering per subject on pair-prob graph + video kNN graph
Wk_all = knn_graph(tr_vid, tr_sbj)
_, _, Wb_all = pair_graphs(tr_pairs, T)
def clusters(sbj_arr, Wb, Wk, div):
    lab = np.zeros(len(sbj_arr), int); off = 0
    for s in np.unique(sbj_arr):
        idx = np.where(sbj_arr == s)[0]
        A = Wb[idx][:, idx] + 0.5 * Wk[idx][:, idx]; A = ((A + A.T) / 2).tocsr()
        k = max(5, len(idx) // div)
        c = SpectralClustering(k, affinity="precomputed", assign_labels="cluster_qr", random_state=0).fit_predict(A)
        lab[idx] = c + off; off += k
    return lab

feats = []
for div in [20, 40, 80]:
    t0 = time.time(); cl = clusters(tr_sbj, Wb_all, Wk_all, div)
    maj = pd.Series(y).groupby(cl).agg(lambda v: v.value_counts().iloc[0] / len(v))
    size = pd.Series(cl).value_counts()
    nseg = pd.Series(seg).groupby(cl).nunique()
    print(f"div {div}: clusters {cl.max()+1}, mean size {size.mean():.1f}, tile-weighted purity {(maj * size.reindex(maj.index)).sum() / T:.3f}, "
          f"mean true segments per cluster {nseg.mean():.2f} ({time.time()-t0:.0f}s)", flush=True)
    feats.append(group_mean(P0_tr, cl))
    cv(np.c_[X_real, feats[-1]], f"(C) real graph + clusters div {div}")
cv(np.c_[X_real, *feats], "(C) real graph + clusters div 20/40/80")
