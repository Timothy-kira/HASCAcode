# %% [markdown]
# # WEAR 2026 — Diagnostic: how much does timeline-graph quality limit stage2?
# Same stage2 features/model, but the train pair graph is synthetic with controlled successor precision
# (oracle = true t->t+1 edges). Compares stage2 OOF across precisions to locate the bottleneck.
# (original stage2 header follows)
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
def mf1(p, w0): q = p.copy(); q[:, 0] *= w0; return f1_score(y, q.argmax(1), average="macro")
def best_null(p): return max((round(mf1(p, w), 4), w) for w in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2])
params = dict(objective="multiclass", num_class=19, learning_rate=0.05, num_leaves=31, min_data_in_leaf=50,
              feature_fraction=0.3, bagging_fraction=0.8, bagging_freq=1, lambda_l2=2.0, verbose=-1, num_threads=os.cpu_count())
nxt_ok = np.r_[tr_meta.file_id.values[1:] == tr_meta.file_id.values[:-1], False]
real_top1 = tr_pairs.loc[tr_pairs.groupby("gi").p.idxmax()]
print("real graph: top1 successor precision", round(real_top1.target.mean(), 3))

def synthetic_pairs(prec, seed=0):
    """Each tile gets one successor edge: the true t+1 with prob `prec`, else a random real candidate of the pair model."""
    r = np.random.default_rng(seed)
    gi = np.where(nxt_ok)[0]
    wrong = tr_pairs[tr_pairs.target == 0].groupby("gi").gj.first()
    use_true = r.random(len(gi)) < prec
    gj = np.where(use_true, gi + 1, wrong.reindex(gi).fillna(-1).astype(int).values)
    keep = gj >= 0
    return pd.DataFrame({"gi": gi[keep], "gj": gj[keep], "p": 1.0})

def cv(pairs, tag):
    X, _ = context(P0_tr, Ftr, pairs, tr_vid, tr_sbj)
    oof2 = np.zeros((T, 19), np.float32)
    for k, (tri, vai) in enumerate(GroupKFold(5).split(X, y, tr_sbj)):
        m = lgb.train(params, lgb.Dataset(X[tri], y[tri]), 2000, valid_sets=[lgb.Dataset(X[vai], y[vai])],
                      callbacks=[lgb.early_stopping(80, verbose=False)])
        oof2[vai] = m.predict(X[vai], num_iteration=m.best_iteration)
    print(f"{tag}: stage2 OOF {best_null(oof2)}", flush=True)

print("base (no graph):", best_null(P0_tr))
cv(tr_pairs, "real soft graph (current)")
for prec in [1.0, 0.7, 0.5]:
    cv(synthetic_pairs(prec), f"synthetic successor precision {prec}")
