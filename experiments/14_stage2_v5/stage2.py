# %% [markdown]
# # WEAR 2026 — Step D3: graph-context stacking
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
print("base avg", best_null(P0_tr))
params = dict(objective="multiclass", num_class=19, learning_rate=0.05, num_leaves=31, min_data_in_leaf=50,
              feature_fraction=0.3, bagging_fraction=0.8, bagging_freq=1, lambda_l2=2.0, verbose=-1, num_threads=os.cpu_count())

def stack_round(Ptr, Pte, r):
    t0 = time.time()
    Xtr, names = context(Ptr, Ftr, tr_pairs, tr_vid, tr_sbj); Xte, _ = context(Pte, Fte, te_pairs, te_vid, te_sbj)
    print(f"round {r}: feats {Xtr.shape} ({time.time()-t0:.0f}s)")
    oof2 = np.zeros((T, 19), np.float32); pte2 = np.zeros((N, 19), np.float32)
    for k, (tri, vai) in enumerate(GroupKFold(5).split(Xtr, y, tr_sbj)):
        t0 = time.time()
        m = lgb.train(params, lgb.Dataset(Xtr[tri], y[tri]), 2000, valid_sets=[lgb.Dataset(Xtr[vai], y[vai])],
                      callbacks=[lgb.early_stopping(80, verbose=False)])
        oof2[vai] = m.predict(Xtr[vai], num_iteration=m.best_iteration); pte2 += m.predict(Xte, num_iteration=m.best_iteration) / 5
        print(f"  fold {k} iters {m.best_iteration} F1 {f1_score(y[vai], oof2[vai].argmax(1), average='macro'):.4f} ({time.time()-t0:.0f}s)")
    print(f"round {r} OOF", best_null(oof2))
    imp = pd.Series(m.feature_importance("gain"), names); print(imp.groupby(imp.index.str.rsplit("_", n=1).str[0]).sum().sort_values(ascending=False).round(0).to_dict())
    return oof2, pte2

oof2, pte2 = stack_round(P0_tr, P0_te, 1)
for r in range(2, ROUNDS + 1):  # extra rounds did not help (v2: 0.7767 -> 0.7605)
    o, t = stack_round(oof2, pte2, r)
    if best_null(o)[0] <= best_null(oof2)[0]: print("no gain, stop"); break
    oof2, pte2 = o, t

# %%
# Final propagation of stage-2 output over the soft pair graph
def propagate(P2, pairs, alpha, iters):
    _, _, Wb = pair_graphs(pairs, len(P2)); Q = P2.copy()
    for _ in range(iters): Q = (1 - alpha) * P2 + alpha * (Wb @ Q)
    return Q
res = [(best_null(propagate(oof2, tr_pairs, a, it)), a, it) for a in [0.3, 0.5, 0.7] for it in [1, 3, 5]]
res.sort(reverse=True); print(res[:5])
(f, w0), a, it = res[0]
if f <= best_null(oof2)[0]: a, it, w0 = 0.0, 0, best_null(oof2)[1]
q = propagate(pte2, te_pairs, a, it) if it else pte2.copy(); q[:, 0] *= w0
np.save(f"{OUT}/oof_stage2.npy", oof2); np.save(f"{OUT}/te_stage2.npy", pte2)
pd.DataFrame({"id": te_meta.id, "target_feature": q.argmax(1)}).to_csv(f"{OUT}/submission.csv", index=False)
print("final: alpha", a, "iters", it, "null_w", w0, "| test dist", (np.bincount(q.argmax(1), minlength=19) / N).round(3))
