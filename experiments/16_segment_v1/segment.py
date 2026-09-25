# %% [markdown]
# # WEAR 2026 — Step E: supervised "same activity" neighbour model (segment recovery)
# diag v2 showed an oracle vote over the true activity segment lifts stage2 to 0.871 (vs 0.78-0.81), while
# unsupervised clusters are too small/impure. Here, for each tile's M nearest within-subject video neighbours, a
# LightGBM pair model predicts P(same label) from video / IMU / base-probability agreement. Base probabilities are
# then aggregated over neighbours with learned weights (1 and 2 hops) -> segment-context features for stage2.
# Outputs: tr_segctx.npy / te_segctx.npy (+ names), tr_seg_pairs.parquet.

# %%
import glob, os, time
import numpy as np, pandas as pd, lightgbm as lgb, scipy.sparse as sp
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold

def find(name): return os.path.dirname(glob.glob(f"/kaggle/input/**/{name}", recursive=True)[0])
P, B, C = find("tr_meta.csv"), find("eval_loc.npy"), find("tr_pairs.parquet")
OUT = "/kaggle/working"; LOCS = ["left_arm", "left_leg", "right_arm", "right_leg"]; M = 60
tr_meta = pd.read_csv(f"{P}/tr_meta.csv"); te_meta = pd.read_csv(f"{P}/te_meta.csv")
tr_imu = np.load(f"{P}/tr_imu.npy"); te_imu = np.load(f"{P}/te_imu.npy")
tr_vid = np.load(f"{P}/tr_vid.npy", mmap_mode="r"); te_vid = np.load(f"{P}/te_vid.npy", mmap_mode="r")
eval_loc = np.load(f"{B}/eval_loc.npy")
te_loc = te_meta.sensor_location.map({l: i for i, l in enumerate(LOCS)}).to_numpy()
y = tr_meta.label.values; T, N = len(tr_meta), len(te_meta)
tr_sbj, te_sbj = tr_meta.sbj.values, te_meta.sbj_id.values
bases = []
for f in sorted(glob.glob("/kaggle/input/**/oof_*.npy", recursive=True)):
    name = os.path.basename(f)[4:-4]; tf = os.path.join(os.path.dirname(f), f"te_{name}.npy")
    if os.path.exists(tf) and name != "stage2": bases.append((name, np.load(f), np.load(tf)))
print("base models:", [b[0] for b in bases])
P0_tr = np.mean([b[1] for b in bases], 0); P0_te = np.mean([b[2] for b in bases], 0)
tr_pairs = pd.read_parquet(f"{C}/tr_pairs.parquet"); te_pairs = pd.read_parquet(f"{C}/te_pairs.parquet")
def mf1(p, w0=1.0): q = p.copy(); q[:, 0] *= w0; return f1_score(y, q.argmax(1), average="macro")
def best_null(p): return max((round(mf1(p, w), 4), w) for w in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2])
print("base avg", best_null(P0_tr))

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


Ftr = imu_feats(tr_imu[np.arange(T), eval_loc], eval_loc); Fte = imu_feats(te_imu, te_loc)
mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-6
pca = PCA(16, random_state=0).fit((Ftr - mu) / sd)
def zimu(F): Z = pca.transform((F - mu) / sd); return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-8)
Ztr, Zte = zimu(Ftr), zimu(Fte)

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


def succ_lookup(pairs, n):
    W = sp.csr_matrix((pairs.p.values, (pairs.gi.values, pairs.gj.values)), (n, n))
    return (W + W.T).tocsr()

def subject_pairs(V, Z, loc, P0, idx, Wsucc):
    V = np.asarray(V, dtype=np.float32); n = len(V)
    mean = V.mean(1); mu_ = mean.mean(0)
    _, S, Vt = np.linalg.svd(mean - mu_, full_matrices=False); W = Vt[:256].T / (S[:256] / np.sqrt(n) + 1e-6)
    w = lambda x: norm((x - mu_) @ W)
    m, first, last = w(mean), w(V[:, 0]), w(V[:, -1])
    Sm = m @ m.T; np.fill_diagonal(Sm, -np.inf)
    nb = np.argpartition(-Sm, M, 1)[:, :M]
    i = np.repeat(np.arange(n), M); j = nb.ravel()
    rank = np.argsort(np.argsort(-Sm, 1), 1)
    sm = Sm[i, j]
    ws = np.asarray(Wsucc[idx[i], idx[j]]).ravel()
    feats = dict(
        s_mean=sm, rank_ij=rank[i, j], rank_ji=rank[j, i], gap=np.take_along_axis(Sm, nb, 1).max(1).repeat(M) - sm,
        s_ff=(first[i] * first[j]).sum(1), s_ll=(last[i] * last[j]).sum(1), s_lf=(last[i] * first[j]).sum(1), s_fl=(first[i] * last[j]).sum(1),
        s_imu=(Z[i] * Z[j]).sum(1), same_loc=(loc[i] == loc[j]).astype(np.float32), loc_i=loc[i], loc_j=loc[j],
        p_dot=(P0[i] * P0[j]).sum(1), p_l1=np.abs(P0[i] - P0[j]).sum(1), p_same=(P0[i].argmax(1) == P0[j].argmax(1)).astype(np.float32),
        p_null_i=P0[i, 0], p_null_j=P0[j, 0], p_max_i=P0[i].max(1), p_max_j=P0[j].max(1), w_succ=ws,
    )
    return idx[i], idx[j], pd.DataFrame(feats).astype(np.float32)

# %%
t0 = time.time()
Ws_tr, Ws_te = succ_lookup(tr_pairs, T), succ_lookup(te_pairs, N)
rows = []
for s in np.unique(tr_sbj):
    idx = np.where(tr_sbj == s)[0]
    gi, gj, Fdf = subject_pairs(tr_vid[idx], Ztr[idx], eval_loc[idx], P0_tr[idx], idx, Ws_tr)
    Fdf["gi"], Fdf["gj"], Fdf["sbj"] = gi, gj, s; Fdf["target"] = (y[gi] == y[gj]).astype(np.int8); rows.append(Fdf)
pairs = pd.concat(rows, ignore_index=True)
FEATS = [c for c in pairs.columns if c not in ("gi", "gj", "sbj", "target")]
print("pairs", pairs.shape, "base rate same-label", pairs.target.mean().round(3), f"{time.time()-t0:.0f}s")

params = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=200, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=os.cpu_count())
subjects = np.unique(tr_sbj)
fold_of = {s: k for k, (_, va) in enumerate(GroupKFold(5).split(subjects, groups=subjects)) for s in subjects[va]}
pairs["p"] = 0.0
for k in range(5):
    va = pairs.sbj.map(fold_of).values == k
    m = lgb.train(params, lgb.Dataset(pairs.loc[~va, FEATS], pairs.target.values[~va]), 400)
    pairs.loc[va, "p"] = m.predict(pairs.loc[va, FEATS])
print("same-label AUC", round(roc_auc_score(pairs.target, pairs.p), 4))
for thr in [0.5, 0.7, 0.9]:
    sel = pairs.p >= thr; print(f"  p>={thr}: {sel.mean():.3f} of pairs, precision {pairs.target[sel].mean():.3f}, mean #nbrs/tile {sel.sum()/T:.1f}")
print(pd.Series(m.feature_importance("gain"), FEATS).sort_values(ascending=False).round(0).to_dict())
final = lgb.train(params, lgb.Dataset(pairs[FEATS], pairs.target.values), 400)

# %%
def seg_context(pairs_df, P0, n):
    feats, names = [], []
    for g in [1, 4, 16]:  # sharpen weights: p**g
        Wn = rownorm(sp.csr_matrix((pairs_df.p.values ** g, (pairs_df.gi.values, pairs_df.gj.values)), (n, n)))
        a1 = Wn @ P0; a2 = Wn @ a1
        feats += [a1, a2]; names += [f"seg_g{g}_h1_{c}" for c in range(19)] + [f"seg_g{g}_h2_{c}" for c in range(19)]
    for thr in [0.7, 0.9]:
        sel = pairs_df.p.values >= thr
        A = sp.csr_matrix((np.ones(sel.sum()), (pairs_df.gi.values[sel], pairs_df.gj.values[sel])), (n, n))
        deg = np.asarray(A.sum(1)).ravel()
        feats += [rownorm(A + sp.eye(n)) @ P0, np.log1p(deg)[:, None]]
        names += [f"seg_t{thr}_{c}" for c in range(19)] + [f"seg_t{thr}_deg"]
    return np.concatenate(feats, 1).astype(np.float32), names

Xtr, names = seg_context(pairs, P0_tr, T)
for nm in ["seg_g1_h1", "seg_g4_h1", "seg_g16_h1", "seg_g4_h2", "seg_t0.9"]:
    cols = [k for k, n_ in enumerate(names) if n_.startswith(nm + "_") and not n_.endswith("deg")]
    print(f"{nm}: direct argmax OOF {best_null(Xtr[:, cols])}")

rows = []
for s in np.unique(te_sbj):
    idx = np.where(te_sbj == s)[0]
    gi, gj, Fdf = subject_pairs(te_vid[idx], Zte[idx], te_loc[idx], P0_te[idx], idx, Ws_te)
    Fdf["gi"], Fdf["gj"] = gi, gj; rows.append(Fdf)
tp = pd.concat(rows, ignore_index=True); tp["p"] = final.predict(tp[FEATS])
Xte, _ = seg_context(tp, P0_te, N)
print("test mean p", tp.p.mean().round(3), "| train mean p", pairs.p.mean().round(3))
np.save(f"{OUT}/tr_segctx.npy", Xtr); np.save(f"{OUT}/te_segctx.npy", Xte)
pd.Series(names).to_csv(f"{OUT}/segctx_names.csv", index=False)
pairs[["gi", "gj", "p", "target"]].to_parquet(f"{OUT}/tr_seg_pairs.parquet"); tp[["gi", "gj", "p"]].to_parquet(f"{OUT}/te_seg_pairs.parquet")
