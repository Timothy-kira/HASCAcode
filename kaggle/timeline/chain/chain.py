# %% [markdown]
# # WEAR 2026 — Step D2: learned successor model -> chains -> smoothing along time
# 1. Candidates: for each tile i, top-K tiles j (same subject) by whitened cos(last frame i, first frame j).
# 2. Pair features (video similarities, mutual ranks, motion continuity, IMU boundary continuity) -> LightGBM
#    "is j the successor of i" (trained on simulated test tiles of train subjects, subject-grouped OOF).
# 3. Greedy linking by descending probability (each tile at most one succ/pred, no cycles) -> chains.
# 4. Smooth class probabilities along chains + successor graph; evaluate OOF macro-F1; apply to test.

# %%
import glob, os, time
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold

def find(name): return os.path.dirname(glob.glob(f"/kaggle/input/**/{name}", recursive=True)[0])
P, B = find("tr_meta.csv"), find("oof_lgb.npy")
OUT = "/kaggle/working"; LOCS = ["left_arm", "left_leg", "right_arm", "right_leg"]
K = 30
tr_meta = pd.read_csv(f"{P}/tr_meta.csv"); te_meta = pd.read_csv(f"{P}/te_meta.csv")
tr_imu = np.load(f"{P}/tr_imu.npy"); te_imu = np.load(f"{P}/te_imu.npy")
tr_vid = np.load(f"{P}/tr_vid.npy", mmap_mode="r"); te_vid = np.load(f"{P}/te_vid.npy", mmap_mode="r")
eval_loc = np.load(f"{B}/eval_loc.npy"); oof = np.load(f"{B}/oof_lgb.npy"); pte = np.load(f"{B}/te_lgb.npy")
te_loc = te_meta.sensor_location.map({l: i for i, l in enumerate(LOCS)}).to_numpy()
y = tr_meta.label.values
tr_x = tr_imu[np.arange(len(tr_meta)), eval_loc].astype(np.float32)  # the single sensor the "test-like" view sees

# %%
def norm(a): return a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)

def subject_pairs(V, X, loc, n_comp=256):
    """Candidate pairs + features for one subject. V (n,15,768), X (n,50,3) single-sensor IMU, loc (n,)."""
    V = np.asarray(V, dtype=np.float32); n = len(V)
    mean = V.mean(1); mu = mean.mean(0)
    _, S, Vt = np.linalg.svd(mean - mu, full_matrices=False)
    W = Vt[:n_comp].T / (S[:n_comp] / np.sqrt(n) + 1e-6)
    w = lambda x: norm((x - mu) @ W)
    first, last, f1_, l1_, m = w(V[:, 0]), w(V[:, -1]), w(V[:, 1]), w(V[:, -2]), w(mean)
    rawc = lambda x: norm(x - mu)
    rfirst, rlast = rawc(V[:, 0]), rawc(V[:, -1])
    vel = (V[:, -1] - V[:, -5]) @ W  # whitened motion at the end of tile i
    S1 = last @ first.T; np.fill_diagonal(S1, -np.inf)
    cand = np.argpartition(-S1, K, 1)[:, :K]
    i = np.repeat(np.arange(n), K); j = cand.ravel()
    rank_row = np.argsort(np.argsort(-S1, 1), 1)  # rank of j among successors of i
    rank_col = np.argsort(np.argsort(-S1, 0), 0)  # rank of i among predecessors of j
    s1 = S1[i, j]
    endx = X[:, -1] + (X[:, -1] - X[:, -3]) / 2
    mag = np.linalg.norm(X, axis=2)
    same = (loc[i] == loc[j]).astype(np.float32)
    feats = dict(
        s1=s1, rank_row=rank_row[i, j], rank_col=rank_col[i, j],
        gap_row=np.take_along_axis(S1, cand, 1).max(1).repeat(K) - s1,
        gap_col=np.where(np.isfinite(S1), S1, -1).max(0)[j] - s1,
        s_inner=(l1_[i] * f1_[j]).sum(1), s_mean=(m[i] * m[j]).sum(1),
        s_raw=(rlast[i] * rfirst[j]).sum(1), s_self_i=(first[i] * last[i]).sum(1), s_self_j=(first[j] * last[j]).sum(1),
        vel_cos=(norm(vel[i]) * norm((V[j, 0] - V[i, -1]) @ W)).sum(1),
        same_loc=same, loc_i=loc[i], loc_j=loc[j],
        imu_gap=np.where(same > 0, np.linalg.norm(endx[i] - X[j, 0], axis=1), np.nan),
        mag_mean_diff=np.abs(mag[i].mean(1) - mag[j].mean(1)), mag_std_diff=np.abs(mag[i].std(1) - mag[j].std(1)),
        mag_edge_diff=np.abs(mag[i, -5:].mean(1) - mag[j, :5].mean(1)),
    )
    return i, j, pd.DataFrame(feats).astype(np.float32)

# %%
t0 = time.time()
rows = []
for s in np.unique(tr_meta.sbj):
    idx = np.where(tr_meta.sbj.values == s)[0]
    i, j, Fdf = subject_pairs(tr_vid[idx], tr_x[idx], eval_loc[idx])
    gi, gj = idx[i], idx[j]
    Fdf["target"] = ((gj == gi + 1) & (tr_meta.file_id.values[gi] == tr_meta.file_id.values[np.minimum(gi + 1, len(tr_meta) - 1)])).astype(np.int8)
    Fdf["gi"], Fdf["gj"], Fdf["sbj"] = gi, gj, s
    rows.append(Fdf)
pairs = pd.concat(rows, ignore_index=True)
FEATS = [c for c in pairs.columns if c not in ("target", "gi", "gj", "sbj")]
print("pairs", pairs.shape, "positives", pairs.target.sum(), "recall@K", round(pairs.target.sum() / len(tr_meta), 3), f"{time.time()-t0:.0f}s")

# %%
params = dict(objective="binary", learning_rate=0.05, num_leaves=63, min_data_in_leaf=100, feature_fraction=0.8,
              bagging_fraction=0.8, bagging_freq=1, verbose=-1, num_threads=os.cpu_count())
pairs["p"] = 0.0
for k, (tri, vai) in enumerate(GroupKFold(5).split(pairs, groups=pairs.sbj)):
    m = lgb.train(params, lgb.Dataset(pairs.loc[tri, FEATS], pairs.target.values[tri]), 600)
    pairs.loc[vai, "p"] = m.predict(pairs.loc[vai, FEATS])
print("pair AUC", round(roc_auc_score(pairs.target, pairs.p), 4))
top = pairs.loc[pairs.groupby("gi").p.idxmax()]
print("reranked top1 successor acc:", round(top.target.mean(), 4), "| raw top1:", round(pairs.loc[pairs.rank_row == 0, "target"].mean(), 4))
imp = pd.Series(m.feature_importance("gain"), FEATS).sort_values(ascending=False); print(imp.round(0).to_dict())
final_pair_model = lgb.train(params, lgb.Dataset(pairs[FEATS], pairs.target.values), 600)

# %%
def link(gi, gj, p, thr):
    """Greedy chain building. Returns succ/pred arrays (-1 = none) over global indices."""
    n = int(max(gi.max(), gj.max())) + 1
    succ = -np.ones(n, int); pred = -np.ones(n, int); head = np.arange(n)  # head: union-find-ish chain id
    def root(a):
        while head[a] != a: head[a] = head[head[a]]; a = head[a]
        return a
    for o in np.argsort(-p):
        if p[o] < thr: break
        a, b = gi[o], gj[o]
        if succ[a] >= 0 or pred[b] >= 0 or root(a) == root(b): continue
        succ[a], pred[b] = b, a; head[root(b)] = root(a)
    return succ, pred

def chains_from(succ, pred):
    out = []
    for s in np.where(pred < 0)[0]:
        c = [s]
        while succ[c[-1]] >= 0: c.append(succ[c[-1]])
        out.append(np.array(c))
    return out

def chain_smooth(P0, chains, w):
    Q = P0.copy(); L = np.log(P0 + 1e-6)
    for c in chains:
        if len(c) < 2: continue
        cs = np.cumsum(np.r_[np.zeros((1, 19)), L[c]], 0)
        lo = np.clip(np.arange(len(c)) - w, 0, len(c)); hi = np.clip(np.arange(len(c)) + w + 1, 0, len(c))
        Q[c] = np.exp((cs[hi] - cs[lo]) / (hi - lo)[:, None])
    return Q / Q.sum(1, keepdims=True)

def mf1(p, w0): q = p.copy(); q[:, 0] *= w0; return f1_score(y, q.argmax(1), average="macro")
def best_null(p): return max((round(mf1(p, w), 4), w) for w in [0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0])

print("baseline", best_null(oof))
res = []
for thr in [0.05, 0.2, 0.4]:
    succ, pred = link(pairs.gi.values, pairs.gj.values, pairs.p.values, thr)
    ch = chains_from(succ, pred)
    ok = succ[succ >= 0] == np.where(succ >= 0)[0] + 1
    L = np.array([len(c) for c in ch])
    print(f"thr {thr}: links {int((succ>=0).sum())} precision {ok.mean():.3f} | chains {len(ch)} mean len {L.mean():.1f} median {np.median(L):.0f}")
    for w in [2, 5, 10, 20]:
        f, nw = best_null(chain_smooth(oof, ch, w)); res.append(dict(thr=thr, w=w, f1=f, null_w=nw)); print("   w", w, f, nw)
res = pd.DataFrame(res).sort_values("f1", ascending=False); print(res.head().to_string())

# %%
# Apply to test
b = res.iloc[0]
rows = []
for s in np.unique(te_meta.sbj_id):
    idx = np.where(te_meta.sbj_id.values == s)[0]
    i, j, Fdf = subject_pairs(te_vid[idx], te_imu[idx].astype(np.float32), te_loc[idx])
    Fdf["gi"], Fdf["gj"] = idx[i], idx[j]; rows.append(Fdf)
tp = pd.concat(rows, ignore_index=True); tp["p"] = final_pair_model.predict(tp[FEATS])
succ, pred = link(tp.gi.values, tp.gj.values, tp.p.values, b.thr); ch = chains_from(succ, pred)
L = np.array([len(c) for c in ch]); print("test chains", len(ch), "mean len", L.mean().round(1), "links", int((succ >= 0).sum()))
q = chain_smooth(pte, ch, int(b.w)); q[:, 0] *= b.null_w
np.save(f"{OUT}/te_chain_probs.npy", q); np.save(f"{OUT}/te_succ.npy", succ)
pairs[["gi", "gj", "p", "target"]].to_parquet(f"{OUT}/tr_pairs.parquet"); tp[["gi", "gj", "p"]].to_parquet(f"{OUT}/te_pairs.parquet")
pd.DataFrame({"id": te_meta.id, "target_feature": q.argmax(1)}).to_csv(f"{OUT}/submission.csv", index=False)
print("best", b.to_dict())
