# %% [markdown]
# # WEAR 2026 — Step D1: transductive smoothing over the shuffled tiles of each subject
# Test tiles are a full session cut into 1 s pieces and shuffled. Tiles that are close in time are close in
# VideoMAE space, so we smooth per-tile class probabilities over within-subject neighbour graphs:
# (a) kNN on whitened tile-mean video embeddings, (b) successor graph (end of tile i -> start of tile j).
# Evaluated on baseline OOF probabilities (subject-grouped), then applied to test.

# %%
import glob, os, time
import numpy as np, pandas as pd
from sklearn.metrics import f1_score

def find(name): return os.path.dirname(glob.glob(f"/kaggle/input/**/{name}", recursive=True)[0])
P, B = find("tr_meta.csv"), find("oof_lgb.npy")
OUT = "/kaggle/working"
tr_meta = pd.read_csv(f"{P}/tr_meta.csv"); te_meta = pd.read_csv(f"{P}/te_meta.csv")
tr_vid = np.load(f"{P}/tr_vid.npy", mmap_mode="r"); te_vid = np.load(f"{P}/te_vid.npy", mmap_mode="r")
oof = np.load(f"{B}/oof_lgb.npy"); pte = np.load(f"{B}/te_lgb.npy")
y = tr_meta.label.values

def norm(a): return a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)
def mf1(p, w0=1.0):
    q = p.copy(); q[:, 0] *= w0; return f1_score(y, q.argmax(1), average="macro")
def best_null(p):
    return max((mf1(p, w), w) for w in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2])
print("baseline OOF macro-F1 (raw, best null w):", round(mf1(oof), 4), best_null(oof))

# %%
def embeddings(V, n_comp=256):
    """Per-subject whitened embeddings: tile mean, first frame, last frame."""
    V = np.asarray(V, dtype=np.float32)
    mean = V.mean(1); mu = mean.mean(0)
    _, S, Vt = np.linalg.svd(mean - mu, full_matrices=False)
    W = Vt[:n_comp].T / (S[:n_comp] / np.sqrt(len(mean)) + 1e-6)
    f = lambda x: norm((x - mu) @ W)
    return f(mean), f(V[:, 0]), f(V[:, -1])

def graphs(emb, k_knn, k_succ):
    m, first, last = emb; n = len(m)
    Sm = m @ m.T; np.fill_diagonal(Sm, -np.inf)
    Ss = last @ first.T; np.fill_diagonal(Ss, -np.inf)
    G = {}
    nb = np.argpartition(-Sm, k_knn, 1)[:, :k_knn]; G["knn"] = nb
    sc = np.argpartition(-Ss, k_succ, 1)[:, :k_succ]; pr = np.argpartition(-Ss.T, k_succ, 1)[:, :k_succ]
    G["succ"] = np.c_[sc, pr]  # likely successors and predecessors
    return G

def smooth(p, nb, alpha, iters):
    q = p.copy()
    for _ in range(iters):
        q = (1 - alpha) * p + alpha * q[nb].mean(1)
    return q

_emb = {}
def run(meta_sbj, V, P0, k_knn, k_succ, alpha, iters, which):
    out = P0.copy()
    for s in np.unique(meta_sbj):
        idx = np.where(meta_sbj == s)[0]
        key = (id(V), s)
        if key not in _emb: _emb[key] = embeddings(V[idx])
        G = graphs(_emb[key], k_knn, k_succ)
        nb = np.concatenate([G[w] for w in which], 1)
        out[idx] = smooth(P0[idx], nb, alpha, iters)
    return out

# %%
# Neighbour label purity (how often graph neighbours share the tile's label)
for s in [0, 7, 14]:
    idx = np.where(tr_meta.sbj.values == s)[0]
    G = graphs(embeddings(tr_vid[idx]), 10, 3)
    for w in G: print(f"sbj {s} {w}: purity {(y[idx][G[w]] == y[idx][:, None]).mean():.3f}")

# %%
res = []
t0 = time.time()
for which in [("knn",), ("succ",), ("knn", "succ")]:
    for k_knn, k_succ in [(5, 2), (10, 3), (20, 5)]:
        for alpha, iters in [(0.5, 1), (0.8, 3), (0.9, 10)]:
            q = run(tr_meta.sbj.values, tr_vid, oof, k_knn, k_succ, alpha, iters, which)
            f, w = best_null(q)
            res.append(dict(which="+".join(which), k_knn=k_knn, k_succ=k_succ, alpha=alpha, iters=iters, f1=round(f, 4), null_w=w))
            print(res[-1], f"{time.time()-t0:.0f}s")
res = pd.DataFrame(res).sort_values("f1", ascending=False); print(res.head(10).to_string())

# %%
b = res.iloc[0]
which = tuple(b.which.split("+"))
q_te = run(te_meta.sbj_id.values, te_vid, pte, int(b.k_knn), int(b.k_succ), b.alpha, int(b.iters), which)
q_te[:, 0] *= b.null_w
np.save(f"{OUT}/te_knn.npy", q_te)
sub = pd.DataFrame({"id": te_meta.id, "target_feature": q_te.argmax(1).astype(int)})
sub.to_csv(f"{OUT}/submission.csv", index=False)
print("best", b.to_dict()); print(sub.target_feature.value_counts(normalize=True).sort_index().round(3).to_dict())
