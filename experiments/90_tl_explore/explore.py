# %% [markdown]
# # WEAR 2026 — Step D exploration: how well can shuffled 1 s tiles be chained back in time?
# For every train tile t (per subject, all tiles shuffled together like the test set) we rank all candidate
# successors and measure top-1 / recall@K of the true t+1 under several similarity definitions.

# %%
import glob, os, time
import numpy as np, pandas as pd
P = os.path.dirname(glob.glob("/kaggle/input/**/tr_meta.csv", recursive=True)[0])
tr_meta = pd.read_csv(f"{P}/tr_meta.csv")
tr_vid = np.load(f"{P}/tr_vid.npy").astype(np.float32)
tr_imu = np.load(f"{P}/tr_imu.npy")
print(tr_vid.shape)

# %%
def norm(a): return a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)

def evaluate(name, Q, K_, subjects=range(0, 22, 3)):
    """Q: query vectors (end of tile), K_: key vectors (start of tile). Returns mean top1 / recall@k of true successor."""
    res = []
    for s in subjects:
        idx = np.where(tr_meta.sbj.values == s)[0]
        S = norm(Q[idx]) @ norm(K_[idx]).T; np.fill_diagonal(S, -np.inf)
        f = tr_meta.file_id.values[idx]; ok = f[:-1] == f[1:]
        rank = (S[:-1] > S[np.arange(len(idx) - 1), np.arange(1, len(idx))][:, None]).sum(1)[ok]
        res.append([(rank < k).mean() for k in (1, 5, 20, 50)])
    r = np.mean(res, 0); print(f"{name:40s} top1 {r[0]:.3f}  R@5 {r[1]:.3f}  R@20 {r[2]:.3f}  R@50 {r[3]:.3f}")

def centred(V):
    V = V.copy()
    for s in tr_meta.sbj.unique():
        m = tr_meta.sbj.values == s; V[m] -= V[m].reshape(-1, 768).mean(0)
    return V

Vc = centred(tr_vid)
evaluate("raw last->first", tr_vid[:, -1], tr_vid[:, 0])
evaluate("centred last->first", Vc[:, -1], Vc[:, 0])
evaluate("centred mean(last3)->mean(first3)", Vc[:, -3:].mean(1), Vc[:, :3].mean(1))
evaluate("centred tile mean->tile mean", Vc.mean(1), Vc.mean(1))
for h in [2, 4, 8]:
    ext = Vc[:, -1] + (Vc[:, -1] - Vc[:, -1 - h]) * (16 / h) * 0.5
    evaluate(f"centred half-extrap h={h}", ext, Vc[:, 0])

# %%
# Whitening (per subject PCA-whiten) often makes cosine far more discriminative for transformer features
def whiten_eval(n_comp):
    res = []
    for s in range(0, 22, 3):
        idx = np.where(tr_meta.sbj.values == s)[0]
        X = Vc[idx]; flat = X.reshape(-1, 768)
        mu = flat.mean(0); U, S_, Vt = np.linalg.svd(flat[::5] - mu, full_matrices=False)
        W = Vt[:n_comp].T / (S_[:n_comp] / np.sqrt(len(flat[::5])))
        q = norm((X[:, -1] - mu) @ W); k = norm((X[:, 0] - mu) @ W)
        S = q @ k.T; np.fill_diagonal(S, -np.inf)
        f = tr_meta.file_id.values[idx]; ok = f[:-1] == f[1:]
        rank = (S[:-1] > S[np.arange(len(idx) - 1), np.arange(1, len(idx))][:, None]).sum(1)[ok]
        res.append([(rank < k_).mean() for k_ in (1, 5, 20, 50)])
    r = np.mean(res, 0); print(f"whitened n={n_comp:<4d}{'':30s} top1 {r[0]:.3f}  R@5 {r[1]:.3f}  R@20 {r[2]:.3f}  R@50 {r[3]:.3f}")
for n in [32, 64, 128, 256]:
    whiten_eval(n)

# %%
# IMU boundary continuity: if tile t and t+1 happen to use the same sensor, the signal is continuous.
# With a random sensor per tile that happens 25% of the time. Measure how distinctive the boundary is.
res = []
for s in range(0, 22, 3):
    idx = np.where(tr_meta.sbj.values == s)[0]
    loc = np.random.default_rng(s).integers(0, 4, len(idx))
    X = tr_imu[idx, loc]  # (n,50,3)
    end = X[:, -1] + (X[:, -1] - X[:, -3]) / 2; start = X[:, 0]
    D = ((end[:, None] - start[None]) ** 2).sum(-1); np.fill_diagonal(D, np.inf)
    same = loc[:-1] == loc[1:]
    D[loc[:, None] != loc[None]] = np.inf  # only compare same-sensor tiles
    f = tr_meta.file_id.values[idx]; ok = (f[:-1] == f[1:]) & same
    rank = (D[:-1] < D[np.arange(len(idx) - 1), np.arange(1, len(idx))][:, None]).sum(1)[ok]
    res.append([(rank < k).mean() for k in (1, 5, 20)])
print("IMU same-sensor boundary: top1 / R@5 / R@20", np.mean(res, 0).round(3))
