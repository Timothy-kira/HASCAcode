# %% [markdown]
# # WEAR 2026 — Step A: tile preparation
# Cuts every training recording into non-overlapping 1 s tiles that mirror the test construction
# (50 IMU rows per sensor, central 15 of 30 VideoMAE frames) and re-packs the test set in the same layout.
# Outputs (in /kaggle/working, consumed by later kernels via `kernel_sources`):
# `tr_imu.npy (T,4,50,3)`, `tr_vid.npy (T,15,768) f16`, `tr_meta.csv`, `te_imu.npy (N,50,3)`, `te_vid.npy (N,15,768) f16`, `te_meta.csv`.

# %%
import glob, os, re, time
import numpy as np, pandas as pd

ROOT = os.path.dirname(glob.glob("/kaggle/input/**/sample_submission.csv", recursive=True)[0])
OUT = "/kaggle/working"
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
# ## Sanity: does the train tile video match the test tile video distribution?
# If our frame slice (8..22) matches the test construction, per-frame-position statistics should agree.

# %%
tr_vid = np.concatenate(vids); del vids
for name, V in [("train", tr_vid[::7]), ("test", te_vid)]:
    Vf = V.astype(np.float32)
    step = np.linalg.norm(Vf[:, 1:] - Vf[:, :-1], axis=2).mean(0)  # mean frame-to-frame change per position
    print(name, "norm", np.linalg.norm(Vf, axis=2).mean().round(2), "frame-step", step.round(2)[[0, 3, 7, 10, 13]])

# %% [markdown]
# ## Feasibility of timeline reconstruction
# For each tile t, is tile t+1 the best match of "last frame of t" → "first frame of candidate" among all tiles of the same subject?

# %%
def succ_scores(V):
    last = V[:, -1].astype(np.float32); first = V[:, 0].astype(np.float32)
    trend = last + (last - V[:, -4].astype(np.float32)) * (16 / 3)  # linear extrapolation 16 frames ahead
    out = {}
    for k, q in [("last", last), ("extrap", trend)]:
        qn = q / np.linalg.norm(q, axis=1, keepdims=True); fn = first / np.linalg.norm(first, axis=1, keepdims=True)
        S = qn @ fn.T; np.fill_diagonal(S, -np.inf); out[k] = S
    return out

rows = []
for sbj, g in tr_meta.groupby("sbj"):
    idx = g.index.values
    S = succ_scores(tr_vid[idx])
    nxt = np.r_[idx[1:], -1]
    valid = (g.file_id.values[:-1] == g.file_id.values[1:])
    for k, M in S.items():
        best = idx[M.argmax(1)]
        mutual = M.argmax(0)[M.argmax(1)] == np.arange(len(idx))
        acc = (best[:-1] == nxt[:-1])[valid]
        rows.append(dict(sbj=sbj, method=k, n=len(idx), top1=acc.mean(), mutual_rate=mutual.mean(),
                         top1_if_mutual=(best[:-1] == nxt[:-1])[valid & mutual[:-1]].mean()))
r = pd.DataFrame(rows); print(r.groupby("method")[["top1", "mutual_rate", "top1_if_mutual"]].mean().round(4)); print(r.round(3).to_string())

# %%
for sbj, g in te_meta.groupby("sbj_id"):
    S = succ_scores(te_vid[g.index.values])["extrap"]
    mutual = S.argmax(0)[S.argmax(1)] == np.arange(len(g))
    srt = np.sort(S, 1)
    print("test sbj", sbj, "n", len(g), "mutual_rate", mutual.mean().round(4), "median top1-top2 margin", np.median(srt[:, -1] - srt[:, -2]).round(4))
