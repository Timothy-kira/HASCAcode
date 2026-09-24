"""Builds wear-eda.ipynb from the cell sources below (keeps the notebook diff-friendly)."""
import json

cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip()})
def code(s): cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip()})

md("""
# 3rd WEAR Dataset Challenge (HASCA 2026) — EDA
Reads the competition data in place on Kaggle (CPU) and summarises its structure.
""")
code(r'''
import glob, os, json
import numpy as np, pandas as pd
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 50)

# Competition data mounts at /kaggle/input/<slug> or /kaggle/input/competitions/<slug>
ROOT = [p for p in glob.glob("/kaggle/input/**/sample_submission.csv", recursive=True)]
ROOT = os.path.dirname(ROOT[0]); print("ROOT =", ROOT)
for dp, dn, fn in sorted(os.walk(ROOT)):
    print(dp.replace(ROOT, "."), f"{len(fn)} files", sorted(fn)[:4], "...")
''')
code(r'''
print(open(f"{ROOT}/participant_meta_data.txt").read())
''')
code(r'''
sub = pd.read_csv(f"{ROOT}/sample_submission.csv"); print(sub.shape); print(sub.head())
tm = pd.read_csv(f"{ROOT}/test/test_meta_data.csv"); print(tm.shape); print(tm.head())
print(tm.nunique())
print(pd.crosstab(tm.sbj_id, tm.sensor_location))
''')
code(r'''
Xi = np.load(f"{ROOT}/test/test_inertial_data.npy", mmap_mode="r")
Xv = np.load(f"{ROOT}/test/test_videomae_data.npy", mmap_mode="r")
print("test inertial", Xi.shape, Xi.dtype, "| test video", Xv.shape, Xv.dtype)
xi = np.asarray(Xi)
print("inertial per-axis mean", xi.mean((0, 1)).round(3), "std", xi.std((0, 1)).round(3))
print("NaN inertial:", np.isnan(xi).sum(), "| NaN video (first 2000):", np.isnan(np.asarray(Xv[:2000])).sum())
''')
code(r'''
LABELS = ["null","jogging","jogging (rotating arms)","jogging (skipping)","jogging (sidesteps)","jogging (butt-kicks)",
          "stretching (triceps)","stretching (lunging)","stretching (shoulders)","stretching (hamstrings)",
          "stretching (lumbar rotation)","push-ups","push-ups (complex)","sit-ups","sit-ups (complex)",
          "burpees","lunges","lunges (complex)","bench-dips"]
rows, label_counts = [], {}
for f in sorted(glob.glob(f"{ROOT}/train/inertial_feat/*.csv")):
    name = os.path.basename(f)[:-4]
    df = pd.read_csv(f)
    v = np.load(f"{ROOT}/train/videomae_feat/{name}.npy", mmap_mode="r")
    if not rows: print(df.columns.tolist()); print(df.head(3)); print(df.dtypes.value_counts())
    lab = df["label"]
    label_counts[name] = lab.astype(str).value_counts()
    rows.append(dict(file=name, sbj=df.sbj_id.iloc[0], imu_rows=len(df), imu_sec=len(df)/50,
                     vid_frames=v.shape[0], vid_sec=v.shape[0]/30, vid_dim=v.shape[1],
                     nan_imu=int(df.drop(columns="label").isna().sum().sum()), nan_label=int(lab.isna().sum())))
summ = pd.DataFrame(rows); summ["sec_diff"] = summ.imu_sec - summ.vid_sec
print(summ.to_string())
print("total hours IMU:", summ.imu_sec.sum() / 3600)
''')
code(r'''
lc = pd.DataFrame(label_counts).fillna(0).astype(int)
lc["total"] = lc.sum(1); lc["share%"] = (100 * lc.total / lc.total.sum()).round(2)
print("raw label values:", lc.index.tolist())
print(lc[["total", "share%"]].sort_values("total", ascending=False).to_string())
''')
code(r'''
# Segment structure: contiguous runs of the same label in one subject file
f = sorted(glob.glob(f"{ROOT}/train/inertial_feat/*.csv"))[0]
df = pd.read_csv(f); lab = df["label"].astype(str)
seg = (lab != lab.shift()).cumsum()
segs = df.groupby(seg).agg(label=("label", "first"), n=("label", "size"))
segs["sec"] = segs.n / 50
print(os.path.basename(f), "segments:", len(segs)); print(segs.to_string())
''')
code(r'''
# Is the test inertial a single sensor per window? Compare its stats to each train sensor location
tr = pd.read_csv(sorted(glob.glob(f"{ROOT}/train/inertial_feat/*.csv"))[0])
for loc in ["right_arm", "left_arm", "right_leg", "left_leg"]:
    c = [f"{loc}_acc_{a}" for a in "xyz"]
    print(loc, "mean", tr[c].mean().values.round(3), "std", tr[c].std().values.round(3))
for loc in tm.sensor_location.unique():
    m = (tm.sensor_location == loc).values
    print("TEST", loc, "mean", xi[m].mean((0, 1)).round(3), "std", xi[m].std((0, 1)).round(3))
''')

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open("wear-eda.ipynb", "w"), indent=1)
