# EDA findings (Kaggle CPU run, notebook `evelynyang02/wear-hasca-2026-eda` v2)

## Test set
- 12,234 windows, 4 unseen subjects (sbj 22–25), each window = ONE sensor, locations balanced (~25% each).
  - sbj_22: 5197, sbj_23: 3128, sbj_24: 1995, sbj_25: 1914
- `test_meta_data.csv` columns are `id, sbj_id, sensor_location` (NOT `subject_id` / `inertial_sensor_location` as the docs say).
- `test_inertial_data.npy`: (12234, 50, 3) float64, no NaN.
- `test_videomae_data.npy`: **(12234, 768, 15)** float64 — channels-first, NOT (N, 15, 768) as documented. Transpose before use.
- `sample_submission.csv` column is `target_feature` (floats in sample; submit ints).
- Test per-location means match train per-location means well → the location tag is reliable; std is somewhat lower in test (likely fewer transitions / different window sampling).

## Train set
- 24 files / 22 subjects (sbj_0 and sbj_14 have a second session `_2`), ~19.3 h total.
- IMU 50 Hz and video 30 FPS are exactly aligned in duration for every file (frame = row*30/50, i.e. 3 frames per 5 IMU rows).
- Label column is a string; `NaN` = null class (0) → 39.7% of all rows. The other 18 classes are near-balanced (3.0–3.6% each).
- **sbj_10 has 154,176 NaN IMU values** — needs imputation or dropping those rows.
- Activities come in long contiguous segments (30–180 s) separated by null gaps; a subject performs each activity 1–3 times.

## Implications
- Build training windows: 1 s (50 IMU rows) × single sensor location (4× augmentation for free), + matching 15 central video frames (drop 7–8 boundary frames of the 30-frame second, to mirror test).
- Validate leave-subject-out (GroupKFold on sbj_id); metric macro-F1 over 19 classes incl. null.
- Null is the biggest and most confusable class; video features should help separate it.
