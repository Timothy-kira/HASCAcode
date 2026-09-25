# Experiment & submission log

Every scheme, its score and where its exact code lives.
- **Code snapshot**: `experiments/<id>/` holds the exact source that ran on Kaggle (`COMMIT` = git commit it was taken from) and `run.log` (Kaggle output).
- **Live code** (latest version of each step): `kaggle/<step>/`; Kaggle kernels are `evelynyang02/wear-hasca-2026-<step>` (CPU).
- OOF = subject-grouped 5-fold macro-F1 on train tiles built like the test set (1 random sensor per 1 s tile). LB = public leaderboard.
- Kaggle account `evelynyang02`, leaderboard name **Evelyn_Yang_02**.

## Submitted

| # | date (UTC) | scheme | code snapshot | OOF | public LB |
|---|---|---|---|---|---|
| 1 | 2026-09-24 | **baseline v1** — LightGBM on 102 hand-crafted single-sensor IMU features (left arm mirrored, location one-hot) + VideoMAE PCA (raw mean / subject-centred mean / std), train on 2 random sensors per tile, null prob ×0.3 | `experiments/01_baseline_v1` | 0.634 | 0.64641 |
| 2 | 2026-09-24 | **knn v1** — baseline probs smoothed over within-subject successor graph (whitened VideoMAE last→first frame, top-3 succ + top-3 pred), α=0.8, 3 iters | `experiments/02_knn_v1` | 0.720 | 0.69442 |
| 3 | 2026-09-24 | **stage2 v1** — graph-context stacking: learned pair model (chain v1) gives soft succ/pred graph; features = own probs & IMU + graph-aggregated probs (1/2/4 hops) & IMU + video-kNN aggregates → LightGBM, then 1 propagation step (α=0.5), null ×0.5 | `experiments/04_stage2_v1` (+ pairs from `03_chain_v1`) | 0.787 | **0.79079** |

## Not submitted (experiments)

| id | scheme | code snapshot | result | conclusion |
|---|---|---|---|---|
| 00 | prep v1/v2 — tiles 1 s, 4 sensors, central 15 video frames; v2 adds `tr_valid` mask (sbj_10 left arm missing 51k rows) | `experiments/00_prep_v1`, `00_prep_v2` | 69,326 train tiles; train/test video stats match | — |
| 90 | timeline explore — successor retrieval by video / IMU similarity | `experiments/90_tl_explore` | whitened-256 top1 0.337, R@50 0.89; IMU same-sensor boundary top1 0.64 | whitening essential |
| 03 | chain v1 — LightGBM pair model (K=30) + greedy hard chains + smoothing along chains | `experiments/03_chain_v1` | pair AUC 0.922, top1 succ 0.42; OOF 0.7135 | hard chains worse than soft graph; pair probs reused by stage2 |
| 06 | chain v2 — K=50 candidates + base-probability agreement pair features | `experiments/06_chain_v2` | pair AUC 0.943 (v1 0.922), recall@K 0.86, top1 succ 0.427; chain-smoothing OOF 0.7184 | small gain; pairs feed stage2 v3 |
| 07 | stage2 v3 — stage2 v1 config (hops 1–4, 1 round) on chain v2 pairs | `experiments/07_stage2_v3` | OOF 0.7792 → 0.7886 after propagation (v1: 0.7819 → 0.7872) | ≈ tie; better pair AUC alone does not move stage2 → not submitted |
| 08 | fusion v1 — IMU 1D-CNN + loc embedding + VideoMAE (subject-centred) temporal branch, rotation/scale/jitter aug, modality dropout, 14 epochs | `experiments/08_fusion_v1` | OOF 0.629 raw / 0.649 (null ×0.2); LGBM 0.645 | on par with LGBM, different model → use as 2nd base in stage2 |
| 09 | chain v3 — ridge next-frame predictor + union candidates + Hungarian one-to-one assignment | `experiments/09_chain_v3` | ridge top1 0.293 (< plain 0.32); pair AUC 0.9395; assignment precision 0.432 | successor precision plateaus ~43% with video similarity; dead end |
| 10 | diag v1 — stage2 with synthetic successor graphs of controlled precision (hops 1–4) | `experiments/10_diag_v1` | no graph 0.643 · real 0.779 · prec 0.5 → 0.786 · 0.7 → 0.794 · 1.0 → **0.817** | with ≤4-hop context even a perfect graph caps at 0.817 → need long-range (segment-level) context |
| 05 | stage2 v2 — hops up to 16 + 2nd stacking round | `experiments/05_stage2_v2` | round1 0.7767 → 0.7829 after prop; round2 0.7605 | worse than v1 → reverted to hops 1–4, 1 round |
