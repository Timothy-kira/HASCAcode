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
| 3 | 2026-09-24 | **stage2 v1** — graph-context stacking: learned pair model (chain v1) gives soft succ/pred graph; features = own probs & IMU + graph-aggregated probs (1/2/4 hops) & IMU + video-kNN aggregates → LightGBM, then 1 propagation step (α=0.5), null ×0.5 | `experiments/04_stage2_v1` (+ pairs from `03_chain_v1`) | 0.787 | 0.79079 |
| 4 | 2026-09-25 | **stage2 v4** — stage2 v1 design with 2 bases averaged (LGBM + fusion NN) on chain v3 graph (soft pair graph + Hungarian-assigned edge hops 1–16) + video kNN, propagation α=0.7, null ×0.6 | `experiments/11_stage2_v4` (+ `08_fusion_v1`, `09_chain_v3`) | 0.806 | 0.81179 |
| 5 | 2026-09-25 | **stage2 v5** — as v4 but 4 bases averaged (lgb, nn, lgb2, nn2; base avg OOF 0.700) | `experiments/14_stage2_v5` | 0.8067 | 0.79690 |
| 6 | 2026-09-25 | **post v1** — stage2 v6 OOF/test probs → 1 propagation step (α=0.7) → per-subject Sinkhorn class-prior balancing towards null τ=0.45 / 18 activities equal, strength λ=1, null ×0.4 | `experiments/18_post_v1` (+ `17_stage2_v6`) | 0.840 | 0.85074 |
| 7 | 2026-09-25 | **post v2** — free-null balancing (equalise the 18 activity shares per subject, null share left to the model), λ=1, 2 rounds of balance → propagate, null ×0.5 | `experiments/19_post_v2` (+ `17_stage2_v6`) | 0.855 | 0.87865 |
| 8 | 2026-09-25 | **post v3** — post v2 applied on stage2 v7 (bases balanced before stacking): free-null balancing λ=1, 2 rounds balance → propagate, null ×0.6 | `experiments/21_post_v3` (+ `20_stage2_v7`) | 0.864 | **0.88865** |

> Note: v4 → v5 had equal OOF (0.806 vs 0.807) but LB 0.812 vs 0.797 — public LB noise is about ±0.015; decide on OOF.

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
| 12 | lgb2 v1 — LightGBM base #2: lr 0.03, num_leaves 31, min_data 100, feature_fraction 0.3, early stop 150, valid-sensor sampling, v1 eval sensor | `experiments/12_lgb2_v1` | OOF 0.661 (null ×0.3) vs lgb v1 0.634 | better single base → add to stage2 |
| 13 | fusion2 v1 — fusion NN, 6 epochs (v1 peaked early), 2-seed average | `experiments/13_fusion2_v1` | OOF 0.669 (null ×0.2) vs fusion v1 0.649 | better → add to stage2 |
| 15 | diag v2 — ceilings for long-range context + spectral clustering (base = lgb only) | `experiments/15_diag_v2` | real graph 0.784 · (A) perfect graph hops≤64 **0.826** · (B) oracle true-segment vote **0.871** · (C) clusters: size 20 purity 0.895 → 0.781, size 40 purity 0.849 → 0.781, size 80 purity 0.765 → 0.784, all 3 scales → 0.791 | segment-level voting is worth up to +0.09; unsupervised clusters too small/impure to deliver it → need a supervised same-segment model |
| 16 | segment v1 — supervised same-label neighbour model: 60 video-nearest within-subject neighbours, LGBM on video/IMU/base-prob agreement + successor prob; learned-weight aggregation of base probs (γ=1/4/16, 1–2 hops, thresholds) | `experiments/16_segment_v1` | same-label AUC 0.902; p≥0.9 precision 0.947 (14.6 nbrs/tile); direct argmax of aggregated probs 0.756 (base avg 0.700) | useful → feed as stage2 features (v6) |
| 17 | stage2 v6 — v5 + 154 segment-context features (segment v1) | `experiments/17_stage2_v6` | OOF 0.799 → 0.8065 after propagation (v5 0.8067) | no gain: segment features redundant with graph features → stage2 saturated ≈0.806 |
| 20 | stage2 v7 — v6 with per-subject free-null balancing of the base probs before building context | `experiments/20_stage2_v7` | base avg 0.700 → 0.724; stage2 OOF 0.8256 → **0.830** after propagation (v6 0.8065) | +0.024 → post v3 runs on top |
| 05 | stage2 v2 — hops up to 16 + 2nd stacking round | `experiments/05_stage2_v2` | round1 0.7767 → 0.7829 after prop; round2 0.7605 | worse than v1 → reverted to hops 1–4, 1 round |
