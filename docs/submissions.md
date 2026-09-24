# Submission log

| date (UTC) | kernel / version | description | OOF macro-F1 | public LB |
|---|---|---|---|---|
| 2026-09-24 | baseline v1 | LightGBM, IMU hand features + VideoMAE PCA (raw / subject-centred / std), null prob x0.3 | 0.634 | 0.646 |
| 2026-09-24 | knn v1 | baseline + successor-graph smoothing (k=3, alpha=0.8, 3 iters) | 0.720 | 0.694 |
