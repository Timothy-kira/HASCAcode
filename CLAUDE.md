# HASCAcode — 3rd WEAR Dataset Challenge @HASCA 2026 (Kaggle)

- All data processing / training runs on **Kaggle CPU notebooks** (account `evelynyang02`, LB name Evelyn_Yang_02); never download competition data locally.
  Write percent-format `kaggle/<step>/<name>.py`, convert with `python kaggle/py2nb.py <file>.py`, push with `kaggle kernels push -p kaggle/<step>`, read results with `kaggle/fetchlog.sh <kernel-slug>`.
- Use only this one Kaggle account (competition rules forbid multiple accounts).
- **Every scheme must be recorded**: after each run add a row to `docs/submissions.md` (scheme, OOF, public LB or "not submitted" + conclusion)
  and snapshot the exact code + Kaggle log into `experiments/<nn>_<name>_v<k>/` (`COMMIT` file = source commit). Commit and push.
- Validation = subject-grouped 5-fold on tiles built like the test set; metric macro-F1 (19 classes incl. null).
