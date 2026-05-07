# models/

Trained model artifacts for OranBench reproduction. Heavy weight files
(`*.pt`, `*.pkl`) are not committed; only `metrics.json` and `README.md`
files describe the training schema, seeds, and evaluation outcomes.

## Subdirectories

- [`ct_wm/`](ct_wm/) — action-conditioned Transformer outcome surrogate trained
  on KuaiRand-Pure (decoder-only, causal self-attention, 4 heads × 2 layers,
  embedding dim 128, history 50, click/like/long-view classification heads
  plus a watch-time regression head). Internal version label `CT-WM`; the
  paper text uses "action-conditioned Transformer outcome model" /
  "outcome surrogate".

## Conventions

- Each subdirectory holds versioned releases (`v0.1/`, `v0.3/`, `v0.4/` ...)
  and one `metrics.json` per release listing per-seed checkpoints
  (sha256-tagged), validation AUC / Spearman / MAE / RMSE, and
  cross-seed mean ± std.
- Path constants are centralized in
  [`backend/causaltwin/paths.py`](../backend/causaltwin/paths.py).
- Loaders read `MODEL_DIR` (env var `OSIM_MODEL_DIR`, default `<repo>/models`).
