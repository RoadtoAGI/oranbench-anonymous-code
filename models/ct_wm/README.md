# ct_wm/

Action-conditioned Transformer outcome surrogate trained on KuaiRand-Pure.

| Version | Validation rows | click AUC | like AUC | long_view AUC | watch_time Spearman |
|---|---|---|---|---|---|
| v0.1 (3-seed pilot) | 118,606 | 0.622 ± 0.001 | 0.510 ± 0.011 | 0.638 ± 0.002 | 0.174 ± 0.004 |
| v0.3 (RankNet+Huber, 2-seed) | 118,606 | 0.643 ± 0.001 | 0.556 ± 0.009 | 0.676 ± 0.001 | 0.191 ± 0.002 |
| v0.4 (Pure full random+standard, meta tower, 2-seed) | 262,267 | 0.671 ± 0.001 | 0.760 ± 0.006 | 0.710 ± 0.001 | 0.255 ± 0.004 |

See `vX.Y/metrics.json` for per-seed checkpoints (sha256-tagged), best epoch,
best validation loss, wall-clock seconds, and hardware tags. Heavy
checkpoints (`*.pt`) are not committed.

## Reproduce

Prepare KuaiRand-Pure parquet under `data/kuairand/processed/clicks.parquet`
(see top-level README), then:

```bash
python3 backend/scripts/train_ct_wm.py --seed 42  --out-dir models/ct_wm/v0.4/seed_42
python3 backend/scripts/train_ct_wm.py --seed 137 --out-dir models/ct_wm/v0.4/seed_137
python3 backend/scripts/eval_ct_wm.py \
    --checkpoints models/ct_wm/v0.4/seed_42/checkpoint.pt \
                  models/ct_wm/v0.4/seed_137/checkpoint.pt \
    --out models/ct_wm/v0.4/metrics.json
```

The training script SSOT for hyperparameters, schema, and historical
revisions is
[`docs/level-3-implementation/research/oranbench-neurips-2026/ct-wm-training-plan.md`](../../docs/level-3-implementation/research/oranbench-neurips-2026/ct-wm-training-plan.md).
