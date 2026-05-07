# ct_wm/

Action-conditioned Transformer outcome surrogate trained on KuaiRand-Pure.

| Version | Validation rows | click AUC | like AUC | long_view AUC | watch_time Spearman |
|---|---|---|---|---|---|
| v0.1 (3-seed pilot, d=128 / L=2 / H=4 / E=2) | 118,606 | 0.622 ± 0.001 | 0.510 ± 0.011 | 0.638 ± 0.002 | 0.174 ± 0.004 |
| v0.3 (RankNet+Huber mixed loss, 2-seed) | 118,606 | 0.643 ± 0.001 | 0.556 ± 0.009 | 0.676 ± 0.001 | 0.191 ± 0.002 |
| v0.4 (Pure full random+standard, meta tower, 2-seed) | 262,267 | 0.671 ± 0.001 | 0.760 ± 0.006 | 0.710 ± 0.001 | 0.255 ± 0.004 |

See `vX.Y/metrics.json` for per-seed checkpoints (sha256-tagged), best epoch,
best validation loss, wall-clock seconds, and hardware tags. Heavy
checkpoints (`*.pt`) are not committed.

## Reproduce v0.1 (3-seed pilot)

Prepare KuaiRand-Pure parquet under `data/kuairand/processed/clicks.parquet`
(see top-level README), then:

```bash
for seed in 42 137 256; do
  python3 backend/scripts/train_ct_wm.py \
      --data data/kuairand/processed/clicks.parquet \
      --out  models/ct_wm/v0.1/seed_${seed} \
      --max-history 50 --d-model 128 --layers 2 --heads 4 \
      --epochs 2 --batch-size 512 --seed ${seed}
done

python3 backend/scripts/eval_ct_wm.py \
    --runs-dir models/ct_wm/v0.1 \
    --data     data/kuairand/processed/clicks.parquet \
    --out      models/ct_wm/v0.1/metrics.json
```

`eval_ct_wm.py` walks every `seed_*/` subdirectory under `--runs-dir`, loads
each `checkpoint.pt`, recomputes the validation loader from the saved config,
and writes per-seed plus aggregate metrics under the e7-v0.2 schema.

## Reproduce v0.3 (RankNet + Huber mixed loss)

Same command as v0.1 with the loss flag and a longer schedule. Per-version
hyperparameters are listed in `train_ct_wm.py --help`; the v0.3 deltas vs
v0.1 are `--watch-time-loss=mixed --rank-alpha=0.7` and the d-model / layers /
heads upgrades inherited from v0.2 (`--d-model 512 --layers 6 --heads 8
--epochs 20 --pos-weight-balance --pos-weight-mode sqrt`).

## Reproduce v0.4 (Pure full + meta tower)

v0.4 trains on the full Pure slice (random + standard, ~2.6 M rows) and adds
side towers over user metadata (25 dims) and basic video metadata (4 dims).

Re-run `prepare_kuairand.py` with the meta-tower flags so the side-tower
parquet files are emitted next to `clicks.parquet`:

```bash
python3 backend/scripts/prepare_kuairand.py \
    --include-standard --include-meta
```

Then add `--user-features` / `--video-features` to the v0.3 training command:

```bash
python3 backend/scripts/train_ct_wm.py \
    --data           data/kuairand/processed/clicks.parquet \
    --user-features  data/kuairand/processed/user_features.parquet \
    --video-features data/kuairand/processed/video_features.parquet \
    --watch-time-loss mixed --rank-alpha 0.7 \
    --d-model 512 --layers 6 --heads 8 --epochs 20 \
    --pos-weight-balance --pos-weight-mode sqrt \
    --out  models/ct_wm/v0.4/seed_${seed} \
    --seed ${seed}
```

Per-seed wall time on a single RTX 4090: ~1.6 h (see `v0.4/metrics.json`).
