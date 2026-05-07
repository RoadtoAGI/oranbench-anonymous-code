# OranBench Anonymous Code Artifact

Anonymous code artifact for OranBench, a benchmark for controlled-exposure
outcome prediction in social-media marketing. The artifact reproduces the
public-data evaluation cards reported in the paper:

- C1 KuaiRand-Pure off-policy evaluation.
- C2 X5 RetailHero uplift modeling.
- C3 Open Bandit Dataset bundled-sample off-policy evaluation.
- C8 creative-aggregate outcome-regression baselines.

It also ships the action-conditioned Transformer outcome surrogate
training/evaluation pipeline (internal label `CT-WM`; the paper text refers to
this as the "action-conditioned Transformer outcome model" / "outcome
surrogate"). Released under `models/ct_wm/` are the per-seed validation
metrics for v0.1 (3-seed pilot), v0.3 (RankNet + Huber, 2-seed), and v0.4
(Pure-full random+standard with meta tower, 2-seed).

Private calibration data, private model weights, runtime state, and source
git history are intentionally excluded. Released scripts operate on public
upstream datasets or package-provided bundled samples.

## Install

Python 3.12 was used for the submitted runs.

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r requirements-oranbench.txt
python3 -m pip install --no-deps -r requirements-obp-no-deps.txt
```

For the action-conditioned Transformer outcome surrogate, additionally install
PyTorch matching your CUDA version (training was done on a single RTX 4090 /
4090D class GPU):

```bash
python3 -m pip install torch
```

## Reproduce Public Dataset Results

Prepare KuaiRand-Pure:

```bash
python3 backend/scripts/pull_kuairand.py --variant pure
python3 backend/scripts/prepare_kuairand.py
```

Prepare X5 RetailHero:

```bash
python3 backend/scripts/pull_x5.py
python3 backend/scripts/prepare_x5.py
```

Run exposure-unit baselines:

```bash
python3 backend/scripts/run_oranbench_baselines.py --cards C1,C2,C3 --seed 42
```

Run creative-aggregate baselines:

```bash
python3 backend/scripts/run_oranbench_aggregate.py \
  --datasets KuaiRand,X5,OBD \
  --baselines group_mean,linear,lightgbm,mlp \
  --seeds 42,137,256
```

Outputs are written to the path supplied via the script's `--out` argument
(default paths are listed in each script's `--help`).

## Train the Action-Conditioned Outcome Surrogate

After `prepare_kuairand.py` produces `data/kuairand/processed/clicks.parquet`,
train all three seeds and write the cross-seed `metrics.json` (replicate the
v0.1 pilot configuration shown below; full hyperparameter list is in
`train_ct_wm.py --help`):

```bash
# train all three seeds (v0.1 pilot config: d_model=128, layers=2, heads=4,
# epochs=2 ; later versions adjust these — see models/ct_wm/README.md)
for seed in 42 137 256; do
  python3 backend/scripts/train_ct_wm.py \
    --data data/kuairand/processed/clicks.parquet \
    --out  models/ct_wm/v0.1/seed_${seed} \
    --max-history 50 --d-model 128 --layers 2 --heads 4 \
    --epochs 2 --batch-size 512 --seed ${seed}
done

# evaluate every seed_*/ subdirectory under --runs-dir and emit the
# cross-seed metrics.json (e7-v0.2 schema)
python3 backend/scripts/eval_ct_wm.py \
    --runs-dir models/ct_wm/v0.1 \
    --data     data/kuairand/processed/clicks.parquet \
    --out      models/ct_wm/v0.1/metrics.json
```

Heavy `*.pt` checkpoints are not committed; only `metrics.json` files with
sha256-tagged checkpoint references, per-seed validation scores, and
cross-seed mean ± std are released. See [`models/ct_wm/README.md`](models/ct_wm/README.md)
for the v0.1 → v0.4 trajectory and per-version configuration deltas.

## Repository Layout

```
.
├── backend/
│   ├── causaltwin/
│   │   ├── __init__.py
│   │   └── paths.py            # central path constants (KuaiRand, models, runtime)
│   └── scripts/
│       ├── pull_kuairand.py
│       ├── prepare_kuairand.py
│       ├── pull_x5.py
│       ├── prepare_x5.py
│       ├── run_oranbench_baselines.py
│       ├── run_oranbench_aggregate.py
│       ├── train_ct_wm.py
│       └── eval_ct_wm.py
├── data/
│   ├── kuairand/{raw,processed}/   # populated by pull_kuairand.py + prepare_kuairand.py
│   └── x5/{raw,processed}/         # populated by pull_x5.py + prepare_x5.py
├── models/
│   └── ct_wm/
│       ├── README.md               # v0.1 → v0.4 trajectory
│       ├── v0.1/metrics.json
│       ├── v0.3/metrics.json
│       └── v0.4/metrics.json
├── requirements-oranbench.txt
└── requirements-obp-no-deps.txt
```

Path constants are centralized in
[`backend/causaltwin/paths.py`](backend/causaltwin/paths.py); override
`OSIM_DATA_DIR` / `OSIM_MODEL_DIR` for non-standard mounts.
