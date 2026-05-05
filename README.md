# OranBench Anonymous Code Artifact

This repository contains the anonymous code artifact for OranBench, a benchmark
for controlled-exposure outcome prediction in social-media marketing.

The artifact includes public-data reproduction scripts for:

- C1 KuaiRand-Pure off-policy evaluation.
- C2 X5 RetailHero uplift modeling.
- C3 Open Bandit Dataset bundled-sample off-policy evaluation.
- C8 creative-aggregate outcome-regression baselines.

Private calibration data, private model files, runtime state, and git history
are intentionally excluded. The released scripts operate on public upstream
datasets or package-provided bundled samples.

## Install

Python 3.12 was used for the submitted runs.

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r requirements-oranbench.txt
python3 -m pip install --no-deps -r requirements-obp-no-deps.txt
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

Committed JSON files under
`docs/level-1-foundation/research/proposal-2026-neurips/results/` contain the
submitted fixed-seed outputs for inspection.
