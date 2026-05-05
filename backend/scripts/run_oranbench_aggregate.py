"""Run OranBench creative-aggregate outcome-regression baselines (E8 / C8).

Fills tab:agg-regression in paper/neurips_2026_template/oranbench.tex by
fitting 4 baselines (group_mean, linear, lightgbm, mlp) on per-creative
aggregate outcomes for KuaiRand-Pure, X5 RetailHero, and OBD bundled.

Metric: RMSE on a fixed 70/15/15 split (split key per dataset). Multi-seed
mean ± std reported across {42, 137, 256} by default.

Usage::

    python3 backend/scripts/run_oranbench_aggregate.py \\
        --datasets KuaiRand,X5,OBD \\
        --baselines group_mean,linear,lightgbm,mlp \\
        --seeds 42,137,256 \\
        --out docs/level-1-foundation/research/proposal-2026-neurips/results/c8_aggregate_regression.json

Design / decisions: see
docs/level-3-implementation/research/oranbench-neurips-2026/e8-idea-to-proposal.md
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from causaltwin.paths import DATA_DIR, REPO_ROOT


PROPOSAL_DIR = REPO_ROOT / "docs" / "level-1-foundation" / "research" / "proposal-2026-neurips"
DEFAULT_OUT = PROPOSAL_DIR / "results" / "c8_aggregate_regression.json"


# ---------- generic helpers ----------


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[oranbench-agg] wrote {path}")


def split_indices(n: int, seed: int, ratios: tuple[float, float, float] = (0.70, 0.15, 0.15)) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_train = int(round(n * ratios[0]))
    n_val = int(round(n * ratios[1]))
    train = idx[:n_train]
    val = idx[n_train : n_train + n_val]
    test = idx[n_train + n_val :]
    return train, val, test


# ---------- baselines ----------


def fit_group_mean(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, seed: int) -> np.ndarray:
    return np.full(len(X_test), float(np.mean(y_train)))


def fit_linear(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, seed: int) -> np.ndarray:
    from sklearn.linear_model import LinearRegression
    if X_train.shape[1] == 0:
        return fit_group_mean(X_train, y_train, X_test, seed)
    # F3: when train aggregate rows are too few to identify regression
    # coefficients (rows <= n_features+1), fall back to group mean. Any
    # fit on n=2 with multiple features produces unbounded extrapolation
    # on test features outside the train hull. This makes group_mean and
    # the other baselines collapse to the same value on the X5 aggregate
    # split (documented in the execution log under F3/F4/F5).
    if len(X_train) <= X_train.shape[1] + 1:
        return fit_group_mean(X_train, y_train, X_test, seed)
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model.predict(X_test)


def fit_lightgbm(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, seed: int) -> np.ndarray:
    import lightgbm as lgb
    if X_train.shape[1] == 0 or len(X_train) <= X_train.shape[1] + 1:
        # F4: degenerate-N fallback (matches fit_linear / fit_mlp).
        return fit_group_mean(X_train, y_train, X_test, seed)
    min_child = 1 if len(X_train) < 50 else 5
    model = lgb.LGBMRegressor(
        n_estimators=200,
        num_leaves=31,
        min_child_samples=min_child,
        learning_rate=0.05,
        random_state=seed,
        verbosity=-1,
    )
    model.fit(X_train, y_train)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names",
            category=UserWarning,
        )
        return model.predict(X_test)


def fit_mlp(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, seed: int) -> np.ndarray:
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    if X_train.shape[1] == 0 or len(X_train) <= X_train.shape[1] + 1:
        # F5: MLP on tiny train sets (X5 aggregate train_n=2) cannot
        # generalize; fall back to group mean. Same guard as fit_linear /
        # fit_lightgbm so the three model baselines collapse to the same
        # value on degenerate splits.
        return fit_group_mean(X_train, y_train, X_test, seed)
    # Standardize both features and target. Target standardization is
    # essential when target rates are very small (e.g. OBD click rate
    # ~ 0.005); without it, MLP's default initialization and L2
    # regularization (alpha=1e-4) produce biased predictions on the
    # output scale.
    feat_scaler = StandardScaler()
    Xt = feat_scaler.fit_transform(X_train)
    Xv = feat_scaler.transform(X_test)
    y_mean = float(np.mean(y_train))
    y_std = float(np.std(y_train))
    if y_std <= 1e-12:
        # constant target -> group mean
        return np.full(len(X_test), y_mean)
    yt = (y_train - y_mean) / y_std
    early_stop = len(X_train) >= 20
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = MLPRegressor(
            hidden_layer_sizes=(16, 8),
            max_iter=500,
            early_stopping=early_stop,
            random_state=seed,
        )
        model.fit(Xt, yt)
        return model.predict(Xv) * y_std + y_mean


BASELINES: dict[str, Callable[[np.ndarray, np.ndarray, np.ndarray, int], np.ndarray]] = {
    "group_mean": fit_group_mean,
    "linear": fit_linear,
    "lightgbm": fit_lightgbm,
    "mlp": fit_mlp,
}


# ---------- dataset loaders ----------
X5_BIN_SOURCE_COLS = ["X_age", "X_purchase_count", "X_total_spend", "X_recency_days"]
X5_NUMERIC_MEAN_COLS = [
    "X_age",
    "X_purchase_count",
    "X_total_spend",
    "X_unique_products",
    "X_unique_stores",
    "X_avg_spend",
    "X_recency_days",
]
X5_BIN_COLS = [f"{c}_bin" for c in X5_BIN_SOURCE_COLS]
X5_FEATURE_COLS = (
    ["treatment", "n_clients", "propensity_estimate_mean"]
    + X5_BIN_COLS
    + [f"{c}_mean" for c in X5_NUMERIC_MEAN_COLS]
    + ["gender_f_share", "gender_m_share", "gender_u_share"]
)


def add_rank_bins(df: pd.DataFrame, cols: list[str], n_bins: int = 5) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        values = pd.to_numeric(df[col], errors="coerce")
        filled = values.fillna(float(values.median()))
        ranked = filled.rank(method="first")
        df[f"{col}_bin"] = pd.qcut(ranked, q=n_bins, labels=False, duplicates="drop").astype(int)
    return df


def load_kuairand_aggregate(min_exposures: int = 50) -> tuple[pd.DataFrame, list[str], str, dict[str, Any]]:
    path = DATA_DIR / "kuairand" / "processed" / "clicks.parquet"
    df = pd.read_parquet(path)
    long_view = pd.to_numeric(df["reward_long_view"], errors="coerce").fillna(0.0)
    df = df.assign(_lv=long_view)
    grouped = (
        df.groupby("item_id")
        .agg(
            n_exposures=("user_id", "size"),
            n_users=("user_id", "nunique"),
            long_view_rate=("_lv", "mean"),
        )
        .reset_index()
    )
    pre_n = len(grouped)
    grouped = grouped[grouped["n_exposures"] >= min_exposures].copy()
    feature_cols = ["n_exposures", "n_users"]
    target_col = "long_view_rate"
    info = {
        "source": str(path),
        "rows_unit_layer": int(len(df)),
        "items_pre_filter": int(pre_n),
        "items_post_filter": int(len(grouped)),
        "min_exposures": int(min_exposures),
        "split_key": "item_id",
        "target": "mean(reward_long_view) per item",
        "features": feature_cols,
    }
    return grouped, feature_cols, target_col, info


def load_x5_aggregate() -> tuple[pd.DataFrame, list[str], str, dict[str, Any]]:
    """X5 aggregate layer is a treatment-conditioned customer-cell rate.

    The split is at the client row level (70/15/15). Within each split we
    aggregate by treatment plus rank-quantile bins over non-outcome customer
    covariates. This keeps the creative-aggregate task public-data only while
    avoiding the old two-row treatment/control aggregate degeneration.
    """
    path = DATA_DIR / "x5" / "processed" / "uplift.parquet"
    df = add_rank_bins(pd.read_parquet(path), X5_BIN_SOURCE_COLS)
    info = {
        "source": str(path),
        "rows_unit_layer": int(len(df)),
        "split_key": "client_id row split; aggregate by treatment plus rank-quantile customer cells within each split",
        "target": "mean(outcome) per (subsplit, treatment, customer-cell) group",
        "aggregation": {
            "treatment_col": "treatment",
            "bin_source_cols": X5_BIN_SOURCE_COLS,
            "binning": "5 rank-quantile bins per source column, computed from non-outcome covariates",
        },
        "features": X5_FEATURE_COLS,
    }
    return df, [], "outcome", info


def build_x5_aggregate_for_split(df: pd.DataFrame, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    sub = df.iloc[idx]
    group_cols = ["treatment"] + X5_BIN_COLS
    rows: list[dict[str, Any]] = []
    for key, g in sub.groupby(group_cols, observed=True):
        key_values = key if isinstance(key, tuple) else (key,)
        row: dict[str, Any] = dict(zip(group_cols, key_values))
        row["n_clients"] = int(len(g))
        row["propensity_estimate_mean"] = float(pd.to_numeric(g["propensity_estimate"], errors="coerce").mean())
        row["outcome_rate"] = float(pd.to_numeric(g["outcome"], errors="coerce").mean())
        for col in X5_NUMERIC_MEAN_COLS:
            row[f"{col}_mean"] = float(pd.to_numeric(g[col], errors="coerce").mean())
        gender = g["X_gender"].astype(str)
        row["gender_f_share"] = float((gender == "F").mean())
        row["gender_m_share"] = float((gender == "M").mean())
        row["gender_u_share"] = float((gender == "U").mean())
        rows.append(row)
    agg = pd.DataFrame(rows)
    X = agg[X5_FEATURE_COLS].to_numpy(dtype=float)
    y = agg["outcome_rate"].to_numpy(dtype=float)
    return X, y


def load_obd_aggregate(min_exposures: int = 5) -> tuple[pd.DataFrame, list[str], str, dict[str, Any]]:
    """Load the OBP bundled OBD sample for both behavior policies and
    aggregate to (item, behavior_policy) pairs."""
    # OBP 0.4.1 vs pandas 3 compat shim (mirrors the deleted
    # run_oranbench_baselines.patch_obp_pandas_compat()).
    _patch_obp_pandas_compat()

    from obp.dataset import OpenBanditDataset

    pieces = []
    for behavior in ("random", "bts"):
        ds = OpenBanditDataset(behavior_policy=behavior, campaign="all")
        feedback = ds.obtain_batch_bandit_feedback()
        action = feedback["action"]
        reward = feedback["reward"]
        position = feedback["position"]
        pscore = feedback["pscore"]
        pieces.append(
            pd.DataFrame(
                {
                    "item_id": action.astype(int),
                    "behavior_policy": behavior,
                    "reward": reward.astype(float),
                    "position": position.astype(int),
                    "pscore": pscore.astype(float),
                }
            )
        )
    df = pd.concat(pieces, ignore_index=True)
    grouped = (
        df.groupby(["item_id", "behavior_policy"])
        .agg(
            n_exposures=("reward", "size"),
            click_rate=("reward", "mean"),
            mean_position=("position", "mean"),
            mean_pscore=("pscore", "mean"),
        )
        .reset_index()
    )
    pre_n = len(grouped)
    grouped = grouped[grouped["n_exposures"] >= min_exposures].copy()
    grouped["policy_random"] = (grouped["behavior_policy"] == "random").astype(int)
    feature_cols = ["n_exposures", "mean_position", "mean_pscore", "policy_random"]
    target_col = "click_rate"
    info = {
        "source": "obp.dataset.OpenBanditDataset(behavior_policy in {random,bts}, campaign=all)",
        "rows_unit_layer": int(len(df)),
        "pairs_pre_filter": int(pre_n),
        "pairs_post_filter": int(len(grouped)),
        "min_exposures": int(min_exposures),
        "split_key": "(item_id, behavior_policy) pair",
        "target": "mean(reward) per (item, behavior_policy)",
        "features": feature_cols,
    }
    return grouped, feature_cols, target_col, info


def _patch_obp_pandas_compat() -> list[str]:
    """Patch pandas 3 positional-argument breaks in OBP 0.4.1's bundled loader.

    Mirrors backend/scripts/run_oranbench_baselines.patch_obp_pandas_compat()
    (deleted file but kept as reference in git for exact behavior).
    """
    applied: list[str] = []
    orig_drop = pd.DataFrame.drop

    def drop_compat(self, labels=None, axis=0, index=None, columns=None, level=None, inplace=False, errors="raise"):
        return orig_drop(self, labels=labels, axis=axis, index=index, columns=columns, level=level, inplace=inplace, errors=errors)

    pd.DataFrame.drop = drop_compat
    applied.append("DataFrame.drop(axis positional -> keyword)")

    orig_concat = pd.concat

    def concat_compat(objs, axis=0, join="outer", ignore_index=False, keys=None, levels=None, names=None, verify_integrity=False, sort=False, copy=None):
        return orig_concat(
            objs,
            axis=axis,
            join=join,
            ignore_index=ignore_index,
            keys=keys,
            levels=levels,
            names=names,
            verify_integrity=verify_integrity,
            sort=sort,
            copy=copy,
        )

    pd.concat = concat_compat
    applied.append("pandas.concat(axis positional -> keyword)")
    return applied


# ---------- evaluation harness ----------


def eval_dataset_kuairand(seeds: list[int], baselines: list[str]) -> dict[str, Any]:
    df, feature_cols, target_col, info = load_kuairand_aggregate()
    X = df[feature_cols].to_numpy(dtype=float)
    y = df[target_col].to_numpy(dtype=float)
    return _run_seeds_holdout(X, y, seeds, baselines, info)


def eval_dataset_obd(seeds: list[int], baselines: list[str]) -> dict[str, Any]:
    df, feature_cols, target_col, info = load_obd_aggregate()
    X = df[feature_cols].to_numpy(dtype=float)
    y = df[target_col].to_numpy(dtype=float)
    return _run_seeds_holdout(X, y, seeds, baselines, info)


def eval_dataset_x5(seeds: list[int], baselines: list[str]) -> dict[str, Any]:
    df, _, _, info = load_x5_aggregate()
    per_baseline_seed_values: dict[str, list[float]] = {b: [] for b in baselines}
    diagnostics: list[dict[str, Any]] = []
    for seed in seeds:
        train_idx, val_idx, test_idx = split_indices(len(df), seed)
        X_train, y_train = build_x5_aggregate_for_split(df, train_idx)
        X_test, y_test = build_x5_aggregate_for_split(df, test_idx)
        diag = {
            "seed": int(seed),
            "client_train_n": int(len(train_idx)),
            "client_test_n": int(len(test_idx)),
            "agg_train_rows": int(len(X_train)),
            "agg_test_rows": int(len(X_test)),
            "test_target_mean": float(np.mean(y_test)) if len(y_test) else None,
            "test_target_std": float(np.std(y_test)) if len(y_test) else None,
        }
        diagnostics.append(diag)
        for name in baselines:
            y_pred = BASELINES[name](X_train, y_train, X_test, seed)
            per_baseline_seed_values[name].append(rmse(y_test, y_pred))
    return _summarize(per_baseline_seed_values, info | {"per_seed_diagnostics": diagnostics})


def _run_seeds_holdout(X: np.ndarray, y: np.ndarray, seeds: list[int], baselines: list[str], info: dict[str, Any]) -> dict[str, Any]:
    n = len(y)
    per_baseline_seed_values: dict[str, list[float]] = {b: [] for b in baselines}
    diagnostics: list[dict[str, Any]] = []
    for seed in seeds:
        train_idx, val_idx, test_idx = split_indices(n, seed)
        X_train = X[train_idx]
        y_train = y[train_idx]
        X_test = X[test_idx]
        y_test = y[test_idx]
        diag = {
            "seed": int(seed),
            "train_n": int(len(train_idx)),
            "val_n": int(len(val_idx)),
            "test_n": int(len(test_idx)),
            "test_target_mean": float(np.mean(y_test)) if len(y_test) else None,
            "test_target_std": float(np.std(y_test)) if len(y_test) else None,
        }
        diagnostics.append(diag)
        for name in baselines:
            y_pred = BASELINES[name](X_train, y_train, X_test, seed)
            per_baseline_seed_values[name].append(rmse(y_test, y_pred))
    return _summarize(per_baseline_seed_values, info | {"per_seed_diagnostics": diagnostics})


def _summarize(per_baseline_seed_values: dict[str, list[float]], info: dict[str, Any]) -> dict[str, Any]:
    """Build a dataset-card payload. Per the E8 prompt spec, baseline rows
    live directly under cards.<Dataset>.<baseline>; the auxiliary `info`
    block (split sizes, filter counts, per-seed diagnostics) is kept under
    cards.<Dataset>.info for reproducibility.
    """
    out: dict[str, Any] = {"info": info}
    for name, vals in per_baseline_seed_values.items():
        arr = np.asarray(vals, dtype=float)
        out[name] = {
            "metric_name": "RMSE",
            "values_per_seed": [float(v) for v in arr.tolist()],
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        }
    return out


# ---------- runner ----------


DATASETS = {
    "KuaiRand": eval_dataset_kuairand,
    "X5": eval_dataset_x5,
    "OBD": eval_dataset_obd,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="KuaiRand,X5,OBD")
    parser.add_argument("--baselines", default="group_mean,linear,lightgbm,mlp")
    parser.add_argument("--seeds", default="42,137,256")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    baselines = [b.strip() for b in args.baselines.split(",") if b.strip()]
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]

    payload: dict[str, Any] = {
        "card": "C8",
        "task": "creative-aggregate outcome regression",
        "metric": "RMSE",
        "metric_note": (
            "RMSE is reported on the per-creative aggregate target rate. "
            "Targets are dataset-specific (long-view rate, conversion rate, "
            "click rate); RMSE values across datasets are not directly comparable."
        ),
        "seeds": seeds,
        "baselines_run": baselines,
        "split": "70/15/15 holdout, split key per dataset (see cards.<dataset>.info.split_key)",
        "cards": {},
    }

    for name in datasets:
        if name not in DATASETS:
            raise SystemExit(f"unknown dataset {name}; choices = {sorted(DATASETS)}")
        print(f"[oranbench-agg] running {name} ...")
        try:
            block = DATASETS[name](seeds, baselines)
            payload["cards"][name] = block
            for b in baselines:
                row = block.get(b)
                if row is None:
                    continue
                print(f"  {name:9s} {b:11s} RMSE = {row['mean']:.6f} ± {row['std']:.6f}")
        except Exception as exc:  # F7 fallback
            fallback = {"info": {"error": f"{type(exc).__name__}: {exc}"}}
            for b in baselines:
                fallback[b] = {"metric_name": "RMSE", "values_per_seed": [], "mean": None, "std": None}
            payload["cards"][name] = fallback
            print(f"  {name} FAILED: {exc}")

    write_json(args.out, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
