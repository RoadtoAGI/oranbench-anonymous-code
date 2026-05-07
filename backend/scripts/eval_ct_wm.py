"""Evaluate per-seed CT-WM checkpoints and write OranBench E7 ``metrics.json``.

Design SSOT: ``docs/level-3-implementation/research/oranbench-neurips-2026/ct-wm-training-plan.md`` §5–§6.
Companion to ``train_ct_wm.py``.

This script reports raw per-seed and aggregated diagnostic scores. There are no
sanity thresholds and no pass/fail verdict: any prior 0.4 watch-time Spearman
threshold or 0.65 binary-AUC threshold has been retired (no published
literature anchor; engineering sanity threshold removed 2026-05-07).

Inputs
------

- ``--runs-dir``: directory containing one ``seed_<n>/checkpoint.pt`` per seed
  produced by ``train_ct_wm.py`` (e.g. ``models/ct_wm/v0.1``).
- ``--data``: same KuaiRand-Pure parquet used for training; the same
  deterministic temporal split is rebuilt to score the held-out tail.

Output
------

A single ``metrics.json`` shaped for OranBench E7 with both per-seed values and
aggregates. Schema::

    {
      "schema_version": "e7-v0.2",
      "data": "...",
      "split": {"rule": "temporal_tail", "val_frac": 0.1},
      "seeds": {
        "42":  {"hardware": "...", "wall_seconds": ...,
                "click_auc": ..., "like_auc": ..., "long_view_auc": ...,
                "watch_time_mae": ..., "watch_time_rmse": ...,
                "watch_time_spearman": ...,
                "checkpoint_sha256": "...",
                "best_epoch": ..., "best_val_loss": ...},
        "137": {...},
        "256": {...}
      },
      "aggregate": {
        "click_auc_mean": ..., "click_auc_std": ...,
        "like_auc_mean": ..., "like_auc_std": ...,
        "long_view_auc_mean": ..., "long_view_auc_std": ...,
        "watch_time_spearman_mean": ..., "watch_time_spearman_std": ...,
        "watch_time_mae_mean": ..., "watch_time_rmse_mean": ...
      }
    }

The file is consumed by the Phase B P2 decision step in
``e7-execution-log.md`` (§Tier 1/2/3 routing).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from causaltwin.paths import KUAIRAND_PROCESSED, MODEL_DIR  # noqa: E402

DEFAULT_DATA = KUAIRAND_PROCESSED / "clicks.parquet"
DEFAULT_RUNS_DIR = MODEL_DIR / "ct_wm" / "v0.1"
DEFAULT_OUT = DEFAULT_RUNS_DIR / "metrics.json"

PAD_IDX = 0


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def rebuild_val_loader(df: pd.DataFrame, ckpt_payload: dict, max_history: int, batch_size: int, val_frac: float, cfg=None):
    """Reapply the deterministic temporal split used in training."""
    import torch
    from torch.utils.data import DataLoader

    # Re-import the helpers from train_ct_wm to keep one source of truth.
    from train_ct_wm import build_user_sequences, collate, temporal_split, load_meta_features  # type: ignore

    action_to_int: dict = ckpt_payload["action_to_int"]
    df = df.copy()
    df["action_id_int"] = df["action_id"].astype(str).map(action_to_int)
    df = df.dropna(subset=["action_id_int"]).reset_index(drop=True)
    df["action_id_int"] = df["action_id_int"].astype(np.int64)
    df["log1p_watch_time"] = np.log1p(df["reward_watch_time"].astype(float))
    user_feat_dict = video_feat_dict = None
    if cfg is not None and getattr(cfg, "user_features", "") and getattr(cfg, "video_features", ""):
        meta = load_meta_features(cfg.user_features, cfg.video_features)
        if meta:
            user_feat_dict, video_feat_dict, _ = meta
    max_rows_per_user = int(getattr(cfg, "max_rows_per_user", 0) or 0) if cfg is not None else 0
    samples = build_user_sequences(
        df, max_history=max_history,
        max_users=0,
        max_rows_per_user=max_rows_per_user,
        user_features_dict=user_feat_dict,
        video_features_dict=video_feat_dict,
    )
    _, val_samples = temporal_split(samples, val_frac=val_frac, df=df)
    loader = DataLoader(
        val_samples,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=lambda b: collate(b, max_history),
    )
    return loader


def evaluate_seed(ckpt_path: Path, df: pd.DataFrame, device_str: str) -> dict:
    import torch
    from sklearn.metrics import roc_auc_score, mean_absolute_error, mean_squared_error
    from scipy.stats import spearmanr

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train_ct_wm import build_model, TrainConfig  # type: ignore

    payload = torch.load(ckpt_path, map_location="cpu")
    cfg = TrainConfig(**payload["config"])
    n_actions = payload["n_actions"]
    device = torch.device("cuda" if (device_str == "auto" and torch.cuda.is_available()) else (device_str if device_str != "auto" else "cpu"))
    model = build_model(n_actions=n_actions, cfg=cfg).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()

    val_loader = rebuild_val_loader(df, payload, cfg.max_history, cfg.batch_size, cfg.val_frac, cfg=cfg)

    preds = {"click": [], "like": [], "long_view": [], "watch_time": []}
    targets = {"click": [], "like": [], "long_view": [], "watch_time": []}
    with torch.no_grad():
        for batch in val_loader:
            tensors = [b.to(device) for b in batch]
            if len(tensors) == 9:
                hist, mask, actions, rc, rl, rlv, rwt, user_feat, video_feat = tensors
                out = model(hist, mask, actions, user_feat=user_feat, video_feat=video_feat)
            else:
                hist, mask, actions, rc, rl, rlv, rwt = tensors
                out = model(hist, mask, actions)
            preds["click"].append(torch.sigmoid(out["click"]).cpu().numpy())
            preds["like"].append(torch.sigmoid(out["like"]).cpu().numpy())
            preds["long_view"].append(torch.sigmoid(out["long_view"]).cpu().numpy())
            preds["watch_time"].append(out["watch_time"].cpu().numpy())
            targets["click"].append(rc.cpu().numpy())
            targets["like"].append(rl.cpu().numpy())
            targets["long_view"].append(rlv.cpu().numpy())
            targets["watch_time"].append(rwt.cpu().numpy())

    for k in preds:
        preds[k] = np.concatenate(preds[k])
        targets[k] = np.concatenate(targets[k])

    def safe_auc(y, p) -> float:
        if len(np.unique(y)) < 2:
            return float("nan")
        return float(roc_auc_score(y, p))

    rho, _ = spearmanr(targets["watch_time"], preds["watch_time"])

    return {
        "checkpoint": str(ckpt_path),
        "checkpoint_sha256": sha256_file(ckpt_path),
        "best_epoch": int(payload.get("epoch", -1)),
        "best_val_loss": float(payload.get("val_loss", float("nan"))),
        "n_val": int(len(preds["click"])),
        "click_auc": safe_auc(targets["click"], preds["click"]),
        "like_auc": safe_auc(targets["like"], preds["like"]),
        "long_view_auc": safe_auc(targets["long_view"], preds["long_view"]),
        "watch_time_mae": float(mean_absolute_error(targets["watch_time"], preds["watch_time"])),
        "watch_time_rmse": float(np.sqrt(mean_squared_error(targets["watch_time"], preds["watch_time"]))),
        "watch_time_spearman": float(rho) if rho == rho else float("nan"),
    }


def aggregate(per_seed: dict[str, dict]) -> dict:
    keys = ("click_auc", "like_auc", "long_view_auc", "watch_time_mae", "watch_time_rmse", "watch_time_spearman")
    out = {}
    for k in keys:
        vals = np.array([v[k] for v in per_seed.values() if v[k] == v[k]])  # drop NaN
        out[f"{k}_mean"] = float(vals.mean()) if len(vals) else float("nan")
        out[f"{k}_std"] = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--hardware-tag", type=str, default="",
                        help="Free-text hardware label written into metrics.json (e.g. 'autodl RTX 4090').")
    args = parser.parse_args()

    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "PyTorch is required to evaluate. This script is staged on the dev "
            f"workstation; run on autodl per train_ct_wm.py docstring. {exc}"
        )

    seed_dirs = sorted(p for p in args.runs_dir.glob("seed_*") if (p / "checkpoint.pt").exists())
    if not seed_dirs:
        raise SystemExit(f"No checkpoints under {args.runs_dir}/seed_*/checkpoint.pt")

    df = pd.read_parquet(args.data)
    df = df.dropna(subset=["reward_click", "reward_like", "reward_long_view", "reward_watch_time"]).reset_index(drop=True)

    per_seed: dict[str, dict] = {}
    for sd in seed_dirs:
        seed_str = sd.name.replace("seed_", "")
        print(f"[ct-wm-eval] evaluating {sd}")
        per = evaluate_seed(sd / "checkpoint.pt", df, args.device)
        # Pull wall-time from training run_config.json if present.
        rc_path = sd / "run_config.json"
        if rc_path.exists():
            rc = json.loads(rc_path.read_text())
            per["wall_seconds"] = rc.get("wall_seconds")
            per["device"] = rc.get("device")
        per["hardware_tag"] = args.hardware_tag or per.get("device", "")
        per_seed[seed_str] = per

    agg = aggregate(per_seed)

    metrics = {
        "schema_version": "e7-v0.2",
        "data": str(args.data),
        "split": {"rule": "temporal_tail", "val_frac": 0.1},
        "seeds": per_seed,
        "aggregate": agg,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"[ct-wm-eval] wrote {args.out}")
    print(json.dumps(agg, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
