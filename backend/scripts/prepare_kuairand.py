"""Prepare KuaiRand raw CSV files into an OranBench C1 parquet table.

The script is intentionally schema-tolerant because KuaiRand Pure/1K/27K ship
slightly different side files. It only processes CSVs that contain both a user
identifier and an item/video identifier.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from causaltwin.paths import KUAIRAND_PROCESSED, KUAIRAND_RAW


USER_COLS = ["user_id", "userId", "uid", "user", "userID"]
ITEM_COLS = ["item_id", "video_id", "videoId", "photo_id", "vid", "item", "itemID"]
TIME_COLS = ["timestamp", "time", "time_ms", "ts", "date", "datetime"]
PROPENSITY_COLS = ["propensity", "pscore", "prob", "logging_prob", "policy_prob", "action_prob"]
REWARD_ALIASES = {
    "reward_click": ["click", "is_click", "clicked"],
    "reward_like": ["like", "is_like", "liked"],
    "reward_long_view": ["long_view", "is_long_view", "longview"],
    "reward_watch_time": ["watch_time", "watch_time_ms", "play_time", "play_time_ms", "play_duration"],
}
KUAIRAND_PURE_RANDOM_POOL_SIZE = 7_583
KUAIRAND_PURE_UNIFORM_PROPENSITY = 1.0 / KUAIRAND_PURE_RANDOM_POOL_SIZE


def first_present(columns: list[str], candidates: list[str]) -> str | None:
    lower = {c.lower(): c for c in columns}
    for name in candidates:
        if name in columns:
            return name
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def score_csv(path: Path) -> tuple[int, dict[str, str | None]]:
    try:
        sample = pd.read_csv(path, nrows=200)
    except Exception:
        return 0, {}
    cols = list(sample.columns)
    mapping = {
        "user_id": first_present(cols, USER_COLS),
        "item_id": first_present(cols, ITEM_COLS),
        "timestamp": first_present(cols, TIME_COLS),
        "propensity": first_present(cols, PROPENSITY_COLS),
    }
    for out_col, aliases in REWARD_ALIASES.items():
        mapping[out_col] = first_present(cols, aliases)
    score = int(mapping["user_id"] is not None) + int(mapping["item_id"] is not None)
    score += sum(1 for out_col in REWARD_ALIASES if mapping[out_col] is not None)
    return score, mapping


def candidate_csvs(raw_dir: Path, include_standard: bool = False) -> list[tuple[Path, dict[str, str | None]]]:
    found = []
    for path in sorted(raw_dir.rglob("*.csv")):
        if not include_standard and "random" not in path.name.lower():
            continue
        score, mapping = score_csv(path)
        if mapping.get("user_id") and mapping.get("item_id") and score >= 3:
            found.append((path, mapping))
    return found


def normalize_chunk(
    chunk: pd.DataFrame,
    mapping: dict[str, str | None],
    source: Path,
    default_propensity: float | None,
) -> pd.DataFrame:
    out = pd.DataFrame()
    out["user_id"] = chunk[mapping["user_id"]].astype(str)
    out["item_id"] = chunk[mapping["item_id"]].astype(str)
    out["action_id"] = out["item_id"]
    if mapping.get("timestamp"):
        out["timestamp"] = chunk[mapping["timestamp"]]
    else:
        out["timestamp"] = pd.NA
    if mapping.get("propensity"):
        out["propensity"] = pd.to_numeric(chunk[mapping["propensity"]], errors="coerce")
        if default_propensity is not None:
            out["propensity"] = out["propensity"].fillna(default_propensity)
    elif default_propensity is not None:
        out["propensity"] = default_propensity
    else:
        out["propensity"] = pd.NA
    for out_col in REWARD_ALIASES:
        src = mapping.get(out_col)
        if src:
            out[out_col] = pd.to_numeric(chunk[src], errors="coerce")
        else:
            out[out_col] = pd.NA
    out["source_file"] = source.name
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=KUAIRAND_RAW)
    parser.add_argument("--out", type=Path, default=KUAIRAND_PROCESSED / "clicks.parquet")
    parser.add_argument("--chunksize", type=int, default=200_000)
    parser.add_argument("--max-rows", type=int, default=0, help="Optional smoke-test row cap.")
    parser.add_argument("--include-standard", action="store_true", help="Include non-random standard logs. Default is C1 random-exposure slice only.")
    parser.add_argument(
        "--uniform-propensity",
        type=float,
        default=KUAIRAND_PURE_UNIFORM_PROPENSITY,
        help="Default propensity for KuaiRand Pure random-exposed rows. Use 0 to leave missing.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cands = candidate_csvs(args.raw_dir, include_standard=args.include_standard)
    if not cands:
        raise SystemExit(f"No KuaiRand random interaction CSVs found under {args.raw_dir}")
    print("[kuairand-prepare] candidate CSVs:")
    for path, mapping in cands:
        print(f"  - {path} :: {mapping}")

    if args.dry_run:
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    total = 0
    propensity_present = False
    default_propensity = args.uniform_propensity if args.uniform_propensity > 0 else None
    for path, mapping in cands:
        usecols = sorted({v for v in mapping.values() if v})
        for chunk in pd.read_csv(path, usecols=usecols, chunksize=args.chunksize):
            norm = normalize_chunk(chunk, mapping, path, default_propensity=default_propensity)
            propensity_present = propensity_present or norm["propensity"].notna().any()
            if args.max_rows and total + len(norm) > args.max_rows:
                norm = norm.iloc[: max(args.max_rows - total, 0)]
            if not norm.empty:
                frames.append(norm)
                total += len(norm)
            if args.max_rows and total >= args.max_rows:
                break
        if args.max_rows and total >= args.max_rows:
            break

    if not frames:
        raise SystemExit("No rows produced from candidate KuaiRand CSVs")
    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(args.out, index=False)
    manifest = {
        "rows": int(len(df)),
        "n_users": int(df["user_id"].nunique()),
        "n_items": int(df["item_id"].nunique()),
        "propensity_status": "present" if propensity_present else "missing_or_unmapped",
        "propensity_source": (
            f"uniform_random_pool_1/{KUAIRAND_PURE_RANDOM_POOL_SIZE}"
            if default_propensity == KUAIRAND_PURE_UNIFORM_PROPENSITY
            else ("custom_uniform" if default_propensity is not None else "missing_or_unmapped")
        ),
        "uniform_propensity": default_propensity,
        "output": str(args.out),
        "sources": [str(p) for p, _ in cands],
        "include_standard": bool(args.include_standard),
    }
    (args.out.parent / "kuairand_prepare_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
