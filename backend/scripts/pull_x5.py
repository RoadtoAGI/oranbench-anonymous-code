"""Download X5 RetailHero uplift data for OranBench C2.

The core three-table mirror follows scikit-uplift. The optional products table
is pulled from the public HuggingFace mirror because scikit-uplift does not
ship it.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from causaltwin.paths import DATA_DIR


X5_DIR = DATA_DIR / "x5"
X5_RAW = X5_DIR / "raw"

FILES = {
    "uplift_train.csv.gz": {
        "url": "https://sklift.s3.eu-west-2.amazonaws.com/uplift_train.csv.gz",
        "md5": "2720bbb659daa9e0989b2777b6a42d19",
        "required": True,
    },
    "clients.csv.gz": {
        "url": "https://sklift.s3.eu-west-2.amazonaws.com/clients.csv.gz",
        "md5": "b9cdeb2806b732771de03e819b3354c5",
        "required": True,
    },
    "purchases.csv.gz": {
        "url": "https://sklift.s3.eu-west-2.amazonaws.com/purchases.csv.gz",
        "md5": "48d2de13428e24e8b61d66fef02957a8",
        "required": True,
    },
    "products.csv.gz": {
        "url": "https://huggingface.co/datasets/pytorch-lifestream/retailhero-uplift/resolve/main/data/products.csv.gz",
        "md5": None,
        "required": False,
    },
}


def md5_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, force: bool = False) -> None:
    if dest.exists() and not force:
        print(f"[x5] reuse existing {dest}")
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    if tmp.exists():
        tmp.unlink()
    with requests.get(url, stream=True, timeout=(15, 60)) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length") or 0)
        done = 0
        with tmp.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                done += len(chunk)
                if total:
                    pct = done * 100 / total
                    print(f"\r[x5] downloading {dest.name}: {pct:5.1f}%", end="", flush=True)
        if total:
            print()
    tmp.replace(dest)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=X5_RAW)
    parser.add_argument("--include-products", action="store_true", default=True)
    parser.add_argument("--core-only", action="store_true", help="Skip optional products.csv.gz.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    args.raw_dir.mkdir(parents=True, exist_ok=True)
    selected = {
        name: spec for name, spec in FILES.items()
        if spec["required"] or (args.include_products and not args.core_only)
    }

    if args.dry_run:
        print(json.dumps(selected, indent=2))
        return 0

    results = {}
    for name, spec in selected.items():
        dest = args.raw_dir / name
        print(f"[x5] source={spec['url']}")
        try:
            download(spec["url"], dest, force=args.force)
        except Exception:
            if spec["required"]:
                raise
            print(f"[x5] optional file failed: {name}")
            results[name] = {"status": "optional_failed", "path": str(dest)}
            continue
        observed = md5_file(dest)
        if spec.get("md5") and observed != spec["md5"]:
            raise SystemExit(f"MD5 mismatch for {dest}: observed={observed} expected={spec['md5']}")
        results[name] = {"status": "ok", "path": str(dest), "md5": observed}
        print(f"[x5] ready {name} md5={observed}")

    manifest = {
        "sources": {
            "scikit_uplift_docs": "https://www.uplift-modeling.com/en/v0.4.0/api/datasets/fetch_x5.html",
            "hf_products_mirror": "https://huggingface.co/datasets/pytorch-lifestream/retailhero-uplift",
        },
        "files": results,
    }
    (args.raw_dir / "x5_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("[x5] ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

