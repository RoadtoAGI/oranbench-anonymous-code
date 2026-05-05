"""Download KuaiRand tarballs for OranBench.

Default target is KuaiRand-Pure because it is small enough for the NeurIPS E&D
sprint data gate while preserving the randomized-exposure benchmark lineage.

Source: https://zenodo.org/records/10439422
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tarfile
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from causaltwin.paths import KUAIRAND_RAW


FILES = {
    "pure": {
        "filename": "KuaiRand-Pure.tar.gz",
        "url": "https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz?download=1",
        "md5": "0820331067a3784d9691136f772b35a7",
        "size_hint": "47.4 MB",
    },
    "1k": {
        "filename": "KuaiRand-1K.tar.gz",
        "url": "https://zenodo.org/records/10439422/files/KuaiRand-1K.tar.gz?download=1",
        "md5": "6b0b9c8222d67fcd4c676218edca3f1f",
        "size_hint": "1.1 GB",
    },
    "27k": {
        "filename": "KuaiRand-27K.tar.gz",
        "url": "https://zenodo.org/records/10439422/files/KuaiRand-27K.tar.gz?download=1",
        "md5": "3e3c799a24e2d23a4d2c757fbf9adf59",
        "size_hint": "9.9 GB",
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
        print(f"[kuairand] reuse existing {dest}")
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
                    print(f"\r[kuairand] downloading {dest.name}: {pct:5.1f}%", end="", flush=True)
        if total:
            print()
    tmp.replace(dest)


def safe_extract_tar(tar_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    root = out_dir.resolve()
    with tarfile.open(tar_path, "r:gz") as tf:
        for member in tf.getmembers():
            target = (out_dir / member.name).resolve()
            if root not in target.parents and target != root:
                raise RuntimeError(f"unsafe tar member path: {member.name}")
        tf.extractall(out_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=sorted(FILES), default="pure")
    parser.add_argument("--raw-dir", type=Path, default=KUAIRAND_RAW)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-extract", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    spec = FILES[args.variant]
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    tar_path = args.raw_dir / spec["filename"]
    extract_dir = args.raw_dir / tar_path.stem.replace(".tar", "")

    if args.dry_run:
        print(json.dumps({"variant": args.variant, **spec, "target": str(tar_path)}, indent=2))
        return 0

    print(f"[kuairand] source={spec['url']} size={spec['size_hint']}")
    download(spec["url"], tar_path, force=args.force)
    observed = md5_file(tar_path)
    if observed != spec["md5"]:
        raise SystemExit(f"MD5 mismatch for {tar_path}: observed={observed} expected={spec['md5']}")
    print(f"[kuairand] md5 ok: {observed}")

    if not args.no_extract:
        print(f"[kuairand] extracting to {extract_dir}")
        safe_extract_tar(tar_path, extract_dir)

    manifest = {
        "source": "https://zenodo.org/records/10439422",
        "variant": args.variant,
        "tarball": str(tar_path),
        "md5": observed,
        "extracted_dir": str(extract_dir) if not args.no_extract else None,
        "license": "CC-BY-4.0",
    }
    (args.raw_dir / "kuairand_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("[kuairand] ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

