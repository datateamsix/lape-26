#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from lape26.corpus.acquire import download_approved_packages, ensure_lock, relock, verify_lock

DEFAULT_DOWNLOAD_DIR = ROOT / "data" / "raw" / "nltk_data"
DEFAULT_LOCK_PATH = ROOT / "data" / "manifests" / "corpus-lock.json"


def run_setup(download_dir: Path, lock_path: Path) -> None:
    download_approved_packages(download_dir)
    ok, message = ensure_lock(download_dir, lock_path)
    print(message)
    if not ok:
        raise SystemExit(1)


def run_lock(download_dir: Path, lock_path: Path) -> bool:
    ok, message = ensure_lock(download_dir, lock_path)
    print(message)
    return ok


def run_relock(download_dir: Path, lock_path: Path) -> None:
    print(relock(download_dir, lock_path))


def run_verify(download_dir: Path, lock_path: Path) -> bool:
    is_valid, message = verify_lock(download_dir, lock_path)
    print(message)
    return is_valid


def main() -> None:
    parser = argparse.ArgumentParser(description="Corpus setup/lock tooling (local only, never run in CI)")
    parser.add_argument("command", choices=["setup", "lock", "relock", "verify"])
    parser.add_argument("--download-dir", type=Path, default=DEFAULT_DOWNLOAD_DIR)
    parser.add_argument("--lock-path", type=Path, default=DEFAULT_LOCK_PATH)
    args = parser.parse_args()

    if args.command == "setup":
        run_setup(args.download_dir, args.lock_path)
    elif args.command == "lock":
        if not run_lock(args.download_dir, args.lock_path):
            raise SystemExit(1)
    elif args.command == "relock":
        run_relock(args.download_dir, args.lock_path)
    elif args.command == "verify":
        if not run_verify(args.download_dir, args.lock_path):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
