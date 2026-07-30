from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "python"))

from setup_corpus import run_lock, run_relock, run_verify  # noqa: E402
from lape26.corpus.acquire import APPROVED_PACKAGES  # noqa: E402


def _make_fake_download_dir(base: Path) -> Path:
    download_dir = base / "nltk_data"
    for package_id in APPROVED_PACKAGES:
        package_dir = download_dir / "corpora" / package_id
        package_dir.mkdir(parents=True, exist_ok=True)
        (package_dir / "sample.txt").write_text(f"fake contents for {package_id}", encoding="utf-8")
    return download_dir


class SetupCorpusCliTests(unittest.TestCase):
    def test_lock_then_verify_succeeds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            download_dir = _make_fake_download_dir(Path(tmp))
            lock_path = Path(tmp) / "corpus-lock.json"
            self.assertTrue(run_lock(download_dir, lock_path))
            self.assertTrue(lock_path.exists())
            self.assertTrue(run_verify(download_dir, lock_path))

    def test_lock_called_twice_does_not_change_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            download_dir = _make_fake_download_dir(Path(tmp))
            lock_path = Path(tmp) / "corpus-lock.json"
            run_lock(download_dir, lock_path)
            before = lock_path.read_bytes()
            self.assertTrue(run_lock(download_dir, lock_path))
            self.assertEqual(before, lock_path.read_bytes())

    def test_relock_updates_after_content_change(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            download_dir = _make_fake_download_dir(Path(tmp))
            lock_path = Path(tmp) / "corpus-lock.json"
            run_lock(download_dir, lock_path)
            (download_dir / "corpora" / "gutenberg" / "sample.txt").write_text("changed", encoding="utf-8")
            self.assertFalse(run_verify(download_dir, lock_path))
            run_relock(download_dir, lock_path)
            self.assertTrue(run_verify(download_dir, lock_path))

    def test_verify_fails_without_lock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            download_dir = _make_fake_download_dir(Path(tmp))
            self.assertFalse(run_verify(download_dir, Path(tmp) / "missing.json"))


if __name__ == "__main__":
    unittest.main()
