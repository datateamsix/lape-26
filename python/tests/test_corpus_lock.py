from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from lape26.corpus.acquire import (
    APPROVED_PACKAGES,
    build_lock_entries,
    ensure_lock,
    read_lock_file,
    relock,
    verify_lock,
    write_lock_file,
)


def _make_fake_download_dir(base: Path) -> Path:
    download_dir = base / "nltk_data"
    for package_id in APPROVED_PACKAGES:
        package_dir = download_dir / "corpora" / package_id
        package_dir.mkdir(parents=True, exist_ok=True)
        (package_dir / "sample.txt").write_text(f"fake contents for {package_id}", encoding="utf-8")
    return download_dir


class LockFileTests(unittest.TestCase):
    def test_build_lock_entries_covers_all_approved_packages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            download_dir = _make_fake_download_dir(Path(tmp))
            entries = build_lock_entries(download_dir, retrieved_at="2026-07-29")
            self.assertEqual({e.package_id for e in entries}, set(APPROVED_PACKAGES))
            for entry in entries:
                self.assertEqual(len(entry.installed_tree_sha256), 64)
                self.assertIsNone(entry.archive_sha256)  # directories: no separate archive retained
                self.assertEqual(entry.retrieved_at, "2026-07-29")

    def test_zip_resource_is_hashed_directly_and_counts_as_its_own_archive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            download_dir = _make_fake_download_dir(Path(tmp))
            # vader_lexicon left as an unextracted .zip instead of a directory
            import shutil
            shutil.rmtree(download_dir / "corpora" / "vader_lexicon")
            (download_dir / "sentiment").mkdir(parents=True, exist_ok=True)
            (download_dir / "sentiment" / "vader_lexicon.zip").write_bytes(b"fake zip bytes")

            entries = {e.package_id: e for e in build_lock_entries(download_dir, retrieved_at="2026-07-29")}
            vader_entry = entries["vader_lexicon"]
            self.assertEqual(vader_entry.archive_sha256, vader_entry.installed_tree_sha256)
            self.assertTrue(vader_entry.resource_path.endswith("vader_lexicon.zip"))

    def test_write_and_read_lock_file_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            download_dir = _make_fake_download_dir(Path(tmp))
            entries = build_lock_entries(download_dir, retrieved_at="2026-07-29")
            lock_path = Path(tmp) / "corpus-lock.json"
            write_lock_file(entries, lock_path)
            loaded = read_lock_file(lock_path)
            self.assertEqual(
                sorted(e.package_id for e in loaded),
                sorted(e.package_id for e in entries),
            )

    def test_verify_lock_passes_when_cache_matches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            download_dir = _make_fake_download_dir(Path(tmp))
            entries = build_lock_entries(download_dir, retrieved_at="2026-07-29")
            lock_path = Path(tmp) / "corpus-lock.json"
            write_lock_file(entries, lock_path)
            is_valid, message = verify_lock(download_dir, lock_path)
            self.assertTrue(is_valid, message)

    def test_verify_lock_fails_when_cache_content_changed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            download_dir = _make_fake_download_dir(Path(tmp))
            entries = build_lock_entries(download_dir, retrieved_at="2026-07-29")
            lock_path = Path(tmp) / "corpus-lock.json"
            write_lock_file(entries, lock_path)

            (download_dir / "corpora" / "gutenberg" / "sample.txt").write_text("changed!", encoding="utf-8")

            is_valid, message = verify_lock(download_dir, lock_path)
            self.assertFalse(is_valid)
            self.assertIn("gutenberg", message)
            self.assertIn("corpus-relock", message)

    def test_verify_lock_fails_when_lock_file_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            download_dir = _make_fake_download_dir(Path(tmp))
            is_valid, message = verify_lock(download_dir, Path(tmp) / "missing-lock.json")
            self.assertFalse(is_valid)
            self.assertIn("not found", message)

    def test_ensure_lock_creates_when_absent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            download_dir = _make_fake_download_dir(Path(tmp))
            lock_path = Path(tmp) / "corpus-lock.json"
            created, message = ensure_lock(download_dir, lock_path)
            self.assertTrue(created)
            self.assertTrue(lock_path.exists())

    def test_ensure_lock_verifies_without_overwriting_when_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            download_dir = _make_fake_download_dir(Path(tmp))
            lock_path = Path(tmp) / "corpus-lock.json"
            ensure_lock(download_dir, lock_path)
            before = lock_path.read_bytes()
            ok, message = ensure_lock(download_dir, lock_path)
            after = lock_path.read_bytes()
            self.assertTrue(ok, message)
            self.assertEqual(before, after)

    def test_ensure_lock_does_not_silently_overwrite_on_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            download_dir = _make_fake_download_dir(Path(tmp))
            lock_path = Path(tmp) / "corpus-lock.json"
            ensure_lock(download_dir, lock_path)
            before = lock_path.read_bytes()

            (download_dir / "corpora" / "gutenberg" / "sample.txt").write_text("changed!", encoding="utf-8")
            ok, message = ensure_lock(download_dir, lock_path)
            after = lock_path.read_bytes()

            self.assertFalse(ok)
            self.assertEqual(before, after)  # never silently overwritten

    def test_relock_always_overwrites_and_reports_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            download_dir = _make_fake_download_dir(Path(tmp))
            lock_path = Path(tmp) / "corpus-lock.json"
            ensure_lock(download_dir, lock_path)

            (download_dir / "corpora" / "gutenberg" / "sample.txt").write_text("changed!", encoding="utf-8")
            summary = relock(download_dir, lock_path)
            self.assertIn("gutenberg", summary)

            is_valid, message = verify_lock(download_dir, lock_path)
            self.assertTrue(is_valid, message)

    def test_relock_reports_no_changes_when_nothing_changed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            download_dir = _make_fake_download_dir(Path(tmp))
            lock_path = Path(tmp) / "corpus-lock.json"
            ensure_lock(download_dir, lock_path)
            summary = relock(download_dir, lock_path)
            self.assertEqual(summary, "No changes.")


if __name__ == "__main__":
    unittest.main()
