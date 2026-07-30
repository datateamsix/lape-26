from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "python"))

from check_corpus_provenance import (  # noqa: E402
    check_artifact_index,
    check_corpus_lock_structure,
    check_provenance,
)
from lape26.corpus.provenance import (  # noqa: E402
    NORMALIZATION_IMPLEMENTATION_PATH,
    NORMALIZATION_PROFILE_ID,
    NORMALIZATION_SPEC_PATH,
    NORMALIZATION_STATUS,
    semantic_content_sha256,
    sha256_file,
)


def _make_fake_root(base: Path) -> Path:
    (base / "python" / "lape26").mkdir(parents=True)
    (base / "python" / "lape26" / "normalize.py").write_text("def normalize_text(t): return t\n", encoding="utf-8")
    (base / "data" / "manifests").mkdir(parents=True)
    (base / "data" / "processed" / "corpus").mkdir(parents=True)
    return base


def _provenance_block(live_sha: str, dirty: bool = False) -> dict[str, object]:
    return {
        "normalization_profile_id": NORMALIZATION_PROFILE_ID,
        "normalization_status": NORMALIZATION_STATUS,
        "normalization_spec_path": NORMALIZATION_SPEC_PATH,
        "normalization_implementation": NORMALIZATION_IMPLEMENTATION_PATH,
        "normalization_live_sha256": live_sha,
        "normalization_committed_blob_sha": "a" * 40,
        "pipeline_source_commit": "b" * 40,
        "working_tree_dirty": dirty,
        "source_tree_digest": "c" * 64,
        "input_data_sha256": "d" * 64,
    }


class CheckProvenanceTests(unittest.TestCase):
    def test_passes_when_provenance_matches_live_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _make_fake_root(Path(tmp))
            live_sha = sha256_file(root / NORMALIZATION_IMPLEMENTATION_PATH)
            artifact_path = root / "data" / "processed" / "corpus" / "artifact.json"
            artifact_path.write_text(json.dumps({"provenance": _provenance_block(live_sha)}), encoding="utf-8")
            errors: list[str] = []
            check_provenance(root, [artifact_path], errors)
            self.assertEqual(errors, [])

    def test_fails_when_checksum_stale(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _make_fake_root(Path(tmp))
            artifact_path = root / "data" / "processed" / "corpus" / "artifact.json"
            artifact_path.write_text(json.dumps({"provenance": _provenance_block("stale-checksum")}), encoding="utf-8")
            errors: list[str] = []
            check_provenance(root, [artifact_path], errors)
            self.assertEqual(len(errors), 1)
            self.assertIn("does not match", errors[0])

    def test_fails_when_working_tree_dirty_true(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _make_fake_root(Path(tmp))
            live_sha = sha256_file(root / NORMALIZATION_IMPLEMENTATION_PATH)
            artifact_path = root / "data" / "processed" / "corpus" / "artifact.json"
            artifact_path.write_text(json.dumps({"provenance": _provenance_block(live_sha, dirty=True)}), encoding="utf-8")
            errors: list[str] = []
            check_provenance(root, [artifact_path], errors)
            self.assertEqual(len(errors), 1)
            self.assertIn("working_tree_dirty", errors[0])


class CheckCorpusLockStructureTests(unittest.TestCase):
    def test_passes_with_all_five_packages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _make_fake_root(Path(tmp))
            lock_path = root / "data" / "manifests" / "corpus-lock.json"
            lock_path.write_text(json.dumps({
                "packages": [
                    {
                        "package_id": pkg, "source_version": "x", "resource_path": "x",
                        "archive_sha256": None, "installed_tree_sha256": "a" * 64,
                        "retrieved_at": "2026-07-29",
                    }
                    for pkg in ("gutenberg", "words", "wordnet", "opinion_lexicon", "vader_lexicon")
                ]
            }), encoding="utf-8")
            errors: list[str] = []
            check_corpus_lock_structure(root, errors)
            self.assertEqual(errors, [])

    def test_fails_when_package_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _make_fake_root(Path(tmp))
            lock_path = root / "data" / "manifests" / "corpus-lock.json"
            lock_path.write_text(json.dumps({
                "packages": [{
                    "package_id": "gutenberg", "source_version": "x", "resource_path": "x",
                    "archive_sha256": None, "installed_tree_sha256": "a" * 64,
                    "retrieved_at": "2026-07-29",
                }]
            }), encoding="utf-8")
            errors: list[str] = []
            check_corpus_lock_structure(root, errors)
            self.assertEqual(len(errors), 1)


class CheckArtifactIndexTests(unittest.TestCase):
    def test_passes_when_semantic_checksums_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _make_fake_root(Path(tmp))
            artifact_payload = {"x": 1, "provenance": {"pipeline_source_commit": "aaa", "working_tree_dirty": False}}
            artifact_path = root / "data" / "processed" / "corpus" / "thing.json"
            artifact_path.write_text(json.dumps(artifact_payload), encoding="utf-8")
            index_path = root / "data" / "processed" / "corpus" / "artifact-index-v0.1.json"
            index_path.write_text(json.dumps({
                "artifactIndexVersion": "artifact-index-v0.1",
                "artifacts": [{
                    "relativePath": "data/processed/corpus/thing.json",
                    "contentSha256": semantic_content_sha256(artifact_payload),
                }]
            }), encoding="utf-8")
            errors: list[str] = []
            check_artifact_index(root, errors)
            self.assertEqual(errors, [])

    def test_fails_when_semantic_checksum_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _make_fake_root(Path(tmp))
            artifact_path = root / "data" / "processed" / "corpus" / "thing.json"
            artifact_path.write_text(json.dumps({"x": 1}), encoding="utf-8")
            index_path = root / "data" / "processed" / "corpus" / "artifact-index-v0.1.json"
            index_path.write_text(json.dumps({
                "artifactIndexVersion": "artifact-index-v0.1",
                "artifacts": [{"relativePath": "data/processed/corpus/thing.json", "contentSha256": "wrong"}]
            }), encoding="utf-8")
            errors: list[str] = []
            check_artifact_index(root, errors)
            self.assertEqual(len(errors), 1)
            self.assertIn("mismatch", errors[0])


if __name__ == "__main__":
    unittest.main()
