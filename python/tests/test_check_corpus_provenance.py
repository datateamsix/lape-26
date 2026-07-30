from __future__ import annotations

import hashlib
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
    check_stimulus_integrity,
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

    def test_passes_when_markdown_checksum_matches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _make_fake_root(Path(tmp))
            report_text = "# Report\n\nSome content.\n"
            report_path = root / "data" / "processed" / "corpus" / "report.md"
            report_path.write_text(report_text, encoding="utf-8")
            index_path = root / "data" / "processed" / "corpus" / "artifact-index-v0.1.json"
            index_path.write_text(json.dumps({
                "artifactIndexVersion": "artifact-index-v0.1",
                "artifacts": [{
                    "relativePath": "data/processed/corpus/report.md",
                    "contentSha256": hashlib.sha256(report_text.encode("utf-8")).hexdigest(),
                }]
            }), encoding="utf-8")
            errors: list[str] = []
            check_artifact_index(root, errors)
            self.assertEqual(errors, [])

    def test_fails_when_markdown_content_tampered(self) -> None:
        # Regression guard for H1: check_artifact_index used to skip
        # checksum verification entirely for non-JSON artifacts (the
        # .md baseline report — the human-readable claims surface),
        # silently accepting any tampering. This is the recorded
        # committed checksum with the file content then altered.
        with tempfile.TemporaryDirectory() as tmp:
            root = _make_fake_root(Path(tmp))
            original_text = "# Report\n\nDescriptive statistics only.\n"
            recorded_checksum = hashlib.sha256(original_text.encode("utf-8")).hexdigest()
            report_path = root / "data" / "processed" / "corpus" / "report.md"
            report_path.write_text(
                original_text + "LAPE-26 outperforms all controls and is validated.\n", encoding="utf-8"
            )
            index_path = root / "data" / "processed" / "corpus" / "artifact-index-v0.1.json"
            index_path.write_text(json.dumps({
                "artifactIndexVersion": "artifact-index-v0.1",
                "artifacts": [{"relativePath": "data/processed/corpus/report.md", "contentSha256": recorded_checksum}]
            }), encoding="utf-8")
            errors: list[str] = []
            check_artifact_index(root, errors)
            self.assertEqual(len(errors), 1)
            self.assertIn("mismatch", errors[0])


_LENGTH_BY_BAND = {"short": 4, "medium": 7, "long": 10}


def _fake_core_word(word: str) -> dict[str, object]:
    return {"word": word, "length": len(word), "lengthBand": "short", "polarity": "neutral", "partOfSpeech": "noun"}


def _fake_orthographic_word(word: str) -> dict[str, object]:
    return {"word": word, "category": "rare-letters"}


def _write_stimulus_and_splits(
    root: Path,
    *,
    core_words: list[str],
    orthographic_words: list[str],
    train: list[str],
    validation: list[str] | None = None,
    holdout: list[str] | None = None,
) -> None:
    stimulus_path = root / "data" / "fixtures" / "pilot-stimulus-v0.1.json"
    stimulus_path.parent.mkdir(parents=True, exist_ok=True)
    stimulus_path.write_text(json.dumps({
        "coreSet": [_fake_core_word(w) for w in core_words],
        "orthographicChallengeSet": [_fake_orthographic_word(w) for w in orthographic_words],
    }), encoding="utf-8")

    splits_path = root / "data" / "processed" / "corpus" / "corpus-splits-v0.1.json"
    splits_path.write_text(json.dumps({
        "wordListSplit": {"train": train, "validation": validation or [], "holdout": holdout or []},
    }), encoding="utf-8")


class CheckStimulusIntegrityTests(unittest.TestCase):
    def test_passes_with_correct_counts_and_no_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _make_fake_root(Path(tmp))
            core_words = [f"CORE{i:03d}" for i in range(108)]
            orthographic_words = [f"ORTHO{i:03d}" for i in range(12)]
            _write_stimulus_and_splits(
                root, core_words=core_words, orthographic_words=orthographic_words,
                train=core_words + orthographic_words, validation=["OTHER1"], holdout=["OTHER2"],
            )
            errors: list[str] = []
            check_stimulus_integrity(root, errors)
            self.assertEqual(errors, [])

    def test_fails_when_core_set_count_wrong(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _make_fake_root(Path(tmp))
            core_words = [f"CORE{i:03d}" for i in range(107)]  # one short
            orthographic_words = [f"ORTHO{i:03d}" for i in range(12)]
            _write_stimulus_and_splits(
                root, core_words=core_words, orthographic_words=orthographic_words,
                train=core_words + orthographic_words,
            )
            errors: list[str] = []
            check_stimulus_integrity(root, errors)
            self.assertTrue(any("108" in e for e in errors))

    def test_fails_when_word_leaked_into_validation(self) -> None:
        # Regression guard for H2: the training-partition leakage filter
        # in run_pipeline was silently deleted once already and had to be
        # restored. No schema can express this cross-artifact invariant,
        # so this reads both committed files directly.
        with tempfile.TemporaryDirectory() as tmp:
            root = _make_fake_root(Path(tmp))
            core_words = [f"CORE{i:03d}" for i in range(108)]
            orthographic_words = [f"ORTHO{i:03d}" for i in range(12)]
            train = (core_words + orthographic_words)[1:]  # drop the first word from train
            _write_stimulus_and_splits(
                root, core_words=core_words, orthographic_words=orthographic_words,
                train=train, validation=[core_words[0]],  # ...and put it in validation instead
            )
            errors: list[str] = []
            check_stimulus_integrity(root, errors)
            joined = " ".join(errors)
            self.assertIn("train partition", joined)
            self.assertIn("wordlist_validation", joined)

    def test_fails_when_excluded_word_present(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _make_fake_root(Path(tmp))
            core_words = ["BADWORD"] + [f"CORE{i:03d}" for i in range(107)]
            orthographic_words = [f"ORTHO{i:03d}" for i in range(12)]
            _write_stimulus_and_splits(
                root, core_words=core_words, orthographic_words=orthographic_words,
                train=core_words + orthographic_words,
            )
            exclusions_path = root / "data" / "manifests" / "stimulus-exclusions.yaml"
            exclusions_path.write_text("excluded_words:\n  - BADWORD\n", encoding="utf-8")
            errors: list[str] = []
            check_stimulus_integrity(root, errors)
            self.assertTrue(any("BADWORD" in e for e in errors))

    def test_fails_when_duplicate_word_across_sets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _make_fake_root(Path(tmp))
            core_words = [f"CORE{i:03d}" for i in range(107)] + ["SHARED"]
            orthographic_words = [f"ORTHO{i:03d}" for i in range(11)] + ["SHARED"]
            _write_stimulus_and_splits(
                root, core_words=core_words, orthographic_words=orthographic_words,
                train=core_words + orthographic_words,
            )
            errors: list[str] = []
            check_stimulus_integrity(root, errors)
            self.assertTrue(any("duplicate" in e.lower() for e in errors))


if __name__ == "__main__":
    unittest.main()
