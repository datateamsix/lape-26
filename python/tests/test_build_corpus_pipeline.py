from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "python"))

from build_corpus_pipeline import run_pipeline  # noqa: E402
from lape26.corpus.stimulus import WordCandidate  # noqa: E402

_LENGTH_BY_BAND = {"short": 4, "medium": 7, "long": 10}
_POS_CYCLE = ["noun", "verb", "adjective", "adverb"]
_BANDS = ("short", "medium", "long")
_POLARITIES = ("positive", "negative", "neutral")


def _synthetic_word_candidates(count_per_cell: int = 25) -> list[WordCandidate]:
    candidates: list[WordCandidate] = []
    for band in _BANDS:
        for polarity in _POLARITIES:
            for i in range(count_per_cell):
                word = f"{band[:2]}{polarity[:2]}{i:02d}".upper()
                candidates.append(
                    WordCandidate(
                        word=word,
                        length=_LENGTH_BY_BAND[band],
                        partOfSpeech=_POS_CYCLE[i % len(_POS_CYCLE)],
                        polarity=polarity,
                        vaderCompound=0.5 if polarity == "positive" else (-0.5 if polarity == "negative" else 0.0),
                        sourceDataset="synthetic-test-fixture",
                        stemKey=word,
                    )
                )
    return candidates


def _synthetic_orthographic_candidates() -> dict[str, list[str]]:
    # Deliberately includes words that also appear as core-set candidates'
    # *word-string prefixes* is NOT the concern here — these are entirely
    # separate strings, so no overlap with the core set is possible by
    # construction; overlap-prevention itself is exercised directly in
    # Task 12's test_corpus_stimulus.py. Here we just need enough distinct
    # candidates to satisfy 4 categories x 3 items.
    return {
        "repeated-letters": ["BALLOON", "ANNOUNCE", "COMMITTEE", "MISSISSIPPI"],
        "rare-letters": ["JAZZ", "QUIZ", "WALTZ", "XYLOPHONE"],
        "vowel-heavy": ["AI", "AREA", "IDEA", "QUEUE"],
        "consonant-heavy": ["STRENGTH", "RHYTHM", "GLYPH", "NYMPH"],
    }


def _synthetic_gutenberg_documents() -> dict[str, list[list[str]]]:
    return {
        "doc-one.txt": [["The", "cat", "sat", "."], ["A", "cat", "ran", "fast", "."]] * 20,
        "doc-two.txt": [["Dogs", "bark", "loudly", "."], ["Birds", "sing", "."]] * 5,
    }


def _run(base: Path, corpus_lock_path: Path | None = None) -> dict[str, Path]:
    lock_path = corpus_lock_path or (base / "fake-corpus-lock.json")
    if corpus_lock_path is None:
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text(json.dumps({"packages": []}), encoding="utf-8")
    return run_pipeline(
        gutenberg_sentences_by_document=_synthetic_gutenberg_documents(),
        word_candidates=_synthetic_word_candidates(),
        orthographic_candidates_by_category=_synthetic_orthographic_candidates(),
        seed=26,
        processed_dir=base / "processed",
        fixtures_dir=base / "fixtures",
        controls_dir=base / "controls",
        corpus_lock_path=lock_path,
    )


class RunPipelineTests(unittest.TestCase):
    def test_writes_all_expected_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            written = _run(Path(tmp))
            expected_ids = {
                "corpus-statistics-v0.1", "corpus-splits-v0.1",
                "sequential-chromatic-v0.1", "frequency-ranked-v0.1",
                "circle-of-fifths-v0.1", "random-seed-026-v0.1",
                "pilot-stimulus-v0.1", "baseline-comparison-v0.1-json",
                "baseline-comparison-v0.1-md", "artifact-index-v0.1",
            }
            self.assertEqual(set(written), expected_ids)
            for path in written.values():
                self.assertTrue(path.exists(), f"missing {path}")

    def test_stimulus_fixture_has_108_core_and_12_orthographic_no_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            written = _run(Path(tmp))
            stimulus = json.loads(written["pilot-stimulus-v0.1"].read_text(encoding="utf-8"))
            self.assertEqual(len(stimulus["coreSet"]), 108)
            self.assertEqual(len(stimulus["orthographicChallengeSet"]), 12)
            core_words = {item["word"] for item in stimulus["coreSet"]}
            orthographic_words = {item["word"] for item in stimulus["orthographicChallengeSet"]}
            self.assertEqual(core_words & orthographic_words, set())

    def test_deterministic_given_same_seed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            first = _run(base / "a")
            second = _run(base / "b")
            first_stimulus = json.loads(first["pilot-stimulus-v0.1"].read_text(encoding="utf-8"))
            second_stimulus = json.loads(second["pilot-stimulus-v0.1"].read_text(encoding="utf-8"))
            self.assertEqual(
                [item["word"] for item in first_stimulus["coreSet"]],
                [item["word"] for item in second_stimulus["coreSet"]],
            )

    def test_frequency_ranked_control_source_partition_is_gutenberg_train(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            written = _run(Path(tmp))
            control = json.loads(written["frequency-ranked-v0.1"].read_text(encoding="utf-8"))
            self.assertEqual(control["sourcePartition"], "gutenberg-train")
            self.assertEqual(control["normalizationProfile"], "lape-text-normalization-v0.1")

    def test_baseline_report_evaluates_holdout_and_validation_strata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            written = _run(Path(tmp))
            report = json.loads(written["baseline-comparison-v0.1-json"].read_text(encoding="utf-8"))
            canonical_strata = set(report["results"]["lape-26-en-general-v0.1"])
            for expected in ("gutenberg_validation", "gutenberg_holdout", "wordlist_validation", "wordlist_holdout"):
                self.assertIn(expected, canonical_strata)

    def test_baseline_report_declares_only_implemented_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            written = _run(Path(tmp))
            report = json.loads(written["baseline-comparison-v0.1-json"].read_text(encoding="utf-8"))
            self.assertEqual(set(report["metricVersions"]), {
                "register_center", "pitch_span", "interval_contour",
                "directional_balance", "repetition_index",
            })
            self.assertEqual(set(report["mappingIds"]), {
                "lape-26-en-general-v0.1", "sequential-chromatic-v0.1",
                "frequency-ranked-v0.1", "circle-of-fifths-v0.1", "random-seed-026-v0.1",
            })

    def test_artifact_index_uses_canonical_relative_paths_regardless_of_write_location(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            written = _run(base)  # physically written under `base`, a temp dir
            index = json.loads(written["artifact-index-v0.1"].read_text(encoding="utf-8"))
            paths = {entry["artifactId"]: entry["relativePath"] for entry in index["artifacts"]}
            # Canonical paths never mention the temp directory — this is
            # exactly what makes corpus-check's regenerated index match
            # the committed one.
            self.assertEqual(paths["corpus-statistics-v0.1"], "data/processed/corpus/corpus-statistics-v0.1.json")
            self.assertEqual(paths["pilot-stimulus-v0.1"], "data/fixtures/pilot-stimulus-v0.1.json")
            self.assertEqual(paths["frequency-ranked-v0.1"], "mappings/controls/frequency-ranked-v0.1.json")
            for path in paths.values():
                self.assertNotIn(str(base), path)

    def test_artifact_index_content_checksum_stable_across_dirty_and_clean_runs(self) -> None:
        # Two runs of the identical inputs produce different
        # pipeline_source_commit/working_tree_dirty (since real repo state
        # may have changed between calls in a live repo), but the semantic
        # content checksum in the index must still match.
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            first = _run(base / "a")
            second = _run(base / "b")
            first_index = json.loads(first["artifact-index-v0.1"].read_text(encoding="utf-8"))
            second_index = json.loads(second["artifact-index-v0.1"].read_text(encoding="utf-8"))
            first_hashes = {e["artifactId"]: e["contentSha256"] for e in first_index["artifacts"] if e["artifactId"] != "artifact-index-v0.1"}
            second_hashes = {e["artifactId"]: e["contentSha256"] for e in second_index["artifacts"] if e["artifactId"] != "artifact-index-v0.1"}
            self.assertEqual(first_hashes, second_hashes)

    def test_artifact_index_corpus_lock_sha256_matches_real_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            lock_path = base / "fake-corpus-lock.json"
            lock_path.write_text(json.dumps({"packages": ["x"]}), encoding="utf-8")
            written = _run(base, corpus_lock_path=lock_path)
            index = json.loads(written["artifact-index-v0.1"].read_text(encoding="utf-8"))
            import hashlib
            expected = hashlib.sha256(lock_path.read_bytes()).hexdigest()
            for entry in index["artifacts"]:
                self.assertEqual(entry["corpusLockSha256"], expected)


if __name__ == "__main__":
    unittest.main()
