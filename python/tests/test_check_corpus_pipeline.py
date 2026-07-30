from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from check_corpus_pipeline import compare_artifact  # noqa: E402


class CompareArtifactTests(unittest.TestCase):
    def test_identical_json_files_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            payload = {"a": 1, "provenance": {"pipeline_source_commit": "aaa", "working_tree_dirty": False}}
            (base / "one.json").write_text(json.dumps(payload), encoding="utf-8")
            (base / "two.json").write_text(json.dumps(payload), encoding="utf-8")
            self.assertIsNone(compare_artifact(base / "one.json", base / "two.json"))

    def test_differing_commit_and_dirty_state_are_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            committed = {"a": 1, "provenance": {"pipeline_source_commit": "aaa", "working_tree_dirty": False}}
            regenerated = {"a": 1, "provenance": {"pipeline_source_commit": "bbb", "working_tree_dirty": True}}
            (base / "one.json").write_text(json.dumps(committed), encoding="utf-8")
            (base / "two.json").write_text(json.dumps(regenerated), encoding="utf-8")
            self.assertIsNone(compare_artifact(base / "one.json", base / "two.json"))

    def test_actual_content_drift_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            committed = {"a": 1, "provenance": {"pipeline_source_commit": "aaa", "working_tree_dirty": False}}
            regenerated = {"a": 2, "provenance": {"pipeline_source_commit": "aaa", "working_tree_dirty": False}}
            (base / "one.json").write_text(json.dumps(committed), encoding="utf-8")
            (base / "two.json").write_text(json.dumps(regenerated), encoding="utf-8")
            self.assertIsNotNone(compare_artifact(base / "one.json", base / "two.json"))

    def test_artifact_index_entries_compare_equal_across_temp_dir_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            index = {
                "artifactIndexVersion": "artifact-index-v0.1",
                "artifacts": [{
                    "artifactId": "corpus-statistics-v0.1",
                    "relativePath": "data/processed/corpus/corpus-statistics-v0.1.json",
                    "contentSha256": "a" * 64,
                    "corpusLockSha256": "b" * 64,
                }],
            }
            (base / "one.json").write_text(json.dumps(index), encoding="utf-8")
            (base / "two.json").write_text(json.dumps(index), encoding="utf-8")
            self.assertIsNone(compare_artifact(base / "one.json", base / "two.json"))

    def test_missing_committed_file_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "two.json").write_text("{}", encoding="utf-8")
            result = compare_artifact(base / "missing.json", base / "two.json")
            self.assertIsNotNone(result)
            self.assertIn("missing", result)

    def test_identical_markdown_files_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "one.md").write_text("# Report\n", encoding="utf-8")
            (base / "two.md").write_text("# Report\n", encoding="utf-8")
            self.assertIsNone(compare_artifact(base / "one.md", base / "two.md"))


if __name__ == "__main__":
    unittest.main()
