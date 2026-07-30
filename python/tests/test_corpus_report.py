from __future__ import annotations

import json
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator

from lape26.core import DEFAULT_MAPPING_PATH
from lape26.corpus.provenance import build_provenance_block
from lape26.corpus.report import (
    METRIC_VERSIONS,
    REPORT_STATEMENT,
    aggregate_summaries,
    build_baseline_comparison_report,
    summarize_word,
)

ROOT = Path(__file__).resolve().parents[2]


class SummarizeWordTests(unittest.TestCase):
    def test_summarize_hammer_matches_golden_vector(self) -> None:
        summary = summarize_word("HAMMER", str(DEFAULT_MAPPING_PATH))
        self.assertEqual(summary["intervals"], [-9, 3, 0, 1, -7])


class AggregateSummariesTests(unittest.TestCase):
    def test_macro_and_micro_sections_present(self) -> None:
        summaries = [
            summarize_word("HAMMER", str(DEFAULT_MAPPING_PATH)),
            summarize_word("MUSIC", str(DEFAULT_MAPPING_PATH)),
        ]
        aggregate = aggregate_summaries(summaries)
        self.assertIn("macro", aggregate)
        self.assertIn("micro", aggregate)
        self.assertEqual(aggregate["itemCount"], 2)

    def test_micro_interval_histogram_sums_to_total_interval_count(self) -> None:
        summaries = [
            summarize_word("HAMMER", str(DEFAULT_MAPPING_PATH)),
            summarize_word("MUSIC", str(DEFAULT_MAPPING_PATH)),
        ]
        aggregate = aggregate_summaries(summaries)
        total_intervals = len(summaries[0]["intervals"]) + len(summaries[1]["intervals"])
        histogram_total = sum(aggregate["micro"]["intervalHistogram"].values())
        self.assertEqual(histogram_total, total_intervals)
        movement = aggregate["micro"]["intervalMovement"]
        self.assertEqual(movement["upward"] + movement["downward"] + movement["repeated"], total_intervals)

    def test_histogram_reflects_actual_signed_values(self) -> None:
        # HAMMER's intervals are [-9, 3, 0, 1, -7] (see test_core.py golden vector)
        summary = summarize_word("HAMMER", str(DEFAULT_MAPPING_PATH))
        aggregate = aggregate_summaries([summary])
        histogram = aggregate["micro"]["intervalHistogram"]
        self.assertEqual(histogram.get("-9"), 1)
        self.assertEqual(histogram.get("3"), 1)
        self.assertEqual(histogram.get("0"), 1)

    def test_empty_summaries_do_not_crash(self) -> None:
        aggregate = aggregate_summaries([])
        self.assertEqual(aggregate["itemCount"], 0)
        self.assertEqual(aggregate["macro"]["registerCenterMidi"], {"mean": 0.0, "min": 0.0, "max": 0.0})
        self.assertEqual(aggregate["micro"]["intervalHistogram"], {})


class BuildBaselineComparisonReportTests(unittest.TestCase):
    def test_report_validates_against_schema_and_states_boundary(self) -> None:
        schema = json.loads((ROOT / "data" / "schemas" / "baseline-comparison.schema.json").read_text(encoding="utf-8"))
        validator = Draft202012Validator(schema)
        provenance = build_provenance_block(
            pipeline_source_paths=["python/lape26/corpus/report.py"], input_data_paths=[],
        )
        report = build_baseline_comparison_report(
            mapping_paths={"lape-26-en-general-v0.1": str(DEFAULT_MAPPING_PATH)},
            stimulus_words_by_stratum={"short_positive": ["HAMMER", "MUSIC"]},
            provenance=provenance,
        )
        validator.validate(report)
        self.assertEqual(report["metricVersions"], METRIC_VERSIONS)
        self.assertIn("does not measure objective musicality", report["statement"])
        self.assertEqual(report["statement"], REPORT_STATEMENT)

    def test_no_ranking_language_in_statement(self) -> None:
        for banned in ("best mapping", "most musical", "highest quality"):
            self.assertNotIn(banned, REPORT_STATEMENT.lower())


if __name__ == "__main__":
    unittest.main()
