from __future__ import annotations

from collections import Counter

from ..analysis import summarize
from ..core import encode_text

METRIC_VERSIONS = {
    "register_center": "register_center_v0.1",
    "pitch_span": "pitch_span_v0.1",
    "interval_contour": "interval_contour_v0.1",
    "directional_balance": "directional_balance_v0.1",
    "repetition_index": "repetition_index_v0.1",
}

REPORT_STATEMENT = (
    "baseline-comparison-v0.1 compares deterministic mappings using "
    "implemented descriptive metrics only. It does not measure objective "
    "musicality, consonance, emotional fit, or listener preference."
)


def summarize_word(word: str, mapping_path: str) -> dict[str, object]:
    events = encode_text(word, mapping_path=mapping_path)
    return summarize(events)


def _distribution(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "min": 0.0, "max": 0.0}
    return {"mean": sum(values) / len(values), "min": min(values), "max": max(values)}


def aggregate_summaries(summaries: list[dict[str, object]]) -> dict[str, object]:
    register_centers = [s["registerCenterMidi"] for s in summaries if s["registerCenterMidi"] is not None]
    pitch_spans = [s["pitchSpanSemitones"] for s in summaries]
    directional_balances = [s["directionalBalance"] for s in summaries]
    repetition_indices = [s["repetitionIndex"] for s in summaries]
    all_intervals = [interval for s in summaries for interval in s["intervals"]]

    histogram: Counter[str] = Counter(str(interval) for interval in all_intervals)

    return {
        "itemCount": len(summaries),
        "macro": {
            "registerCenterMidi": _distribution(register_centers),
            "pitchSpanSemitones": _distribution(pitch_spans),
            "directionalBalance": _distribution(directional_balances),
            "repetitionIndex": _distribution(repetition_indices),
        },
        "micro": {
            "intervalMovement": {
                "upward": sum(1 for i in all_intervals if i > 0),
                "downward": sum(1 for i in all_intervals if i < 0),
                "repeated": sum(1 for i in all_intervals if i == 0),
            },
            "intervalHistogram": dict(sorted(histogram.items(), key=lambda item: int(item[0]))),
        },
    }


def build_baseline_comparison_report(
    *,
    mapping_paths: dict[str, str],
    stimulus_words_by_stratum: dict[str, list[str]],
    provenance: dict[str, object],
) -> dict[str, object]:
    results: dict[str, object] = {}
    for mapping_id, mapping_path in mapping_paths.items():
        by_stratum: dict[str, object] = {}
        for stratum_name, words in stimulus_words_by_stratum.items():
            summaries = [summarize_word(word, mapping_path) for word in words]
            by_stratum[stratum_name] = aggregate_summaries(summaries)
        results[mapping_id] = by_stratum

    return {
        "reportId": "baseline-comparison-v0.1",
        "statement": REPORT_STATEMENT,
        "metricVersions": METRIC_VERSIONS,
        "mappingIds": list(mapping_paths),
        "pipelineVersion": "corpus-pipeline-v0.1",
        "results": results,
        "provenance": provenance,
    }
