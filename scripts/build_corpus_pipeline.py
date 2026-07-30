#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

from lape26.core import DEFAULT_MAPPING_PATH
from lape26.corpus.acquire import verify_lock
from lape26.corpus.controls import (
    POOL_MIDI,
    build_control_mapping_document,
    generate_circle_of_fifths_control,
    generate_frequency_ranked_control,
    generate_random_seed_control,
    generate_sequential_chromatic_control,
)
from lape26.corpus.provenance import REPO_ROOT, build_provenance_block, sha256_file, semantic_content_sha256
from lape26.corpus.report import build_baseline_comparison_report
from lape26.corpus.splits import split_documents_by_word_count, split_word_list
from lape26.corpus.stats import compute_statistics
from lape26.corpus.stimulus import (
    WordCandidate,
    length_band,
    select_core_set,
    select_orthographic_challenge_set,
)
from lape26.corpus.tokens import tokenize_sentences

PIPELINE_SOURCE_PATHS = [
    "python/lape26/normalize.py",
    "python/lape26/core.py",
    "python/lape26/analysis.py",
    "python/lape26/corpus/tokens.py",
    "python/lape26/corpus/stats.py",
    "python/lape26/corpus/splits.py",
    "python/lape26/corpus/controls.py",
    "python/lape26/corpus/stimulus.py",
    "python/lape26/corpus/report.py",
    "python/lape26/corpus/provenance.py",
    "python/lape26/corpus/acquire.py",
    "python/lape26/corpus/nltk_adapter.py",
    "scripts/build_corpus_pipeline.py",
]
INPUT_DATA_PATHS = [
    "data/manifests/corpus-lock.json",
    "data/manifests/stimulus-exclusions.yaml",
    "data/manifests/gutenberg.yaml",
    "data/manifests/words.yaml",
    "data/manifests/wordnet.yaml",
    "data/manifests/opinion-lexicon.yaml",
    "data/manifests/vader.yaml",
]

_CANONICAL_RELATIVE_PATHS = {
    "corpus-statistics-v0.1": "data/processed/corpus/corpus-statistics-v0.1.json",
    "corpus-splits-v0.1": "data/processed/corpus/corpus-splits-v0.1.json",
    "sequential-chromatic-v0.1": "mappings/controls/sequential-chromatic-v0.1.json",
    "frequency-ranked-v0.1": "mappings/controls/frequency-ranked-v0.1.json",
    "circle-of-fifths-v0.1": "mappings/controls/circle-of-fifths-v0.1.json",
    "random-seed-026-v0.1": "mappings/controls/random-seed-026-v0.1.json",
    "pilot-stimulus-v0.1": "data/fixtures/pilot-stimulus-v0.1.json",
    "baseline-comparison-v0.1-json": "data/processed/corpus/baseline-comparison-v0.1.json",
    "baseline-comparison-v0.1-md": "data/processed/corpus/baseline-comparison-v0.1.md",
}

STRATUM_SAMPLE_LIMIT = 200


def _deterministic_sample(items: list[str], seed: int, limit: int) -> list[str]:
    pool = sorted(set(items))
    if len(pool) <= limit:
        return pool
    rng = random.Random(seed)
    shuffled = pool[:]
    rng.shuffle(shuffled)
    return sorted(shuffled[:limit])


def run_pipeline(
    *,
    gutenberg_sentences_by_document: dict[str, list[list[str]]],
    word_candidates: list[WordCandidate],
    orthographic_candidates_by_category: dict[str, list[str]],
    seed: int,
    processed_dir: Path,
    fixtures_dir: Path,
    controls_dir: Path,
    corpus_lock_path: Path,
) -> dict[str, Path]:
    provenance = build_provenance_block(
        pipeline_source_paths=sorted(PIPELINE_SOURCE_PATHS),
        input_data_paths=sorted(INPUT_DATA_PATHS),
    )
    corpus_lock_sha256 = (
        sha256_file(corpus_lock_path) if corpus_lock_path.exists() else hashlib.sha256(b"MISSING").hexdigest()
    )

    tokenized_by_document = {
        document_id: tokenize_sentences(document_id, raw_sentences)
        for document_id, raw_sentences in gutenberg_sentences_by_document.items()
    }
    document_word_counts = {
        document_id: sum(len(s.words) for s in sentences)
        for document_id, sentences in tokenized_by_document.items()
    }
    gutenberg_split = split_documents_by_word_count(document_word_counts, seed=seed)

    word_list = sorted({candidate.word.upper() for candidate in word_candidates})
    word_split = split_word_list(word_list, seed=seed)

    train_sentences = [
        sentence for document_id in gutenberg_split.train for sentence in tokenized_by_document[document_id]
    ]
    all_sentences = [sentence for sentences in tokenized_by_document.values() for sentence in sentences]

    corpus_statistics = compute_statistics(all_sentences)
    train_statistics = compute_statistics(train_sentences)

    processed_dir.mkdir(parents=True, exist_ok=True)
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    controls_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    payloads_for_index: dict[str, tuple[dict[str, object], int | None]] = {}

    statistics_payload = {
        "artifactId": "corpus-statistics-v0.1",
        "artifactVersion": "0.1.0",
        **asdict(corpus_statistics),
        "provenance": provenance,
    }
    statistics_path = processed_dir / "corpus-statistics-v0.1.json"
    _write_json(statistics_path, statistics_payload)
    written["corpus-statistics-v0.1"] = statistics_path
    payloads_for_index["corpus-statistics-v0.1"] = (statistics_payload, None)

    splits_payload = {
        "artifactId": "corpus-splits-v0.1",
        "artifactVersion": "0.1.0",
        "wordListSplit": asdict(word_split),
        "gutenbergSplit": asdict(gutenberg_split),
        "provenance": provenance,
    }
    splits_path = processed_dir / "corpus-splits-v0.1.json"
    _write_json(splits_path, splits_payload)
    written["corpus-splits-v0.1"] = splits_path
    payloads_for_index["corpus-splits-v0.1"] = (splits_payload, seed)

    control_specs = {
        "sequential-chromatic-v0.1": (
            "sequential-chromatic", "alphabetical-ascending-chromatic",
            generate_sequential_chromatic_control(POOL_MIDI), None, None,
        ),
        "frequency-ranked-v0.1": (
            "frequency-ranked", "gutenberg-train-character-frequency-nearest-register-center",
            generate_frequency_ranked_control(train_statistics.characterFrequency, POOL_MIDI),
            None, "gutenberg-train",
        ),
        "circle-of-fifths-v0.1": (
            "circle-of-fifths", "fifths-cycle-nearest-available-fallback",
            generate_circle_of_fifths_control(POOL_MIDI), None, None,
        ),
        "random-seed-026-v0.1": (
            "random-seed", "seeded-uniform-permutation",
            generate_random_seed_control(POOL_MIDI, seed), seed, None,
        ),
    }
    control_paths: dict[str, Path] = {}
    for control_id, (control_type, method, assignment, control_seed, source_partition) in control_specs.items():
        document = build_control_mapping_document(
            control_id=control_id,
            control_type=control_type,
            generation_method=method,
            assignment=assignment,
            seed=control_seed,
            source_partition=source_partition,
            provenance=provenance,
        )
        path = controls_dir / f"{control_id}.json"
        _write_json(path, document)
        control_paths[control_id] = path
        written[control_id] = path
        payloads_for_index[control_id] = (document, control_seed)

    used_words: set[str] = set()
    used_stems: set[str] = set()
    core_set = select_core_set(word_candidates, seed=seed, used_words=used_words, used_stems=used_stems)
    orthographic_set = select_orthographic_challenge_set(
        orthographic_candidates_by_category, seed=seed, used_words=used_words,
    )

    core_set_items = [
        {
            "word": c.word,
            "length": c.length,
            "lengthBand": length_band(c.length),
            "polarity": c.polarity,
            "vaderCompound": c.vaderCompound,
            "partOfSpeech": c.partOfSpeech,
            "sourceDataset": c.sourceDataset,
        }
        for c in core_set
    ]
    stimulus_payload = {
        "artifactId": "pilot-stimulus-v0.1",
        "artifactVersion": "0.1.0",
        "seed": seed,
        "coreSet": core_set_items,
        "orthographicChallengeSet": [asdict(o) for o in orthographic_set],
        "interpretationBoundary": (
            "Sentiment labels are sampling metadata only. Do not assume positive "
            "words sound consonant, major, pleasant, or high-pitched, or that "
            "negative words sound dissonant, minor, or low-pitched."
        ),
        "representativenessBoundary": (
            "This is a controlled evaluation set, not a natural-language "
            "frequency sample. It must not be described as representative of "
            "English without reliable usage-frequency balancing."
        ),
        "provenance": provenance,
    }
    stimulus_path = fixtures_dir / "pilot-stimulus-v0.1.json"
    _write_json(stimulus_path, stimulus_payload)
    written["pilot-stimulus-v0.1"] = stimulus_path
    payloads_for_index["pilot-stimulus-v0.1"] = (stimulus_payload, seed)

    strata: dict[str, list[str]] = {}
    for item in core_set_items:
        strata.setdefault(f"{item['lengthBand']}_{item['polarity']}", []).append(item["word"])
    strata["orthographic_challenge"] = [o.word for o in orthographic_set]

    def _words_in_documents(document_ids: tuple[str, ...]) -> list[str]:
        return sorted({
            word.normalizedText
            for document_id in document_ids
            for sentence in tokenized_by_document[document_id]
            for word in sentence.words
        })

    strata["gutenberg_validation"] = _deterministic_sample(
        _words_in_documents(gutenberg_split.validation), seed, STRATUM_SAMPLE_LIMIT,
    )
    strata["gutenberg_holdout"] = _deterministic_sample(
        _words_in_documents(gutenberg_split.holdout), seed, STRATUM_SAMPLE_LIMIT,
    )
    strata["wordlist_validation"] = _deterministic_sample(list(word_split.validation), seed, STRATUM_SAMPLE_LIMIT)
    strata["wordlist_holdout"] = _deterministic_sample(list(word_split.holdout), seed, STRATUM_SAMPLE_LIMIT)

    mapping_paths = {"lape-26-en-general-v0.1": str(DEFAULT_MAPPING_PATH)}
    mapping_paths.update({control_id: str(path) for control_id, path in control_paths.items()})

    report_payload = build_baseline_comparison_report(
        mapping_paths=mapping_paths,
        stimulus_words_by_stratum=strata,
        provenance=provenance,
    )
    report_json_path = processed_dir / "baseline-comparison-v0.1.json"
    _write_json(report_json_path, report_payload)
    written["baseline-comparison-v0.1-json"] = report_json_path
    payloads_for_index["baseline-comparison-v0.1-json"] = (report_payload, seed)

    report_markdown = _render_report_markdown(report_payload)
    report_md_path = processed_dir / "baseline-comparison-v0.1.md"
    report_md_path.write_text(report_markdown, encoding="utf-8")
    written["baseline-comparison-v0.1-md"] = report_md_path

    index_payload = _build_artifact_index(payloads_for_index, report_markdown, provenance, corpus_lock_sha256)
    index_path = processed_dir / "artifact-index-v0.1.json"
    _write_json(index_path, index_payload)
    written["artifact-index-v0.1"] = index_path

    return written


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")


def _render_report_markdown(report: dict[str, object]) -> str:
    lines = [
        f"# {report['reportId']}",
        "",
        report["statement"],
        "",
        f"Pipeline version: `{report['pipelineVersion']}`",
        "",
        "Strata include the pilot fixture (core cells + orthographic challenge) "
        "as well as gutenberg_validation, gutenberg_holdout, wordlist_validation, "
        "and wordlist_holdout — the frequency-ranked control is fit on the "
        "Gutenberg train partition only and evaluated here against material it "
        "never saw.",
        "",
        "## Metric versions",
        "",
    ]
    for name, version in report["metricVersions"].items():
        lines.append(f"- `{name}`: `{version}`")
    lines.append("")
    lines.append("## Results by mapping and stratum (macro = per-item average, micro = pooled)")
    for mapping_id, by_stratum in report["results"].items():
        lines.append("")
        lines.append(f"### {mapping_id}")
        lines.append("")
        lines.append("| Stratum | Items | Register mean (macro) | Pitch span mean (macro) | Directional balance mean (macro) | Repetition mean (macro) | Up/Down/Repeat (micro) |")
        lines.append("|---|---:|---:|---:|---:|---:|---|")
        for stratum_name, aggregate in by_stratum.items():
            macro = aggregate["macro"]
            movement = aggregate["micro"]["intervalMovement"]
            lines.append(
                f"| {stratum_name} | {aggregate['itemCount']} "
                f"| {macro['registerCenterMidi']['mean']:.2f} "
                f"| {macro['pitchSpanSemitones']['mean']:.2f} "
                f"| {macro['directionalBalance']['mean']:.3f} "
                f"| {macro['repetitionIndex']['mean']:.3f} "
                f"| {movement['upward']}/{movement['downward']}/{movement['repeated']} |"
            )
    return "\n".join(lines) + "\n"


def _build_artifact_index(
    payloads_for_index: dict[str, tuple[dict[str, object], int | None]],
    report_markdown: str,
    provenance: dict[str, object],
    corpus_lock_sha256: str,
) -> dict[str, object]:
    artifacts = []
    for artifact_id, (payload, seed) in sorted(payloads_for_index.items()):
        artifacts.append({
            "artifactId": artifact_id,
            "artifactVersion": "0.1.0",
            "relativePath": _CANONICAL_RELATIVE_PATHS[artifact_id],
            "contentSha256": semantic_content_sha256(payload),
            "pipelineVersion": "corpus-pipeline-v0.1",
            "normalizationProfile": provenance["normalization_profile_id"],
            "seed": seed,
            "corpusLockSha256": corpus_lock_sha256,
        })
    artifacts.append({
        "artifactId": "baseline-comparison-v0.1-md",
        "artifactVersion": "0.1.0",
        "relativePath": _CANONICAL_RELATIVE_PATHS["baseline-comparison-v0.1-md"],
        "contentSha256": hashlib.sha256(report_markdown.encode("utf-8")).hexdigest(),
        "pipelineVersion": "corpus-pipeline-v0.1",
        "normalizationProfile": provenance["normalization_profile_id"],
        "seed": None,
        "corpusLockSha256": corpus_lock_sha256,
    })
    artifacts.sort(key=lambda entry: entry["artifactId"])
    return {"artifactIndexVersion": "artifact-index-v0.1", "artifacts": artifacts}


def main() -> None:
    from setup_corpus import DEFAULT_DOWNLOAD_DIR, DEFAULT_LOCK_PATH

    from lape26.corpus.nltk_adapter import (
        load_gutenberg_sentences_by_document,
        load_orthographic_candidates,
        load_word_candidates,
    )

    is_valid, message = verify_lock(DEFAULT_DOWNLOAD_DIR, DEFAULT_LOCK_PATH)
    print(message)
    if not is_valid:
        raise SystemExit(1)

    exclusions_path = REPO_ROOT / "data" / "manifests" / "stimulus-exclusions.yaml"
    written = run_pipeline(
        gutenberg_sentences_by_document=load_gutenberg_sentences_by_document(DEFAULT_DOWNLOAD_DIR),
        word_candidates=load_word_candidates(DEFAULT_DOWNLOAD_DIR, exclusions_path),
        orthographic_candidates_by_category=load_orthographic_candidates(DEFAULT_DOWNLOAD_DIR, exclusions_path),
        seed=26,
        processed_dir=REPO_ROOT / "data" / "processed" / "corpus",
        fixtures_dir=REPO_ROOT / "data" / "fixtures",
        controls_dir=REPO_ROOT / "mappings" / "controls",
        corpus_lock_path=DEFAULT_LOCK_PATH,
    )
    for artifact_id, path in sorted(written.items()):
        print(f"Wrote {artifact_id}: {path}")


if __name__ == "__main__":
    main()
