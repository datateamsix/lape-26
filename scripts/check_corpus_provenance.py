#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from jsonschema import Draft202012Validator

from lape26.corpus.provenance import (
    NORMALIZATION_IMPLEMENTATION_PATH,
    NORMALIZATION_PROFILE_ID,
    NORMALIZATION_SPEC_PATH,
    NORMALIZATION_STATUS,
    semantic_content_sha256,
    sha256_file,
)

ARTIFACT_SCHEMAS = {
    "data/processed/corpus/corpus-statistics-v0.1.json": "corpus-statistics.schema.json",
    "data/processed/corpus/corpus-splits-v0.1.json": "corpus-splits.schema.json",
    "data/processed/corpus/baseline-comparison-v0.1.json": "baseline-comparison.schema.json",
    "data/processed/corpus/artifact-index-v0.1.json": "artifact-index.schema.json",
    "data/fixtures/pilot-stimulus-v0.1.json": "pilot-stimulus.schema.json",
    "mappings/controls/sequential-chromatic-v0.1.json": "control-mapping.schema.json",
    "mappings/controls/frequency-ranked-v0.1.json": "control-mapping.schema.json",
    "mappings/controls/circle-of-fifths-v0.1.json": "control-mapping.schema.json",
    "mappings/controls/random-seed-026-v0.1.json": "control-mapping.schema.json",
}
# Artifacts that carry a `provenance` block (the artifact index deliberately does not)
PROVENANCE_BEARING_ARTIFACTS = [p for p in ARTIFACT_SCHEMAS if not p.endswith("artifact-index-v0.1.json")]


def check_schema_validation(root: Path, errors: list[str]) -> None:
    for relative_path, schema_name in ARTIFACT_SCHEMAS.items():
        artifact_path = root / relative_path
        if not artifact_path.exists():
            errors.append(f"missing artifact: {relative_path}")
            continue
        schema = json.loads((root / "data" / "schemas" / schema_name).read_text(encoding="utf-8"))
        validator = Draft202012Validator(schema)
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        for error in validator.iter_errors(payload):
            errors.append(f"{relative_path}: schema violation: {error.message}")


def check_provenance(root: Path, artifact_paths: list[Path], errors: list[str]) -> None:
    live_sha256 = sha256_file(root / NORMALIZATION_IMPLEMENTATION_PATH)
    for artifact_path in artifact_paths:
        if not artifact_path.exists():
            continue
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        provenance = payload.get("provenance")
        if provenance is None:
            continue  # e.g. artifact-index-v0.1.json, which has no provenance block
        if provenance.get("normalization_profile_id") != NORMALIZATION_PROFILE_ID:
            errors.append(f"{artifact_path}: unexpected normalization_profile_id")
            continue
        if provenance.get("normalization_status") != NORMALIZATION_STATUS:
            errors.append(f"{artifact_path}: unexpected normalization_status")
            continue
        if provenance.get("normalization_spec_path") != NORMALIZATION_SPEC_PATH:
            errors.append(f"{artifact_path}: unexpected normalization_spec_path")
            continue
        if provenance.get("normalization_implementation") != NORMALIZATION_IMPLEMENTATION_PATH:
            errors.append(f"{artifact_path}: unexpected normalization_implementation")
            continue
        if provenance.get("normalization_live_sha256") != live_sha256:
            errors.append(
                f"{artifact_path}: normalization_live_sha256 does not match "
                f"live {NORMALIZATION_IMPLEMENTATION_PATH} — regenerate this artifact"
            )
        if provenance.get("working_tree_dirty") is True:
            errors.append(f"{artifact_path}: committed with working_tree_dirty=true")


def check_corpus_lock_structure(root: Path, errors: list[str]) -> None:
    lock_path = root / "data" / "manifests" / "corpus-lock.json"
    if not lock_path.exists():
        errors.append(f"missing {lock_path}")
        return
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    packages = payload.get("packages", [])
    package_ids = {entry.get("package_id") for entry in packages}
    expected = {"gutenberg", "words", "wordnet", "opinion_lexicon", "vader_lexicon"}
    if package_ids != expected:
        errors.append(f"corpus-lock.json package_id set {package_ids} does not match expected {expected}")
    required_fields = ("package_id", "source_version", "resource_path", "archive_sha256", "installed_tree_sha256", "retrieved_at")
    for entry in packages:
        for field in required_fields:
            if field not in entry:
                errors.append(f"corpus-lock.json entry missing field {field!r}: {entry}")


def check_artifact_index(root: Path, errors: list[str]) -> None:
    index_path = root / "data" / "processed" / "corpus" / "artifact-index-v0.1.json"
    if not index_path.exists():
        errors.append(f"missing {index_path}")
        return
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    for entry in payload.get("artifacts", []):
        relative_path = entry.get("relativePath")
        if relative_path is None:
            errors.append(f"artifact-index entry missing relativePath: {entry}")
            continue
        artifact_path = root / relative_path
        if not artifact_path.exists():
            errors.append(f"artifact-index references missing file: {relative_path}")
            continue
        if artifact_path.suffix == ".json":
            actual_payload = json.loads(artifact_path.read_text(encoding="utf-8"))
            actual_checksum = semantic_content_sha256(actual_payload)
        else:
            # Non-JSON artifacts (the .md report) are hashed as plain bytes
            # upstream (build_corpus_pipeline.py's _build_artifact_index),
            # normalizing CRLF to LF first so the check is independent of
            # the checkout's line-ending configuration.
            normalized = artifact_path.read_bytes().replace(b"\r\n", b"\n")
            actual_checksum = hashlib.sha256(normalized).hexdigest()
        if entry.get("contentSha256") != actual_checksum:
            errors.append(
                f"artifact-index checksum mismatch for {relative_path}: "
                f"recorded {entry.get('contentSha256')}, actual {actual_checksum}"
            )


def check_stimulus_integrity(root: Path, errors: list[str]) -> None:
    """Guards the committed pilot-stimulus fixture against the exact
    regression class this project has already hit once: the
    training-partition leakage filter in build_corpus_pipeline.py's
    run_pipeline was silently deleted by an implementer and had to be
    restored (see branch history). No schema can express "these words
    must come from that other artifact's train partition," so this check
    reads both committed artifacts directly and verifies it by hand.
    """
    import yaml

    stimulus_path = root / "data" / "fixtures" / "pilot-stimulus-v0.1.json"
    splits_path = root / "data" / "processed" / "corpus" / "corpus-splits-v0.1.json"
    exclusions_path = root / "data" / "manifests" / "stimulus-exclusions.yaml"
    if not stimulus_path.exists() or not splits_path.exists():
        return  # already reported as missing by check_schema_validation

    stimulus = json.loads(stimulus_path.read_text(encoding="utf-8"))
    core_words = [item["word"] for item in stimulus.get("coreSet", [])]
    orthographic_words = [item["word"] for item in stimulus.get("orthographicChallengeSet", [])]
    all_words = core_words + orthographic_words

    if len(core_words) != 108:
        errors.append(f"pilot-stimulus-v0.1.json: expected 108 coreSet words, found {len(core_words)}")
    if len(orthographic_words) != 12:
        errors.append(
            f"pilot-stimulus-v0.1.json: expected 12 orthographicChallengeSet words, found {len(orthographic_words)}"
        )
    if len(set(all_words)) != len(all_words):
        seen: set[str] = set()
        duplicates = sorted({word for word in all_words if word in seen or seen.add(word)})
        errors.append(f"pilot-stimulus-v0.1.json: duplicate words across coreSet/orthographicChallengeSet: {duplicates}")

    if exclusions_path.exists():
        exclusions_data = yaml.safe_load(exclusions_path.read_text(encoding="utf-8")) or {}
        excluded_words = {str(word).upper() for word in exclusions_data.get("excluded_words", [])}
        leaked = set(all_words) & excluded_words
        if leaked:
            errors.append(f"pilot-stimulus-v0.1.json: contains manually-excluded word(s): {sorted(leaked)}")

    splits = json.loads(splits_path.read_text(encoding="utf-8"))
    word_split = splits.get("wordListSplit", {})
    train_words = set(word_split.get("train", []))
    validation_words = set(word_split.get("validation", []))
    holdout_words = set(word_split.get("holdout", []))

    stimulus_word_set = set(all_words)
    not_in_train = stimulus_word_set - train_words
    if not_in_train:
        errors.append(
            f"pilot-stimulus-v0.1.json: word(s) not in corpus-splits-v0.1.json's train partition "
            f"(leakage-prevention filter may be broken): {sorted(not_in_train)}"
        )
    leaked_into_validation = stimulus_word_set & validation_words
    if leaked_into_validation:
        errors.append(f"pilot-stimulus-v0.1.json: word(s) leaked into wordlist_validation: {sorted(leaked_into_validation)}")
    leaked_into_holdout = stimulus_word_set & holdout_words
    if leaked_into_holdout:
        errors.append(f"pilot-stimulus-v0.1.json: word(s) leaked into wordlist_holdout: {sorted(leaked_into_holdout)}")


def main() -> None:
    errors: list[str] = []
    check_schema_validation(ROOT, errors)
    check_provenance(ROOT, [ROOT / p for p in PROVENANCE_BEARING_ARTIFACTS], errors)
    check_corpus_lock_structure(ROOT, errors)
    check_artifact_index(ROOT, errors)
    check_stimulus_integrity(ROOT, errors)

    if errors:
        print("check_corpus_provenance found problems:")
        for error in errors:
            print(f"  - {error}")
        raise SystemExit(1)
    print("check_corpus_provenance: all committed corpus artifacts are valid.")


if __name__ == "__main__":
    main()
