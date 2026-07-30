#!/usr/bin/env python3
from __future__ import annotations

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
            actual_checksum = None  # non-JSON artifacts (the .md report) use a plain content hash upstream
        if actual_checksum is not None and entry.get("contentSha256") != actual_checksum:
            errors.append(
                f"artifact-index checksum mismatch for {relative_path}: "
                f"recorded {entry.get('contentSha256')}, actual {actual_checksum}"
            )


def main() -> None:
    errors: list[str] = []
    check_schema_validation(ROOT, errors)
    check_provenance(ROOT, [ROOT / p for p in PROVENANCE_BEARING_ARTIFACTS], errors)
    check_corpus_lock_structure(ROOT, errors)
    check_artifact_index(ROOT, errors)

    if errors:
        print("check_corpus_provenance found problems:")
        for error in errors:
            print(f"  - {error}")
        raise SystemExit(1)
    print("check_corpus_provenance: all committed corpus artifacts are valid.")


if __name__ == "__main__":
    main()
