#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "scripts"))

_IGNORED_PROVENANCE_FIELDS = ("pipeline_source_commit", "working_tree_dirty")

COMMITTED_PROCESSED_DIR = ROOT / "data" / "processed" / "corpus"
COMMITTED_FIXTURES_DIR = ROOT / "data" / "fixtures"
COMMITTED_CONTROLS_DIR = ROOT / "mappings" / "controls"
COMMITTED_LOCK_PATH = ROOT / "data" / "manifests" / "corpus-lock.json"


def _normalize_for_comparison(payload: dict[str, object]) -> dict[str, object]:
    normalized = copy.deepcopy(payload)
    provenance = normalized.get("provenance")
    if isinstance(provenance, dict):
        for field in _IGNORED_PROVENANCE_FIELDS:
            provenance.pop(field, None)
    return normalized


def compare_artifact(committed_path: Path, regenerated_path: Path) -> str | None:
    if not committed_path.exists():
        return f"committed file missing at {committed_path}"
    if not regenerated_path.exists():
        return f"regenerated file missing at {regenerated_path}"

    if committed_path.suffix == ".json":
        committed = _normalize_for_comparison(json.loads(committed_path.read_text(encoding="utf-8")))
        regenerated = _normalize_for_comparison(json.loads(regenerated_path.read_text(encoding="utf-8")))
        if committed != regenerated:
            return f"content differs from committed {committed_path}"
        return None

    if committed_path.read_text(encoding="utf-8") != regenerated_path.read_text(encoding="utf-8"):
        return f"content differs from committed {committed_path}"
    return None


def check(tmp_root: Path) -> list[str]:
    from build_corpus_pipeline import run_pipeline
    from lape26.corpus.acquire import verify_lock
    from lape26.corpus.nltk_adapter import (
        load_gutenberg_sentences_by_document,
        load_orthographic_candidates,
        load_word_candidates,
    )
    from setup_corpus import DEFAULT_DOWNLOAD_DIR, DEFAULT_LOCK_PATH

    is_valid, message = verify_lock(DEFAULT_DOWNLOAD_DIR, DEFAULT_LOCK_PATH)
    if not is_valid:
        raise SystemExit(f"corpus-check aborted: {message}")

    exclusions_path = ROOT / "data" / "manifests" / "stimulus-exclusions.yaml"
    regenerated = run_pipeline(
        gutenberg_sentences_by_document=load_gutenberg_sentences_by_document(DEFAULT_DOWNLOAD_DIR),
        word_candidates=load_word_candidates(DEFAULT_DOWNLOAD_DIR, exclusions_path),
        orthographic_candidates_by_category=load_orthographic_candidates(DEFAULT_DOWNLOAD_DIR, exclusions_path),
        seed=26,
        processed_dir=tmp_root / "processed",
        fixtures_dir=tmp_root / "fixtures",
        controls_dir=tmp_root / "controls",
        corpus_lock_path=DEFAULT_LOCK_PATH,
    )

    committed_paths = {
        "corpus-statistics-v0.1": COMMITTED_PROCESSED_DIR / "corpus-statistics-v0.1.json",
        "corpus-splits-v0.1": COMMITTED_PROCESSED_DIR / "corpus-splits-v0.1.json",
        "sequential-chromatic-v0.1": COMMITTED_CONTROLS_DIR / "sequential-chromatic-v0.1.json",
        "frequency-ranked-v0.1": COMMITTED_CONTROLS_DIR / "frequency-ranked-v0.1.json",
        "circle-of-fifths-v0.1": COMMITTED_CONTROLS_DIR / "circle-of-fifths-v0.1.json",
        "random-seed-026-v0.1": COMMITTED_CONTROLS_DIR / "random-seed-026-v0.1.json",
        "pilot-stimulus-v0.1": COMMITTED_FIXTURES_DIR / "pilot-stimulus-v0.1.json",
        "baseline-comparison-v0.1-json": COMMITTED_PROCESSED_DIR / "baseline-comparison-v0.1.json",
        "baseline-comparison-v0.1-md": COMMITTED_PROCESSED_DIR / "baseline-comparison-v0.1.md",
        "artifact-index-v0.1": COMMITTED_PROCESSED_DIR / "artifact-index-v0.1.json",
    }

    mismatches: list[str] = []
    for artifact_id, committed_path in committed_paths.items():
        mismatch = compare_artifact(committed_path, regenerated[artifact_id])
        if mismatch:
            mismatches.append(f"{artifact_id}: {mismatch}")
    return mismatches


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="lape26-corpus-check-") as tmp:
        mismatches = check(Path(tmp))
    if mismatches:
        print("corpus-check found differences:")
        for mismatch in mismatches:
            print(f"  - {mismatch}")
        raise SystemExit(1)
    print("corpus-check: regenerated artifacts match committed artifacts.")


if __name__ == "__main__":
    main()
