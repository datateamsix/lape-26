# Corpus Evaluation Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a reproducible, license-aware corpus and stimulus pipeline (dataset manifests, normalization-provenance tracking, corpus statistics, leakage-safe deterministic splits, a 120-item pilot stimulus fixture, four control mappings, and a descriptive-only baseline comparison report) without modifying the canonical mapping.

**Architecture:** A new `python/lape26/corpus/` subpackage (tokens, stats, splits, provenance, controls, acquire, stimulus, report — each pure/testable, NLTK imported only inside `acquire.py`) is orchestrated by thin CLI scripts (`scripts/setup_corpus.py`, `scripts/build_corpus_pipeline.py`, `scripts/check_corpus_pipeline.py`, `scripts/check_corpus_provenance.py`). `normalize_text` is extracted from `core.py` into its own `normalize.py` so it has an independent, checksummable identity for provenance tracking.

**Tech Stack:** Python 3.11+, NLTK (research-only dependency), PyYAML, stdlib `unittest` (matches existing `python/tests` convention), Node.js/TypeScript untouched (this phase is Python-only).

## Global Constraints

- `mappings/lape-26-en-general-v0.1.json` must remain byte-identical throughout — verified by a test.
- `specs/text-normalization.md` is not modified or "finalized" in this workstream.
- CI (`ci.yml`) must never call `nltk.download()` or otherwise touch the network. All new CI steps run against `python/tests/fixtures/tiny-corpus-sample.txt` and small synthetic data.
- Excluded from the pipeline entirely: Brown corpus, Names corpus, CMUdict. Approved packages are exactly: `gutenberg`, `words`, `wordnet`, `opinion_lexicon`, `vader_lexicon`.
- `frequency-ranked` control must be built from the Gutenberg **train** partition only.
- The 120-item pilot stimulus fixture is not a member of the train/validation/holdout split — it's a separate, frozen, independently versioned artifact.
- No unimplemented-metric language (`tonal_fit`, etc.) or ranking language ("best mapping", "most musical") anywhere in generated reports.
- `ROADMAP.md` phase numbers are not renumbered; only Phase 2's two subsections and one Phase 0 checkbox change.
- NLTK is imported only inside `python/lape26/corpus/acquire.py`, never at module scope elsewhere in `corpus/`.
- Full spec: `docs/superpowers/specs/2026-07-29-phase-1-corpus-evaluation-foundation-design.md`.

---

## Task 1: JSON schemas for generated artifacts

**Files:**
- Create: `data/schemas/dataset-manifest.schema.json`
- Create: `data/schemas/corpus-provenance.schema.json`
- Create: `data/schemas/corpus-statistics.schema.json`
- Create: `data/schemas/corpus-splits.schema.json`
- Create: `data/schemas/pilot-stimulus.schema.json`
- Create: `data/schemas/baseline-comparison.schema.json`
- Create: `data/schemas/control-mapping.schema.json`
- Test: `python/tests/test_schemas_valid.py`

**Interfaces:**
- Produces: 7 schema files under `data/schemas/`, each a valid JSON Schema (draft 2020-12) document, consumed by `scripts/check_corpus_provenance.py` (Task 16) and by later generator tasks.

- [ ] **Step 1: Write the failing test**

Create `python/tests/test_schemas_valid.py`:

```python
from __future__ import annotations

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCHEMAS_DIR = ROOT / "data" / "schemas"

EXPECTED_SCHEMAS = [
    "dataset-manifest.schema.json",
    "corpus-provenance.schema.json",
    "corpus-statistics.schema.json",
    "corpus-splits.schema.json",
    "pilot-stimulus.schema.json",
    "baseline-comparison.schema.json",
    "control-mapping.schema.json",
]


class SchemaFilesTests(unittest.TestCase):
    def test_all_expected_schemas_exist_and_parse(self) -> None:
        for filename in EXPECTED_SCHEMAS:
            path = SCHEMAS_DIR / filename
            with self.subTest(filename=filename):
                self.assertTrue(path.exists(), f"missing {path}")
                data = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(data["$schema"], "https://json-schema.org/draft/2020-12/schema")
                self.assertIn("required", data)
                self.assertIn("properties", data)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=python python3 -m unittest python.tests.test_schemas_valid -v`
Expected: FAIL — `data/schemas` does not exist.

- [ ] **Step 3: Create the 7 schema files**

Create `data/schemas/dataset-manifest.schema.json`:

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://lape26.org/schemas/dataset-manifest.schema.json",
  "title": "LAPE-26 Dataset Manifest",
  "type": "object",
  "required": [
    "dataset_id", "name", "version", "source_url", "retrieved_at",
    "license", "redistribution_allowed", "content_type", "language",
    "locale", "checksum_sha256", "role", "preparation_steps", "known_biases"
  ],
  "properties": {
    "dataset_id": { "type": "string", "minLength": 1 },
    "name": { "type": "string", "minLength": 1 },
    "version": { "type": "string" },
    "source_url": { "type": "string" },
    "retrieved_at": { "type": "string", "format": "date" },
    "license": { "type": "string" },
    "redistribution_allowed": { "type": "boolean" },
    "redistribution_notes": { "type": "string" },
    "content_type": { "type": "string" },
    "language": { "type": "string" },
    "locale": { "type": "string" },
    "checksum_sha256": { "type": "string" },
    "role": { "type": "string", "minLength": 1 },
    "must_not_be_used_for": { "type": "array", "items": { "type": "string" } },
    "preparation_steps": { "type": "array", "items": { "type": "string" }, "minItems": 1 },
    "known_biases": { "type": "array", "items": { "type": "string" } }
  },
  "additionalProperties": false
}
```

Create `data/schemas/corpus-provenance.schema.json`:

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://lape26.org/schemas/corpus-provenance.schema.json",
  "title": "LAPE-26 Corpus Provenance Block",
  "type": "object",
  "required": [
    "normalization_profile_id", "normalization_status", "normalization_spec_path",
    "normalization_implementation", "normalization_implementation_sha256",
    "normalization_git_blob_sha", "pipeline_source_commit", "working_tree_dirty",
    "source_tree_digest"
  ],
  "properties": {
    "normalization_profile_id": { "const": "lape-text-normalization-v0.1" },
    "normalization_status": { "const": "provisional-frozen" },
    "normalization_spec_path": { "const": "specs/text-normalization.md" },
    "normalization_implementation": { "const": "python/lape26/normalize.py" },
    "normalization_implementation_sha256": { "type": "string", "pattern": "^[0-9a-f]{64}$" },
    "normalization_git_blob_sha": { "type": "string", "pattern": "^[0-9a-f]{40}$" },
    "pipeline_source_commit": { "type": "string", "pattern": "^[0-9a-f]{40}$" },
    "working_tree_dirty": { "type": "boolean" },
    "source_tree_digest": { "type": "string", "pattern": "^[0-9a-f]{64}$" }
  },
  "additionalProperties": false
}
```

Create `data/schemas/corpus-statistics.schema.json`:

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://lape26.org/schemas/corpus-statistics.schema.json",
  "title": "LAPE-26 Corpus Statistics",
  "type": "object",
  "required": [
    "artifactId", "artifactVersion", "characterFrequency", "withinWordBigrams",
    "crossWordBoundaryBigrams", "wordInitialLetters", "wordFinalLetters",
    "sentenceInitialLetters", "sentenceFinalLetters", "wordLengthStats",
    "sentenceLengthStats", "wordCount", "sentenceCount", "provenance"
  ],
  "properties": {
    "artifactId": { "const": "corpus-statistics-v0.1" },
    "artifactVersion": { "type": "string" },
    "characterFrequency": { "type": "object", "patternProperties": { "^[A-Z]$": { "type": "integer", "minimum": 0 } }, "additionalProperties": false },
    "withinWordBigrams": { "type": "object", "additionalProperties": { "type": "integer", "minimum": 0 } },
    "crossWordBoundaryBigrams": { "type": "object", "additionalProperties": { "type": "integer", "minimum": 0 } },
    "wordInitialLetters": { "type": "object", "additionalProperties": { "type": "integer", "minimum": 0 } },
    "wordFinalLetters": { "type": "object", "additionalProperties": { "type": "integer", "minimum": 0 } },
    "sentenceInitialLetters": { "type": "object", "additionalProperties": { "type": "integer", "minimum": 0 } },
    "sentenceFinalLetters": { "type": "object", "additionalProperties": { "type": "integer", "minimum": 0 } },
    "wordLengthStats": { "type": "object", "required": ["min", "max", "mean", "median"], "additionalProperties": false, "properties": { "min": { "type": "number" }, "max": { "type": "number" }, "mean": { "type": "number" }, "median": { "type": "number" } } },
    "sentenceLengthStats": { "type": "object", "required": ["min", "max", "mean", "median"], "additionalProperties": false, "properties": { "min": { "type": "number" }, "max": { "type": "number" }, "mean": { "type": "number" }, "median": { "type": "number" } } },
    "wordCount": { "type": "integer", "minimum": 0 },
    "sentenceCount": { "type": "integer", "minimum": 0 },
    "provenance": { "type": "object" }
  },
  "additionalProperties": false
}
```

Create `data/schemas/corpus-splits.schema.json`:

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://lape26.org/schemas/corpus-splits.schema.json",
  "title": "LAPE-26 Corpus Splits",
  "type": "object",
  "required": ["artifactId", "artifactVersion", "wordListSplit", "gutenbergSplit", "provenance"],
  "properties": {
    "artifactId": { "const": "corpus-splits-v0.1" },
    "artifactVersion": { "type": "string" },
    "wordListSplit": {
      "type": "object",
      "required": ["train", "validation", "holdout", "seed"],
      "additionalProperties": false,
      "properties": {
        "train": { "type": "array", "items": { "type": "string" } },
        "validation": { "type": "array", "items": { "type": "string" } },
        "holdout": { "type": "array", "items": { "type": "string" } },
        "seed": { "type": "integer" }
      }
    },
    "gutenbergSplit": {
      "type": "object",
      "required": ["train", "validation", "holdout", "seed"],
      "additionalProperties": false,
      "properties": {
        "train": { "type": "array", "items": { "type": "string" } },
        "validation": { "type": "array", "items": { "type": "string" } },
        "holdout": { "type": "array", "items": { "type": "string" } },
        "seed": { "type": "integer" }
      }
    },
    "provenance": { "type": "object" }
  },
  "additionalProperties": false
}
```

Create `data/schemas/pilot-stimulus.schema.json`:

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://lape26.org/schemas/pilot-stimulus.schema.json",
  "title": "LAPE-26 Pilot Stimulus Fixture",
  "type": "object",
  "required": ["artifactId", "artifactVersion", "seed", "coreSet", "orthographicChallengeSet", "interpretationBoundary", "representativenessBoundary", "provenance"],
  "properties": {
    "artifactId": { "const": "pilot-stimulus-v0.1" },
    "artifactVersion": { "type": "string" },
    "seed": { "type": "integer" },
    "coreSet": {
      "type": "array",
      "minItems": 108,
      "maxItems": 108,
      "items": {
        "type": "object",
        "required": ["word", "length", "lengthBand", "polarity", "vaderCompound", "partOfSpeech", "sourceDataset"],
        "additionalProperties": false,
        "properties": {
          "word": { "type": "string" },
          "length": { "type": "integer" },
          "lengthBand": { "enum": ["short", "medium", "long"] },
          "polarity": { "enum": ["positive", "negative", "neutral"] },
          "vaderCompound": { "type": "number" },
          "partOfSpeech": { "type": "string" },
          "sourceDataset": { "type": "string" }
        }
      }
    },
    "orthographicChallengeSet": {
      "type": "array",
      "minItems": 12,
      "maxItems": 12,
      "items": {
        "type": "object",
        "required": ["word", "category"],
        "additionalProperties": false,
        "properties": {
          "word": { "type": "string" },
          "category": { "enum": ["repeated-letters", "rare-letters", "vowel-heavy", "consonant-heavy"] }
        }
      }
    },
    "interpretationBoundary": { "type": "string" },
    "representativenessBoundary": { "type": "string" },
    "provenance": { "type": "object" }
  },
  "additionalProperties": false
}
```

Create `data/schemas/baseline-comparison.schema.json`:

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://lape26.org/schemas/baseline-comparison.schema.json",
  "title": "LAPE-26 Baseline Comparison Report",
  "type": "object",
  "required": ["reportId", "statement", "metricVersions", "mappingIds", "pipelineVersion", "results", "provenance"],
  "properties": {
    "reportId": { "const": "baseline-comparison-v0.1" },
    "statement": { "type": "string", "minLength": 1 },
    "metricVersions": {
      "type": "object",
      "required": ["register_center", "pitch_span", "interval_contour", "directional_balance", "repetition_index"],
      "additionalProperties": false,
      "properties": {
        "register_center": { "const": "register_center_v0.1" },
        "pitch_span": { "const": "pitch_span_v0.1" },
        "interval_contour": { "const": "interval_contour_v0.1" },
        "directional_balance": { "const": "directional_balance_v0.1" },
        "repetition_index": { "const": "repetition_index_v0.1" }
      }
    },
    "mappingIds": { "type": "array", "items": { "type": "string" } },
    "pipelineVersion": { "const": "corpus-pipeline-v0.1" },
    "results": { "type": "object" },
    "provenance": { "type": "object" }
  },
  "additionalProperties": false
}
```

Create `data/schemas/control-mapping.schema.json`:

```json
{
  "$schema": "https://json-schema.org/draft/2020-12/schema",
  "$id": "https://lape26.org/schemas/control-mapping.schema.json",
  "title": "LAPE-26 Control Mapping",
  "type": "object",
  "required": [
    "mappingId", "version", "status", "controlType", "generationMethod",
    "alphabet", "tuning", "range", "letters", "provenance"
  ],
  "properties": {
    "$schema": { "type": "string" },
    "mappingId": { "type": "string", "minLength": 1 },
    "version": { "type": "string" },
    "status": { "enum": ["experimental", "candidate", "stable", "deprecated"] },
    "controlType": { "enum": ["sequential-chromatic", "frequency-ranked", "circle-of-fifths", "random-seed"] },
    "generationMethod": { "type": "string", "minLength": 1 },
    "seed": { "type": ["integer", "null"] },
    "sourcePartition": { "type": ["string", "null"] },
    "alphabet": { "const": "ABCDEFGHIJKLMNOPQRSTUVWXYZ" },
    "tuning": {
      "type": "object",
      "required": ["system", "referencePitch", "referenceMidi", "referenceFrequencyHz"],
      "additionalProperties": false,
      "properties": {
        "system": { "const": "12-TET" },
        "referencePitch": { "const": "A4" },
        "referenceMidi": { "const": 69 },
        "referenceFrequencyHz": { "const": 440 },
        "frequencyFormula": { "type": "string" }
      }
    },
    "range": {
      "type": "object",
      "required": ["lowestMidi", "highestMidi"],
      "additionalProperties": false,
      "properties": {
        "lowestPitch": { "type": "string" },
        "lowestMidi": { "type": "integer" },
        "highestPitch": { "type": "string" },
        "highestMidi": { "type": "integer" }
      }
    },
    "letters": {
      "type": "object",
      "minProperties": 26,
      "maxProperties": 26,
      "patternProperties": {
        "^[A-Z]$": {
          "type": "object",
          "required": ["pitch", "midi", "frequencyHz"],
          "additionalProperties": false,
          "properties": {
            "pitch": { "type": "string" },
            "midi": { "type": "integer", "minimum": 0, "maximum": 127 },
            "frequencyHz": { "type": "number", "exclusiveMinimum": 0 }
          }
        }
      },
      "additionalProperties": false
    },
    "provenance": { "type": "object" }
  },
  "additionalProperties": false
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=python python3 -m unittest python.tests.test_schemas_valid -v`
Expected: PASS (7 subtests).

- [ ] **Step 5: Commit**

```bash
git add data/schemas/ python/tests/test_schemas_valid.py
git commit -m "feat: add JSON schemas for corpus pipeline artifacts"
```

---

## Task 2: Extract normalization into its own module

**Files:**
- Create: `python/lape26/normalize.py`
- Modify: `python/lape26/core.py`
- Test: `python/tests/test_normalize.py`

**Interfaces:**
- Produces: `lape26.normalize.normalize_text(text: str) -> str` (moved verbatim from `core.py`, same behavior).
- `lape26.core.normalize_text` continues to exist via re-export (existing `python/tests/test_core.py` and `python/lape26/__init__.py` import it from `lape26.core` and must keep working unmodified).

- [ ] **Step 1: Write the failing test**

Create `python/tests/test_normalize.py`:

```python
from __future__ import annotations

import unittest

from lape26.normalize import normalize_text


class NormalizeTests(unittest.TestCase):
    def test_uppercases_and_strips_non_letters(self) -> None:
        self.assertEqual(normalize_text("Café, don't! 26"), "CAFEDONT")

    def test_keeps_only_ascii_letters_after_decomposition(self) -> None:
        self.assertEqual(normalize_text("naïve"), "NAIVE")

    def test_empty_string(self) -> None:
        self.assertEqual(normalize_text(""), "")


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=python python3 -m unittest python.tests.test_normalize -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'lape26.normalize'`

- [ ] **Step 3: Create normalize.py and update core.py**

Create `python/lape26/normalize.py`:

```python
from __future__ import annotations

import unicodedata


def normalize_text(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed.upper() if "A" <= ch <= "Z")
```

In `python/lape26/core.py`, replace the existing `normalize_text` definition (lines defining `def normalize_text(text: str) -> str:` and its body) with an import, keeping everything else in the file unchanged:

```python
from __future__ import annotations

import json
import math
import unicodedata
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from .normalize import normalize_text

DEFAULT_MAPPING_PATH = Path(__file__).resolve().parents[2] / "mappings" / "lape-26-en-general-v0.1.json"


@dataclass(frozen=True)
class LetterEvent:
    sourceCharacter: str
    normalizedCharacter: str
    sourceIndex: int
    normalizedIndex: int
    pitch: str
    midi: int
    frequencyHz: float
    mappingVersion: str
    normalizationProfile: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def midi_to_frequency(midi: int, reference_hz: float = 440.0) -> float:
    if not 0 <= midi <= 127:
        raise ValueError("MIDI pitch must be between 0 and 127")
    return reference_hz * (2.0 ** ((midi - 69) / 12.0))


def load_mapping(path: str | Path = DEFAULT_MAPPING_PATH) -> dict[str, Any]:
    mapping_path = Path(path)
    data = json.loads(mapping_path.read_text(encoding="utf-8"))
    letters = data.get("letters", {})
    if set(letters) != set("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
        raise ValueError("Mapping must define exactly A-Z")
    midi_values = [entry["midi"] for entry in letters.values()]
    if len(midi_values) != len(set(midi_values)):
        raise ValueError("Mapping MIDI assignments must be unique")
    return data


def encode_text(text: str, mapping_path: str | Path = DEFAULT_MAPPING_PATH) -> list[dict[str, Any]]:
    mapping = load_mapping(mapping_path)
    normalized = normalize_text(text)
    events: list[dict[str, Any]] = []
    normalized_index = 0
    for source_index, source_character in enumerate(unicodedata.normalize("NFKD", text)):
        normalized_character = source_character.upper()
        if not ("A" <= normalized_character <= "Z"):
            continue
        entry = mapping["letters"][normalized_character]
        event = LetterEvent(
            sourceCharacter=source_character,
            normalizedCharacter=normalized_character,
            sourceIndex=source_index,
            normalizedIndex=normalized_index,
            pitch=entry["pitch"],
            midi=int(entry["midi"]),
            frequencyHz=midi_to_frequency(int(entry["midi"])),
            mappingVersion=mapping["mappingId"],
            normalizationProfile=mapping["normalizationProfile"],
        )
        events.append(event.to_dict())
        normalized_index += 1

    if "".join(event["normalizedCharacter"] for event in events) != normalized:
        raise RuntimeError("Normalization and encoding paths diverged")
    return events
```

- [ ] **Step 4: Run all Python tests to verify nothing broke and the new test passes**

Run: `PYTHONPATH=python python3 -m unittest discover -s python/tests -v`
Expected: All tests pass, including `test_normalize` (3 tests) and the existing `test_core` (5 tests).

- [ ] **Step 5: Commit**

```bash
git add python/lape26/normalize.py python/lape26/core.py python/tests/test_normalize.py
git commit -m "refactor: extract normalize_text into its own module for provenance tracking"
```

---

## Task 3: Research dependencies and gitignore

**Files:**
- Create: `requirements-research.txt`
- Modify: `pyproject.toml`
- Modify: `.gitignore`

**Interfaces:**
- Produces: a pinned, installable dependency set (`nltk`, `pyyaml`) used by every later task that imports `nltk` or `yaml`.

- [ ] **Step 1: Create requirements-research.txt**

```text
nltk==3.9.1
pyyaml==6.0.2
jsonschema==4.23.0
```

`jsonschema` validates the committed artifacts against `data/schemas/*.json` in CI (Task 16) — this is a plain PyPI package install, not corpus data acquisition, so it's fine for CI to install it.

- [ ] **Step 2: Add optional-dependencies group to pyproject.toml**

In `pyproject.toml`, add after the existing `[project.scripts]` block:

```toml
[project.optional-dependencies]
research = ["nltk>=3.9,<4", "pyyaml>=6.0,<7", "jsonschema>=4.23,<5"]
```

- [ ] **Step 3: Update .gitignore**

Append to `.gitignore`:

```text
# Corpus pipeline (research)
data/raw/nltk_data/
data/processed/.tmp/
data/processed/.check/
```

(`__pycache__/` and `*.pyc` are already present in the existing Python section — no duplicate needed.)

- [ ] **Step 4: Install and verify**

Run: `pip3 install -r requirements-research.txt`
Expected: `nltk` and `pyyaml` install successfully.

Run: `python3 -c "import nltk, yaml, jsonschema; print(nltk.__version__, yaml.__version__, jsonschema.__version__)"`
Expected: prints all three version strings without error.

- [ ] **Step 5: Commit**

```bash
git add requirements-research.txt pyproject.toml .gitignore
git commit -m "build: add pinned NLTK/PyYAML research dependencies"
```

---

## Task 4: Dataset manifests and corpus README

**Files:**
- Create: `data/manifests/gutenberg.yaml`
- Create: `data/manifests/words.yaml`
- Create: `data/manifests/wordnet.yaml`
- Create: `data/manifests/opinion-lexicon.yaml`
- Create: `data/manifests/vader.yaml`
- Create: `data/corpus/README.md`
- Test: `python/tests/test_dataset_manifests.py`

**Interfaces:**
- Produces: 5 YAML files, each validating against `data/schemas/dataset-manifest.schema.json` (Task 1), consumed by `scripts/setup_corpus.py` (Task 13) to know which NLTK package IDs are approved.
- `checksum_sha256` in each manifest is intentionally `""` at commit time — it is not a placeholder-to-fill-in-later omission, it's populated by `make corpus-lock` cross-referencing `data/manifests/corpus-lock.json` (Task 10), since these are NLTK-hosted corpora with no single static upstream file to hash in advance.

- [ ] **Step 1: Write the failing test**

Create `python/tests/test_dataset_manifests.py`:

```python
from __future__ import annotations

import json
import unittest
from pathlib import Path

import yaml
from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[2]
MANIFESTS_DIR = ROOT / "data" / "manifests"
SCHEMA_PATH = ROOT / "data" / "schemas" / "dataset-manifest.schema.json"

EXPECTED_MANIFESTS = {
    "gutenberg.yaml": "gutenberg",
    "words.yaml": "words",
    "wordnet.yaml": "wordnet",
    "opinion-lexicon.yaml": "opinion_lexicon",
    "vader.yaml": "vader_lexicon",
}

FORBIDDEN_DATASET_IDS = {"brown", "names", "cmudict"}


class DatasetManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.validator = Draft202012Validator(json.loads(SCHEMA_PATH.read_text(encoding="utf-8")))

    def test_all_five_manifests_exist_and_validate(self) -> None:
        for filename, expected_package_id in EXPECTED_MANIFESTS.items():
            path = MANIFESTS_DIR / filename
            with self.subTest(filename=filename):
                self.assertTrue(path.exists(), f"missing {path}")
                manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
                self.validator.validate(manifest)
                self.assertEqual(manifest["dataset_id"], expected_package_id)

    def test_no_forbidden_datasets_present(self) -> None:
        for path in MANIFESTS_DIR.glob("*.yaml"):
            manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
            self.assertNotIn(manifest.get("dataset_id"), FORBIDDEN_DATASET_IDS)

    def test_opinion_lexicon_and_vader_roles_exclude_musical_objective(self) -> None:
        for filename in ("opinion-lexicon.yaml", "vader.yaml"):
            manifest = yaml.safe_load((MANIFESTS_DIR / filename).read_text(encoding="utf-8"))
            self.assertIn(
                "musical objective",
                " ".join(manifest.get("must_not_be_used_for", [])).lower(),
            )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=python python3 -m unittest python.tests.test_dataset_manifests -v`
Expected: FAIL — manifest files don't exist yet.

- [ ] **Step 3: Create the 5 manifests and data/corpus/README.md**

Create `data/manifests/gutenberg.yaml`:

```yaml
dataset_id: gutenberg
name: NLTK Gutenberg Sample Corpus
version: "nltk-3.9.1-bundled"
source_url: "https://www.nltk.org/nltk_data/ (package: gutenberg)"
retrieved_at: "2026-07-29"
license: "Public Domain (US) — Project Gutenberg texts, bundled by NLTK"
redistribution_allowed: true
redistribution_notes: >
  Public-domain source texts. This project redistributes only derived
  aggregate statistics, splits, and comparison reports — not full text.
content_type: natural-text-corpus
language: en
locale: general
checksum_sha256: ""
role: >
  Natural text: sentences, word/sentence boundaries, character and bigram
  transitions. Primary source of natural character/bigram frequency and
  sentence-length statistics. Source for the frequency-ranked control
  mapping (train partition only).
must_not_be_used_for: []
preparation_steps:
  - "nltk.download('gutenberg') via scripts/setup_corpus.py (make corpus-setup)"
  - "nltk.corpus.gutenberg.fileids() enumerates the ~18 bundled documents"
  - "nltk.corpus.gutenberg.sents(fileid) yields sentences as word-token lists per document"
  - "Each word token normalized via lape26.normalize.normalize_text before statistics"
known_biases:
  - "19th-century British/American literary English; not representative of contemporary usage"
  - "Heavily skewed toward a small number of authors (Austen, Shakespeare, Bible, Melville, etc.)"
  - "Only ~18 documents — Gutenberg-derived splits and frequencies have limited statistical power"
```

Create `data/manifests/words.yaml`:

```yaml
dataset_id: words
name: NLTK Words Corpus
version: "nltk-3.9.1-bundled"
source_url: "https://www.nltk.org/nltk_data/ (package: words)"
retrieved_at: "2026-07-29"
license: "Public Domain / unrestricted word list (derived from Unix words file)"
redistribution_allowed: true
redistribution_notes: >
  Unrestricted word list; only the 80/10/10 split assignment and the
  120-item stimulus fixture's selected words are redistributed, not the
  full list.
content_type: word-list
language: en
locale: general
checksum_sha256: ""
role: >
  Candidate vocabulary and orthographic stress cases (repeated/rare
  letters, vowel/consonant-heavy words) for the pilot stimulus fixture.
  Source for the word-list 80/10/10 train/validation/holdout split.
must_not_be_used_for:
  - "Natural usage-frequency statistics (use Gutenberg instead)"
preparation_steps:
  - "nltk.download('words') via scripts/setup_corpus.py (make corpus-setup)"
  - "nltk.corpus.words.words() yields the flat word list"
  - "Deduplicated and normalized via lape26.normalize.normalize_text"
known_biases:
  - "Includes many obscure/archaic entries; not frequency-weighted"
  - "No part-of-speech or usage information — used only for candidate generation"
```

Create `data/manifests/wordnet.yaml`:

```yaml
dataset_id: wordnet
name: Princeton WordNet (via NLTK)
version: "3.0-bundled-nltk-3.9.1"
source_url: "https://wordnet.princeton.edu/ (package: wordnet, via NLTK)"
retrieved_at: "2026-07-29"
license: "WordNet License (permissive; free for any purpose with copyright/permission notice retained)"
redistribution_allowed: true
redistribution_notes: >
  Only per-item part-of-speech and validation results for the 120-item
  stimulus fixture are redistributed, not the full WordNet database.
content_type: lexical-database
language: en
locale: general
checksum_sha256: ""
role: >
  Dictionary validation (real-word check), part-of-speech tagging, and
  morphological-variant grouping (stem key) for stimulus candidates.
must_not_be_used_for:
  - "Usage-frequency inference — WordNet membership is not a frequency signal"
preparation_steps:
  - "nltk.download('wordnet') via scripts/setup_corpus.py (make corpus-setup)"
  - "nltk.corpus.wordnet.synsets(word) validates real-word membership and yields part-of-speech"
  - "nltk.corpus.wordnet.morphy(word) yields a lemma used as the morphological stem key"
known_biases:
  - "General-purpose lexicon; may lack domain-specific, slang, or very recent vocabulary"
  - "Multiple senses per word; part-of-speech selection uses the first synset's POS"
```

Create `data/manifests/opinion-lexicon.yaml`:

```yaml
dataset_id: opinion_lexicon
name: Hu & Liu Opinion Lexicon (via NLTK)
version: "2004-bundled-nltk-3.9.1"
source_url: "https://www.cs.uic.edu/~liub/FBS/sentiment-analysis.html (package: opinion_lexicon, via NLTK)"
retrieved_at: "2026-07-29"
license: >
  Published for academic/research use (Hu & Liu, KDD 2004). No explicit
  commercial-redistribution grant found; this project's use is treated as
  research use, and only small per-item derived facts for the 120-item
  stimulus fixture are published, per project maintainer decision recorded
  during design review (see docs/superpowers/specs/2026-07-29-phase-1-corpus-evaluation-foundation-design.md).
redistribution_allowed: true
redistribution_notes: >
  Treated as redistributable for this project's derived, per-item stimulus
  metadata (word, positive/negative label, VADER score) under research/
  academic use. Full lexicon files stay in data/raw/nltk_data/ (gitignored).
content_type: sentiment-lexicon
language: en
locale: general
checksum_sha256: ""
role: "Positive/negative candidate word labels for the pilot stimulus fixture."
must_not_be_used_for:
  - "Directly influencing the musical objective or any control mapping"
  - "Character or bigram frequency statistics"
preparation_steps:
  - "nltk.download('opinion_lexicon') via scripts/setup_corpus.py (make corpus-setup)"
  - "nltk.corpus.opinion_lexicon.positive() and .negative() yield the two word lists"
known_biases:
  - "Built from product-review text; polarity judgments may not generalize to all registers"
  - "Binary positive/negative only — no intensity scoring (VADER supplies that)"
citation: "Minqing Hu and Bing Liu, \"Mining and Summarizing Customer Reviews\", KDD 2004."
```

Create `data/manifests/vader.yaml`:

```yaml
dataset_id: vader_lexicon
name: VADER Sentiment Lexicon (via NLTK)
version: "2014-bundled-nltk-3.9.1"
source_url: "https://github.com/cjhutto/vaderSentiment (package: vader_lexicon, via NLTK)"
retrieved_at: "2026-07-29"
license: "MIT License"
redistribution_allowed: true
redistribution_notes: >
  MIT-licensed; per-item VADER compound scores for the 120-item stimulus
  fixture are redistributed freely.
content_type: sentiment-lexicon
language: en
locale: general
checksum_sha256: ""
role: "Sentiment scoring and polarity confirmation for pilot stimulus candidates."
must_not_be_used_for:
  - "Directly influencing the musical objective or any control mapping"
preparation_steps:
  - "nltk.download('vader_lexicon') via scripts/setup_corpus.py (make corpus-setup)"
  - "nltk.sentiment.vader.SentimentIntensityAnalyzer().polarity_scores(word)['compound'] scores each candidate"
known_biases:
  - "Tuned on social-media text; single-word scoring may differ from in-context sentiment"
citation: "Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. ICWSM-14."
```

Create `data/corpus/README.md`:

```markdown
# Corpus Pipeline

This directory documents the Phase 2 ("Corpus and stimulus foundation")
corpus and stimulus pipeline. See the full design spec at
`docs/superpowers/specs/2026-07-29-phase-1-corpus-evaluation-foundation-design.md`.

## Normalization provisional-freeze notice

This pipeline treats `lape-text-normalization-v0.1` as a provisional
frozen profile for corpus-generation purposes. It has not been formally
finalized under ROADMAP Phase 1. Any change to the profile or its
implementation requires regeneration and re-versioning of all dependent
corpus statistics, splits, stimulus fixtures, control mappings derived
from corpus statistics, and comparison reports.

## Dataset roles

| Dataset | Role | Must NOT be used for |
|---|---|---|
| Gutenberg | Natural text, sentences, boundaries, character/bigram transitions, sentence-length stats, primary source of natural character/bigram frequency | — |
| words | Candidate vocabulary, orthographic stress cases | Natural usage frequency |
| WordNet | Dictionary validation, part-of-speech, morphological grouping | Usage-frequency inference |
| Opinion Lexicon | Positive/negative candidate labels | Directly influencing the musical objective |
| VADER | Sentiment scoring / polarity confirmation | Directly influencing the musical objective |

Excluded entirely: Brown corpus, Names corpus, CMUdict.

## Directory map

```text
data/
  manifests/          Dataset manifests + corpus-lock.json
  corpus/README.md    This file
  schemas/            JSON Schemas for every generated artifact
  raw/nltk_data/       Downloaded corpora (gitignored, local only)
  processed/corpus/    Statistics, splits, comparison report (committed)
  fixtures/pilot-stimulus-v0.1.json   120-item fixture (committed)
mappings/controls/     Generated control mapping JSON files (committed)
```

## Commands

- `make corpus-setup` — downloads exactly the 5 approved NLTK packages, writes `data/manifests/corpus-lock.json`. Local only, never run in CI, never touches undeclared packages.
- `make corpus-relock` — explicit, intentional lock regeneration from the current local cache.
- `make corpus-pipeline` — offline; verifies the lock, then generates statistics, splits, the stimulus fixture, control mappings, and the baseline report.
- `make corpus-check` — local-only maintainer safeguard; regenerates into a temp directory and diffs against committed artifacts. Not run in CI.

## Leakage prevention

The `frequency-ranked` control mapping is built from the Gutenberg
**train** partition only, then evaluated against validation, holdout, and
the pilot fixture. The 120-item pilot stimulus fixture is not a member of
the train/validation/holdout split — it is a separate, frozen,
independently versioned evaluation fixture whose candidate words are drawn
only from training-side vocabulary.
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=python python3 -m unittest python.tests.test_dataset_manifests -v`
Expected: PASS (3 tests, one with 5 subtests).

- [ ] **Step 5: Commit**

```bash
git add data/manifests/gutenberg.yaml data/manifests/words.yaml data/manifests/wordnet.yaml \
        data/manifests/opinion-lexicon.yaml data/manifests/vader.yaml data/corpus/README.md \
        python/tests/test_dataset_manifests.py
git commit -m "docs: add license-aware dataset manifests and corpus pipeline README"
```

---

## Task 5: CorpusToken structure preservation

**Files:**
- Create: `python/lape26/corpus/__init__.py`
- Create: `python/lape26/corpus/tokens.py`
- Test: `python/tests/test_corpus_tokens.py`

**Interfaces:**
- Produces:
  - `WordToken` dataclass: `sourceText: str, normalizedText: str, sentenceIndex: int, wordIndexInSentence: int, isSentenceInitial: bool, isSentenceFinal: bool`
  - `SentenceToken` dataclass: `words: tuple[WordToken, ...], sentenceIndex: int`
  - `tokenize_sentences(raw_sentences: list[list[str]]) -> list[SentenceToken]` — takes pre-split sentences (each a list of raw word strings; sentence/word splitting itself is the adapter layer's job in Task 10), normalizes each word via `lape26.normalize.normalize_text`, drops words that normalize to empty (pure punctuation), and marks sentence-initial/final words. Empty sentences (all-punctuation) are dropped entirely.
- Consumed by: Task 6 (`stats.py`), Task 9 (`controls.py` frequency-ranked), Task 12 (`report.py`).

- [ ] **Step 1: Write the failing test**

Create `python/lape26/corpus/__init__.py`:

```python
"""LAPE-26 corpus and stimulus pipeline (research tooling, not part of the reference encoder)."""
```

Create `python/tests/test_corpus_tokens.py`:

```python
from __future__ import annotations

import unittest

from lape26.corpus.tokens import tokenize_sentences


class TokenizeSentencesTests(unittest.TestCase):
    def test_marks_sentence_initial_and_final(self) -> None:
        sentences = tokenize_sentences([["The", "cat", "sat", "."]])
        self.assertEqual(len(sentences), 1)
        words = sentences[0].words
        self.assertEqual([w.normalizedText for w in words], ["THE", "CAT", "SAT"])
        self.assertTrue(words[0].isSentenceInitial)
        self.assertFalse(words[0].isSentenceFinal)
        self.assertTrue(words[-1].isSentenceFinal)
        self.assertFalse(words[-1].isSentenceInitial)

    def test_pure_punctuation_tokens_are_dropped(self) -> None:
        sentences = tokenize_sentences([["Hello", ",", "world", "!"]])
        self.assertEqual([w.normalizedText for w in sentences[0].words], ["HELLO", "WORLD"])

    def test_all_punctuation_sentence_is_dropped_entirely(self) -> None:
        sentences = tokenize_sentences([["..."], ["Real", "sentence", "."]])
        self.assertEqual(len(sentences), 1)
        self.assertEqual([w.normalizedText for w in sentences[0].words], ["REAL", "SENTENCE"])

    def test_single_word_sentence_is_both_initial_and_final(self) -> None:
        sentences = tokenize_sentences([["Stop", "."]])
        word = sentences[0].words[0]
        self.assertTrue(word.isSentenceInitial)
        self.assertTrue(word.isSentenceFinal)

    def test_sentence_index_is_preserved_across_dropped_sentences(self) -> None:
        sentences = tokenize_sentences([["..."], ["One", "."], ["Two", "."]])
        self.assertEqual([s.sentenceIndex for s in sentences], [1, 2])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=python python3 -m unittest python.tests.test_corpus_tokens -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lape26.corpus'`

- [ ] **Step 3: Write minimal implementation**

Create `python/lape26/corpus/tokens.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

from ..normalize import normalize_text


@dataclass(frozen=True)
class WordToken:
    sourceText: str
    normalizedText: str
    sentenceIndex: int
    wordIndexInSentence: int
    isSentenceInitial: bool
    isSentenceFinal: bool


@dataclass(frozen=True)
class SentenceToken:
    words: tuple[WordToken, ...]
    sentenceIndex: int


def tokenize_sentences(raw_sentences: list[list[str]]) -> list[SentenceToken]:
    sentences: list[SentenceToken] = []
    for sentence_index, raw_words in enumerate(raw_sentences):
        normalized_pairs = [(raw, normalize_text(raw)) for raw in raw_words]
        normalized_pairs = [(raw, norm) for raw, norm in normalized_pairs if norm]
        if not normalized_pairs:
            continue

        last_index = len(normalized_pairs) - 1
        words = tuple(
            WordToken(
                sourceText=raw_word,
                normalizedText=normalized,
                sentenceIndex=sentence_index,
                wordIndexInSentence=word_index,
                isSentenceInitial=word_index == 0,
                isSentenceFinal=word_index == last_index,
            )
            for word_index, (raw_word, normalized) in enumerate(normalized_pairs)
        )
        sentences.append(SentenceToken(words=words, sentenceIndex=sentence_index))
    return sentences
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=python python3 -m unittest python.tests.test_corpus_tokens -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add python/lape26/corpus/__init__.py python/lape26/corpus/tokens.py python/tests/test_corpus_tokens.py
git commit -m "feat: add CorpusToken structure preserving word/sentence boundaries"
```

---

## Task 6: Corpus statistics

**Files:**
- Create: `python/lape26/corpus/stats.py`
- Test: `python/tests/test_corpus_stats.py`

**Interfaces:**
- Consumes: `SentenceToken`/`WordToken` from Task 5 (`lape26.corpus.tokens`).
- Produces: `CorpusStatistics` dataclass with fields `characterFrequency, withinWordBigrams, crossWordBoundaryBigrams, wordInitialLetters, wordFinalLetters, sentenceInitialLetters, sentenceFinalLetters, wordLengthStats, sentenceLengthStats, wordCount, sentenceCount` (all dict[str,int] except the two length-stats dicts which are `{"min","max","mean","median"} -> float`), and `compute_statistics(sentences: list[SentenceToken]) -> CorpusStatistics`.
- Consumed by: Task 9 (`controls.py`, frequency-ranked), Task 14 (`build_corpus_pipeline.py`).

- [ ] **Step 1: Write the failing test**

Create `python/tests/test_corpus_stats.py`:

```python
from __future__ import annotations

import unittest

from lape26.corpus.stats import compute_statistics
from lape26.corpus.tokens import tokenize_sentences


class ComputeStatisticsTests(unittest.TestCase):
    def test_within_word_bigrams_single_word(self) -> None:
        stats = compute_statistics(tokenize_sentences([["cat"]]))
        self.assertEqual(stats.withinWordBigrams, {"AT": 1, "CA": 1})
        self.assertEqual(stats.characterFrequency, {"A": 1, "C": 1, "T": 1})
        self.assertEqual(stats.crossWordBoundaryBigrams, {})

    def test_cross_word_transition_not_counted_as_within_word(self) -> None:
        stats = compute_statistics(tokenize_sentences([["cat", "dog"]]))
        self.assertEqual(stats.withinWordBigrams, {"AT": 1, "CA": 1, "DO": 1, "OG": 1})
        self.assertEqual(stats.crossWordBoundaryBigrams, {"TD": 1})
        self.assertNotIn("TD", stats.withinWordBigrams)

    def test_word_and_sentence_boundary_letters(self) -> None:
        stats = compute_statistics(tokenize_sentences([["cat", "dog"]]))
        self.assertEqual(stats.wordInitialLetters, {"C": 1, "D": 1})
        self.assertEqual(stats.wordFinalLetters, {"G": 1, "T": 1})
        self.assertEqual(stats.sentenceInitialLetters, {"C": 1})
        self.assertEqual(stats.sentenceFinalLetters, {"G": 1})

    def test_word_and_sentence_length_stats(self) -> None:
        stats = compute_statistics(tokenize_sentences([["cat", "dog"], ["a"]]))
        self.assertEqual(stats.wordCount, 3)
        self.assertEqual(stats.sentenceCount, 2)
        self.assertEqual(stats.wordLengthStats, {"min": 1.0, "max": 3.0, "mean": 7 / 3, "median": 3.0})
        self.assertEqual(stats.sentenceLengthStats, {"min": 1.0, "max": 2.0, "mean": 1.5, "median": 1.5})

    def test_no_cross_word_bigram_across_sentence_boundary(self) -> None:
        stats = compute_statistics(tokenize_sentences([["cat"], ["dog"]]))
        self.assertEqual(stats.crossWordBoundaryBigrams, {})


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=python python3 -m unittest python.tests.test_corpus_stats -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lape26.corpus.stats'`

- [ ] **Step 3: Write minimal implementation**

Create `python/lape26/corpus/stats.py`:

```python
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from statistics import mean, median

from .tokens import SentenceToken


@dataclass(frozen=True)
class CorpusStatistics:
    characterFrequency: dict[str, int]
    withinWordBigrams: dict[str, int]
    crossWordBoundaryBigrams: dict[str, int]
    wordInitialLetters: dict[str, int]
    wordFinalLetters: dict[str, int]
    sentenceInitialLetters: dict[str, int]
    sentenceFinalLetters: dict[str, int]
    wordLengthStats: dict[str, float]
    sentenceLengthStats: dict[str, float]
    wordCount: int
    sentenceCount: int


def compute_statistics(sentences: list[SentenceToken]) -> CorpusStatistics:
    character_frequency: Counter[str] = Counter()
    within_word_bigrams: Counter[str] = Counter()
    cross_word_bigrams: Counter[str] = Counter()
    word_initial: Counter[str] = Counter()
    word_final: Counter[str] = Counter()
    sentence_initial: Counter[str] = Counter()
    sentence_final: Counter[str] = Counter()
    word_lengths: list[int] = []
    sentence_lengths: list[int] = []

    for sentence in sentences:
        sentence_lengths.append(len(sentence.words))
        previous_final_letter: str | None = None
        for word in sentence.words:
            text = word.normalizedText
            character_frequency.update(text)
            for a, b in zip(text, text[1:]):
                within_word_bigrams[f"{a}{b}"] += 1
            word_lengths.append(len(text))
            word_initial[text[0]] += 1
            word_final[text[-1]] += 1
            if word.isSentenceInitial:
                sentence_initial[text[0]] += 1
            if word.isSentenceFinal:
                sentence_final[text[-1]] += 1
            if previous_final_letter is not None:
                cross_word_bigrams[f"{previous_final_letter}{text[0]}"] += 1
            previous_final_letter = text[-1]

    return CorpusStatistics(
        characterFrequency=dict(sorted(character_frequency.items())),
        withinWordBigrams=dict(sorted(within_word_bigrams.items())),
        crossWordBoundaryBigrams=dict(sorted(cross_word_bigrams.items())),
        wordInitialLetters=dict(sorted(word_initial.items())),
        wordFinalLetters=dict(sorted(word_final.items())),
        sentenceInitialLetters=dict(sorted(sentence_initial.items())),
        sentenceFinalLetters=dict(sorted(sentence_final.items())),
        wordLengthStats=_length_stats(word_lengths),
        sentenceLengthStats=_length_stats(sentence_lengths),
        wordCount=len(word_lengths),
        sentenceCount=len(sentences),
    )


def _length_stats(lengths: list[int]) -> dict[str, float]:
    if not lengths:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0}
    return {
        "min": float(min(lengths)),
        "max": float(max(lengths)),
        "mean": float(mean(lengths)),
        "median": float(median(lengths)),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=python python3 -m unittest python.tests.test_corpus_stats -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add python/lape26/corpus/stats.py python/tests/test_corpus_stats.py
git commit -m "feat: compute corpus statistics with within/cross-word bigram separation"
```

---

## Task 7: Deterministic leakage-safe splits

**Files:**
- Create: `python/lape26/corpus/splits.py`
- Test: `python/tests/test_corpus_splits.py`

**Interfaces:**
- Produces:
  - `WordListSplit` dataclass: `train: tuple[str,...], validation: tuple[str,...], holdout: tuple[str,...], seed: int`
  - `split_word_list(words: list[str], seed: int, train_ratio: float = 0.8, validation_ratio: float = 0.1) -> WordListSplit`
  - `DocumentSplit` dataclass: same shape as `WordListSplit` but partitions hold document IDs.
  - `split_documents_by_word_count(document_word_counts: dict[str, int], seed: int, train_ratio: float = 0.8, validation_ratio: float = 0.1) -> DocumentSplit` — deterministic greedy bin-balancing that keeps every document intact in exactly one partition while approximating the target word-count ratios (assigns each shuffled document to whichever partition is furthest below its target share).
- Consumed by: Task 9 (`controls.py`, frequency-ranked uses the Gutenberg train partition), Task 14 (`build_corpus_pipeline.py`).

- [ ] **Step 1: Write the failing test**

Create `python/tests/test_corpus_splits.py`:

```python
from __future__ import annotations

import unittest

from lape26.corpus.splits import split_documents_by_word_count, split_word_list


class SplitWordListTests(unittest.TestCase):
    def test_no_overlap_and_full_coverage(self) -> None:
        words = [f"word{i}" for i in range(20)]
        result = split_word_list(words, seed=26)
        self.assertEqual(set(result.train) & set(result.validation), set())
        self.assertEqual(set(result.train) & set(result.holdout), set())
        self.assertEqual(set(result.validation) & set(result.holdout), set())
        self.assertEqual(
            set(result.train) | set(result.validation) | set(result.holdout),
            set(words),
        )

    def test_approximate_80_10_10_proportions(self) -> None:
        words = [f"word{i}" for i in range(20)]
        result = split_word_list(words, seed=26)
        self.assertEqual(len(result.train), 16)
        self.assertEqual(len(result.validation), 2)
        self.assertEqual(len(result.holdout), 2)

    def test_deterministic_given_same_seed(self) -> None:
        words = [f"word{i}" for i in range(20)]
        first = split_word_list(words, seed=26)
        second = split_word_list(words, seed=26)
        self.assertEqual(first, second)

    def test_deduplicates_input(self) -> None:
        result = split_word_list(["cat", "cat", "dog"], seed=26)
        total = len(result.train) + len(result.validation) + len(result.holdout)
        self.assertEqual(total, 2)


class SplitDocumentsByWordCountTests(unittest.TestCase):
    def _sample_counts(self) -> dict[str, int]:
        return {f"doc{i}": 1000 + i * 137 for i in range(18)}

    def test_every_document_assigned_exactly_once(self) -> None:
        counts = self._sample_counts()
        result = split_documents_by_word_count(counts, seed=26)
        all_assigned = list(result.train) + list(result.validation) + list(result.holdout)
        self.assertEqual(sorted(all_assigned), sorted(counts))
        self.assertEqual(len(all_assigned), len(set(all_assigned)))

    def test_deterministic_given_same_seed(self) -> None:
        counts = self._sample_counts()
        first = split_documents_by_word_count(counts, seed=26)
        second = split_documents_by_word_count(counts, seed=26)
        self.assertEqual(first, second)

    def test_train_partition_captures_majority_of_word_count(self) -> None:
        counts = self._sample_counts()
        result = split_documents_by_word_count(counts, seed=26)
        train_words = sum(counts[doc] for doc in result.train)
        total_words = sum(counts.values())
        self.assertGreater(train_words / total_words, 0.6)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=python python3 -m unittest python.tests.test_corpus_splits -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lape26.corpus.splits'`

- [ ] **Step 3: Write minimal implementation**

Create `python/lape26/corpus/splits.py`:

```python
from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class WordListSplit:
    train: tuple[str, ...]
    validation: tuple[str, ...]
    holdout: tuple[str, ...]
    seed: int


def split_word_list(
    words: list[str],
    seed: int,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
) -> WordListSplit:
    deduplicated = sorted(set(words))
    rng = random.Random(seed)
    shuffled = deduplicated[:]
    rng.shuffle(shuffled)

    total = len(shuffled)
    train_end = round(total * train_ratio)
    validation_end = train_end + round(total * validation_ratio)

    return WordListSplit(
        train=tuple(sorted(shuffled[:train_end])),
        validation=tuple(sorted(shuffled[train_end:validation_end])),
        holdout=tuple(sorted(shuffled[validation_end:])),
        seed=seed,
    )


@dataclass(frozen=True)
class DocumentSplit:
    train: tuple[str, ...]
    validation: tuple[str, ...]
    holdout: tuple[str, ...]
    seed: int


def split_documents_by_word_count(
    document_word_counts: dict[str, int],
    seed: int,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
) -> DocumentSplit:
    """Assign whole documents to train/validation/holdout, keeping every
    document intact in exactly one partition, while approximating the
    target word-count ratios via deterministic greedy bin-balancing:
    documents are shuffled with `seed`, then each is assigned to whichever
    partition is currently furthest below its target word-count share.
    """
    document_ids = sorted(document_word_counts)
    rng = random.Random(seed)
    shuffled_ids = document_ids[:]
    rng.shuffle(shuffled_ids)

    total_words = sum(document_word_counts.values())
    holdout_ratio = 1 - train_ratio - validation_ratio
    targets = {
        "train": total_words * train_ratio,
        "validation": total_words * validation_ratio,
        "holdout": total_words * holdout_ratio,
    }
    assigned: dict[str, list[str]] = {"train": [], "validation": [], "holdout": []}
    running: dict[str, int] = {"train": 0, "validation": 0, "holdout": 0}

    for document_id in shuffled_ids:
        deficits = {name: targets[name] - running[name] for name in targets}
        chosen = max(deficits, key=lambda name: deficits[name])
        assigned[chosen].append(document_id)
        running[chosen] += document_word_counts[document_id]

    return DocumentSplit(
        train=tuple(sorted(assigned["train"])),
        validation=tuple(sorted(assigned["validation"])),
        holdout=tuple(sorted(assigned["holdout"])),
        seed=seed,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=python python3 -m unittest python.tests.test_corpus_splits -v`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add python/lape26/corpus/splits.py python/tests/test_corpus_splits.py
git commit -m "feat: add leakage-safe deterministic word-list and document splits"
```

---

## Task 8: Provenance block builder

**Files:**
- Create: `python/lape26/corpus/provenance.py`
- Test: `python/tests/test_corpus_provenance.py`

**Interfaces:**
- Produces:
  - Constants: `NORMALIZATION_PROFILE_ID = "lape-text-normalization-v0.1"`, `NORMALIZATION_STATUS = "provisional-frozen"`, `NORMALIZATION_SPEC_PATH = "specs/text-normalization.md"`, `NORMALIZATION_IMPLEMENTATION_PATH = "python/lape26/normalize.py"`
  - `sha256_file(path: Path) -> str`
  - `git_blob_sha(relative_path: str) -> str`
  - `git_head_commit() -> str`
  - `git_is_dirty() -> bool`
  - `source_tree_digest(paths: list[str]) -> str`
  - `build_provenance_block(*, pipeline_source_paths: list[str]) -> dict[str, object]` — returns exactly the fields required by `data/schemas/corpus-provenance.schema.json` (Task 1).
- Consumed by: every artifact generator (Tasks 9, 11, 12, 14) and the CI checker (Task 16).

- [ ] **Step 1: Write the failing test**

Create `python/tests/test_corpus_provenance.py`:

```python
from __future__ import annotations

import re
import unittest
from pathlib import Path

from lape26.corpus.provenance import (
    NORMALIZATION_IMPLEMENTATION_PATH,
    NORMALIZATION_PROFILE_ID,
    NORMALIZATION_SPEC_PATH,
    NORMALIZATION_STATUS,
    REPO_ROOT,
    build_provenance_block,
    sha256_file,
)

HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class ProvenanceBlockTests(unittest.TestCase):
    def test_provenance_block_has_all_required_fields(self) -> None:
        block = build_provenance_block(pipeline_source_paths=[NORMALIZATION_IMPLEMENTATION_PATH])
        self.assertEqual(block["normalization_profile_id"], NORMALIZATION_PROFILE_ID)
        self.assertEqual(block["normalization_status"], NORMALIZATION_STATUS)
        self.assertEqual(block["normalization_spec_path"], NORMALIZATION_SPEC_PATH)
        self.assertEqual(block["normalization_implementation"], NORMALIZATION_IMPLEMENTATION_PATH)
        self.assertRegex(block["normalization_implementation_sha256"], HEX64)
        self.assertRegex(block["normalization_git_blob_sha"], HEX40)
        self.assertRegex(block["pipeline_source_commit"], HEX40)
        self.assertIsInstance(block["working_tree_dirty"], bool)
        self.assertRegex(block["source_tree_digest"], HEX64)

    def test_sha256_file_matches_live_normalize_module(self) -> None:
        path = REPO_ROOT / NORMALIZATION_IMPLEMENTATION_PATH
        block = build_provenance_block(pipeline_source_paths=[NORMALIZATION_IMPLEMENTATION_PATH])
        self.assertEqual(block["normalization_implementation_sha256"], sha256_file(path))

    def test_source_tree_digest_is_deterministic_for_same_paths(self) -> None:
        first = build_provenance_block(pipeline_source_paths=[NORMALIZATION_IMPLEMENTATION_PATH])
        second = build_provenance_block(pipeline_source_paths=[NORMALIZATION_IMPLEMENTATION_PATH])
        self.assertEqual(first["source_tree_digest"], second["source_tree_digest"])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=python python3 -m unittest python.tests.test_corpus_provenance -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lape26.corpus.provenance'`

- [ ] **Step 3: Write minimal implementation**

Create `python/lape26/corpus/provenance.py`:

```python
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

NORMALIZATION_PROFILE_ID = "lape-text-normalization-v0.1"
NORMALIZATION_STATUS = "provisional-frozen"
NORMALIZATION_SPEC_PATH = "specs/text-normalization.md"
NORMALIZATION_IMPLEMENTATION_PATH = "python/lape26/normalize.py"

REPO_ROOT = Path(__file__).resolve().parents[3]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_git(args: list[str]) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def git_blob_sha(relative_path: str) -> str:
    return _run_git(["hash-object", relative_path])


def git_head_commit() -> str:
    return _run_git(["rev-parse", "HEAD"])


def git_is_dirty() -> bool:
    return bool(_run_git(["status", "--porcelain"]))


def source_tree_digest(paths: list[str]) -> str:
    hasher = hashlib.sha256()
    for relative_path in sorted(paths):
        hasher.update(relative_path.encode("utf-8"))
        hasher.update(sha256_file(REPO_ROOT / relative_path).encode("utf-8"))
    return hasher.hexdigest()


def build_provenance_block(*, pipeline_source_paths: list[str]) -> dict[str, object]:
    normalize_path = REPO_ROOT / NORMALIZATION_IMPLEMENTATION_PATH
    return {
        "normalization_profile_id": NORMALIZATION_PROFILE_ID,
        "normalization_status": NORMALIZATION_STATUS,
        "normalization_spec_path": NORMALIZATION_SPEC_PATH,
        "normalization_implementation": NORMALIZATION_IMPLEMENTATION_PATH,
        "normalization_implementation_sha256": sha256_file(normalize_path),
        "normalization_git_blob_sha": git_blob_sha(NORMALIZATION_IMPLEMENTATION_PATH),
        "pipeline_source_commit": git_head_commit(),
        "working_tree_dirty": git_is_dirty(),
        "source_tree_digest": source_tree_digest(pipeline_source_paths),
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=python python3 -m unittest python.tests.test_corpus_provenance -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add python/lape26/corpus/provenance.py python/tests/test_corpus_provenance.py
git commit -m "feat: add provenance block builder (checksums, git blob/commit, dirty state)"
```

---

## Task 9: Control mapping generators

**Files:**
- Create: `python/lape26/corpus/controls.py`
- Test: `python/tests/test_corpus_controls.py`

**Interfaces:**
- Consumes: `lape26.core.midi_to_frequency` (existing), `lape26.corpus.provenance.build_provenance_block` (Task 8).
- Produces:
  - `POOL_MIDI: list[int]` (`[48, 49, ..., 73]`), `POOL_LOWEST_MIDI = 48`, `POOL_HIGHEST_MIDI = 73`
  - `midi_to_pitch_name(midi: int) -> str` (e.g. `60 -> "C4"`, `69 -> "A4"`)
  - `generate_sequential_chromatic_control(pool_midi: list[int]) -> dict[str, int]`
  - `generate_frequency_ranked_control(letter_frequency: dict[str, int], pool_midi: list[int]) -> dict[str, int]`
  - `generate_circle_of_fifths_control(pool_midi: list[int]) -> dict[str, int]`
  - `generate_random_seed_control(pool_midi: list[int], seed: int) -> dict[str, int]`
  - `build_control_mapping_document(*, control_id: str, control_type: str, generation_method: str, assignment: dict[str, int], seed: int | None, source_partition: str | None, provenance: dict[str, object]) -> dict[str, object]` — assembles the full document validated by `data/schemas/control-mapping.schema.json` (Task 1).
- Consumed by: Task 14 (`build_corpus_pipeline.py`).

- [ ] **Step 1: Write the failing test**

Create `python/tests/test_corpus_controls.py`:

```python
from __future__ import annotations

import json
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator

from lape26.corpus.controls import (
    POOL_MIDI,
    build_control_mapping_document,
    generate_circle_of_fifths_control,
    generate_frequency_ranked_control,
    generate_random_seed_control,
    generate_sequential_chromatic_control,
    midi_to_pitch_name,
)
from lape26.corpus.provenance import build_provenance_block

ROOT = Path(__file__).resolve().parents[2]
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _assert_valid_control(test_case: unittest.TestCase, assignment: dict[str, int]) -> None:
    test_case.assertEqual(set(assignment), set(LETTERS))
    test_case.assertEqual(len(set(assignment.values())), 26)
    for midi in assignment.values():
        test_case.assertIn(midi, POOL_MIDI)


class MidiToPitchNameTests(unittest.TestCase):
    def test_known_values_match_canonical_mapping(self) -> None:
        self.assertEqual(midi_to_pitch_name(60), "C4")
        self.assertEqual(midi_to_pitch_name(69), "A4")
        self.assertEqual(midi_to_pitch_name(48), "C3")
        self.assertEqual(midi_to_pitch_name(73), "C#5")


class SequentialChromaticTests(unittest.TestCase):
    def test_ascends_alphabet_to_pitch(self) -> None:
        result = generate_sequential_chromatic_control(POOL_MIDI)
        _assert_valid_control(self, result)
        self.assertEqual(result["A"], 48)
        self.assertEqual(result["Z"], 73)


class FrequencyRankedTests(unittest.TestCase):
    def test_most_frequent_letter_gets_pitch_nearest_center(self) -> None:
        result = generate_frequency_ranked_control({"Z": 100}, POOL_MIDI)
        _assert_valid_control(self, result)
        self.assertEqual(result["Z"], 60)


class CircleOfFifthsTests(unittest.TestCase):
    def test_produces_valid_bijection(self) -> None:
        result = generate_circle_of_fifths_control(POOL_MIDI)
        _assert_valid_control(self, result)

    def test_first_two_letters_follow_fifths_where_slots_available(self) -> None:
        result = generate_circle_of_fifths_control(POOL_MIDI)
        self.assertEqual(result["A"], 48)
        self.assertEqual(result["B"], 55)


class RandomSeedControlTests(unittest.TestCase):
    def test_deterministic_given_same_seed(self) -> None:
        first = generate_random_seed_control(POOL_MIDI, seed=26)
        second = generate_random_seed_control(POOL_MIDI, seed=26)
        self.assertEqual(first, second)
        _assert_valid_control(self, first)

    def test_different_seed_differs(self) -> None:
        a = generate_random_seed_control(POOL_MIDI, seed=26)
        b = generate_random_seed_control(POOL_MIDI, seed=27)
        self.assertNotEqual(a, b)


class BuildControlMappingDocumentTests(unittest.TestCase):
    def test_document_validates_against_schema(self) -> None:
        schema = json.loads((ROOT / "data" / "schemas" / "control-mapping.schema.json").read_text(encoding="utf-8"))
        validator = Draft202012Validator(schema)
        assignment = generate_sequential_chromatic_control(POOL_MIDI)
        provenance = build_provenance_block(pipeline_source_paths=["python/lape26/corpus/controls.py"])
        document = build_control_mapping_document(
            control_id="sequential-chromatic-v0.1",
            control_type="sequential-chromatic",
            generation_method="alphabetical-ascending-chromatic",
            assignment=assignment,
            seed=None,
            source_partition=None,
            provenance=provenance,
        )
        validator.validate(document)

    def test_rejects_incomplete_assignment(self) -> None:
        with self.assertRaises(ValueError):
            build_control_mapping_document(
                control_id="broken",
                control_type="sequential-chromatic",
                generation_method="test",
                assignment={"A": 48},
                seed=None,
                source_partition=None,
                provenance={},
            )


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=python python3 -m unittest python.tests.test_corpus_controls -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lape26.corpus.controls'`

- [ ] **Step 3: Write minimal implementation**

Create `python/lape26/corpus/controls.py`:

```python
from __future__ import annotations

import random

from ..core import midi_to_frequency

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
POOL_LOWEST_MIDI = 48
POOL_HIGHEST_MIDI = 73
POOL_MIDI = list(range(POOL_LOWEST_MIDI, POOL_HIGHEST_MIDI + 1))

_PITCH_CLASS_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def midi_to_pitch_name(midi: int) -> str:
    octave = midi // 12 - 1
    name = _PITCH_CLASS_NAMES[midi % 12]
    return f"{name}{octave}"


def generate_sequential_chromatic_control(pool_midi: list[int]) -> dict[str, int]:
    ordered_pool = sorted(pool_midi)
    return dict(zip(LETTERS, ordered_pool))


def generate_frequency_ranked_control(
    letter_frequency: dict[str, int],
    pool_midi: list[int],
) -> dict[str, int]:
    ranked_letters = sorted(LETTERS, key=lambda letter: (-letter_frequency.get(letter, 0), letter))
    center = (min(pool_midi) + max(pool_midi)) / 2
    ranked_pitches = sorted(pool_midi, key=lambda midi: (abs(midi - center), midi))
    return dict(zip(ranked_letters, ranked_pitches))


def _circle_of_fifths_pitch_classes(letter_count: int) -> list[int]:
    return [(i * 7) % 12 for i in range(letter_count)]


def generate_circle_of_fifths_control(pool_midi: list[int]) -> dict[str, int]:
    """Advance pitch class by a perfect fifth (7 semitones) per letter,
    starting at C (pitch class 0). The fixed 26-note pool (2 octaves + 2
    semitones) does not contain 3 instances of every pitch class visited 3
    times by the 26-letter fifths cycle, so when a letter's ideal pitch
    class has no unused pool slot left, it takes the closest unused MIDI
    value instead (ties broken by lower value). This keeps the result a
    valid bijection onto the fixed pool while remaining fully deterministic.
    """
    pitch_classes = _circle_of_fifths_pitch_classes(len(pool_midi))
    available = set(pool_midi)
    assignment: dict[str, int] = {}
    for letter, pitch_class in zip(LETTERS, pitch_classes):
        candidates = sorted(midi for midi in available if midi % 12 == pitch_class)
        if candidates:
            chosen = candidates[0]
        else:
            reference = min(pool_midi) + pitch_class
            chosen = min(available, key=lambda midi: (abs(midi - reference), midi))
        assignment[letter] = chosen
        available.discard(chosen)
    return assignment


def generate_random_seed_control(pool_midi: list[int], seed: int) -> dict[str, int]:
    rng = random.Random(seed)
    shuffled = pool_midi[:]
    rng.shuffle(shuffled)
    return dict(zip(LETTERS, shuffled))


def build_control_mapping_document(
    *,
    control_id: str,
    control_type: str,
    generation_method: str,
    assignment: dict[str, int],
    seed: int | None,
    source_partition: str | None,
    provenance: dict[str, object],
) -> dict[str, object]:
    if set(assignment) != set(LETTERS):
        raise ValueError("Control assignment must cover exactly A-Z")
    if len(set(assignment.values())) != 26:
        raise ValueError("Control assignment must use 26 unique MIDI values")

    letters = {
        letter: {
            "pitch": midi_to_pitch_name(midi),
            "midi": midi,
            "frequencyHz": round(midi_to_frequency(midi), 6),
        }
        for letter, midi in assignment.items()
    }
    return {
        "$schema": "../../data/schemas/control-mapping.schema.json",
        "mappingId": control_id,
        "version": "0.1.0-experimental",
        "status": "experimental",
        "controlType": control_type,
        "generationMethod": generation_method,
        "seed": seed,
        "sourcePartition": source_partition,
        "alphabet": LETTERS,
        "tuning": {
            "system": "12-TET",
            "referencePitch": "A4",
            "referenceMidi": 69,
            "referenceFrequencyHz": 440.0,
            "frequencyFormula": "f(m)=440*2^((m-69)/12)",
        },
        "range": {
            "lowestPitch": midi_to_pitch_name(POOL_LOWEST_MIDI),
            "lowestMidi": POOL_LOWEST_MIDI,
            "highestPitch": midi_to_pitch_name(POOL_HIGHEST_MIDI),
            "highestMidi": POOL_HIGHEST_MIDI,
        },
        "letters": letters,
        "provenance": provenance,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=python python3 -m unittest python.tests.test_corpus_controls -v`
Expected: PASS (9 tests).

- [ ] **Step 5: Commit**

```bash
git add python/lape26/corpus/controls.py python/tests/test_corpus_controls.py
git commit -m "feat: add sequential/frequency-ranked/circle-of-fifths/random-seed control generators"
```

---

## Task 10: Corpus acquisition and lock file (network-free parts)

**Files:**
- Create: `python/lape26/corpus/acquire.py`
- Test: `python/tests/test_corpus_lock.py`

**Interfaces:**
- Produces:
  - `APPROVED_PACKAGES = ("gutenberg", "words", "wordnet", "opinion_lexicon", "vader_lexicon")`
  - `LockEntry` dataclass: `package_id: str, source_version: str, archive_sha256: str, installed_content_sha256: str, retrieved_at: str`
  - `download_approved_packages(download_dir: Path) -> None` — the **only** function in `corpus/` that imports `nltk`, and it imports it locally inside the function body so the module itself is importable without `nltk` installed.
  - `build_lock_entries(download_dir: Path, retrieved_at: str | None = None) -> list[LockEntry]`
  - `write_lock_file(entries: list[LockEntry], lock_path: Path) -> None`
  - `read_lock_file(lock_path: Path) -> list[LockEntry]`
  - `verify_lock(download_dir: Path, lock_path: Path) -> tuple[bool, str]`
- Consumed by: Task 13 (`scripts/setup_corpus.py`), Task 14 (`build_corpus_pipeline.py` calls `verify_lock` before doing anything else).

- [ ] **Step 1: Write the failing test**

Create `python/tests/test_corpus_lock.py`:

```python
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from lape26.corpus.acquire import (
    APPROVED_PACKAGES,
    build_lock_entries,
    read_lock_file,
    verify_lock,
    write_lock_file,
)


def _make_fake_download_dir(base: Path) -> Path:
    download_dir = base / "nltk_data"
    for package_id in APPROVED_PACKAGES:
        package_dir = download_dir / "corpora" / package_id
        package_dir.mkdir(parents=True, exist_ok=True)
        (package_dir / "sample.txt").write_text(f"fake contents for {package_id}", encoding="utf-8")
    return download_dir


class LockFileTests(unittest.TestCase):
    def test_build_lock_entries_covers_all_approved_packages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            download_dir = _make_fake_download_dir(Path(tmp))
            entries = build_lock_entries(download_dir, retrieved_at="2026-07-29")
            self.assertEqual({e.package_id for e in entries}, set(APPROVED_PACKAGES))
            for entry in entries:
                self.assertEqual(len(entry.installed_content_sha256), 64)
                self.assertEqual(entry.retrieved_at, "2026-07-29")

    def test_write_and_read_lock_file_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            download_dir = _make_fake_download_dir(Path(tmp))
            entries = build_lock_entries(download_dir, retrieved_at="2026-07-29")
            lock_path = Path(tmp) / "corpus-lock.json"
            write_lock_file(entries, lock_path)
            loaded = read_lock_file(lock_path)
            self.assertEqual(
                sorted(e.package_id for e in loaded),
                sorted(e.package_id for e in entries),
            )

    def test_verify_lock_passes_when_cache_matches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            download_dir = _make_fake_download_dir(Path(tmp))
            entries = build_lock_entries(download_dir, retrieved_at="2026-07-29")
            lock_path = Path(tmp) / "corpus-lock.json"
            write_lock_file(entries, lock_path)
            is_valid, message = verify_lock(download_dir, lock_path)
            self.assertTrue(is_valid, message)

    def test_verify_lock_fails_when_cache_content_changed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            download_dir = _make_fake_download_dir(Path(tmp))
            entries = build_lock_entries(download_dir, retrieved_at="2026-07-29")
            lock_path = Path(tmp) / "corpus-lock.json"
            write_lock_file(entries, lock_path)

            (download_dir / "corpora" / "gutenberg" / "sample.txt").write_text("changed!", encoding="utf-8")

            is_valid, message = verify_lock(download_dir, lock_path)
            self.assertFalse(is_valid)
            self.assertIn("gutenberg", message)
            self.assertIn("corpus-relock", message)

    def test_verify_lock_fails_when_lock_file_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            download_dir = _make_fake_download_dir(Path(tmp))
            is_valid, message = verify_lock(download_dir, Path(tmp) / "missing-lock.json")
            self.assertFalse(is_valid)
            self.assertIn("not found", message)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=python python3 -m unittest python.tests.test_corpus_lock -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lape26.corpus.acquire'`

- [ ] **Step 3: Write minimal implementation**

Create `python/lape26/corpus/acquire.py`:

```python
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

APPROVED_PACKAGES = ("gutenberg", "words", "wordnet", "opinion_lexicon", "vader_lexicon")


@dataclass(frozen=True)
class LockEntry:
    package_id: str
    source_version: str
    archive_sha256: str
    installed_content_sha256: str
    retrieved_at: str


def download_approved_packages(download_dir: Path) -> None:
    """Download exactly the 5 approved NLTK packages into download_dir.
    `nltk` is imported here, not at module scope, so this module (and the
    rest of the corpus pipeline) stays importable and testable without
    nltk installed or any network access unless this function is called.
    """
    import nltk

    download_dir.mkdir(parents=True, exist_ok=True)
    for package_id in APPROVED_PACKAGES:
        nltk.download(package_id, download_dir=str(download_dir), quiet=True)


def _hash_directory(path: Path) -> str:
    hasher = hashlib.sha256()
    for file_path in sorted(path.rglob("*")):
        if file_path.is_file():
            hasher.update(str(file_path.relative_to(path)).encode("utf-8"))
            hasher.update(file_path.read_bytes())
    return hasher.hexdigest()


def _find_package_dir(download_dir: Path, package_id: str) -> Path:
    directories = [candidate for candidate in download_dir.rglob(package_id) if candidate.is_dir()]
    if not directories:
        raise FileNotFoundError(f"Downloaded package directory not found for {package_id!r}")
    return directories[0]


def build_lock_entries(download_dir: Path, retrieved_at: str | None = None) -> list[LockEntry]:
    retrieved_at = retrieved_at or datetime.now(timezone.utc).date().isoformat()
    entries = []
    for package_id in APPROVED_PACKAGES:
        package_dir = _find_package_dir(download_dir, package_id)
        content_hash = _hash_directory(package_dir)
        entries.append(
            LockEntry(
                package_id=package_id,
                source_version="documented-or-unknown",
                archive_sha256=content_hash,
                installed_content_sha256=content_hash,
                retrieved_at=retrieved_at,
            )
        )
    return entries


def write_lock_file(entries: list[LockEntry], lock_path: Path) -> None:
    payload = {"packages": [asdict(entry) for entry in entries]}
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_lock_file(lock_path: Path) -> list[LockEntry]:
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    return [LockEntry(**entry) for entry in payload["packages"]]


def verify_lock(download_dir: Path, lock_path: Path) -> tuple[bool, str]:
    if not lock_path.exists():
        return False, f"Lock file not found at {lock_path}"

    locked_entries = {entry.package_id: entry for entry in read_lock_file(lock_path)}
    for package_id in APPROVED_PACKAGES:
        if package_id not in locked_entries:
            return False, f"Lock file missing entry for {package_id!r}"
        try:
            package_dir = _find_package_dir(download_dir, package_id)
        except FileNotFoundError as error:
            return False, str(error)
        current_hash = _hash_directory(package_dir)
        if current_hash != locked_entries[package_id].installed_content_sha256:
            return False, (
                f"Local cache for {package_id!r} does not match corpus-lock.json "
                f"(expected {locked_entries[package_id].installed_content_sha256}, "
                f"got {current_hash}). Run `make corpus-relock` if this is intentional."
            )
    return True, "Corpus lock verified"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=python python3 -m unittest python.tests.test_corpus_lock -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add python/lape26/corpus/acquire.py python/tests/test_corpus_lock.py
git commit -m "feat: add corpus lock file with NLTK acquisition isolated from module scope"
```

---

## Task 11: Pilot stimulus selection

**Files:**
- Create: `python/lape26/corpus/stimulus.py`
- Test: `python/tests/test_corpus_stimulus.py`

**Interfaces:**
- Produces:
  - `WordCandidate` dataclass: `word: str, length: int, partOfSpeech: str, polarity: str, vaderCompound: float, sourceDataset: str, stemKey: str` (linguistic work — lemmatization/stemming for `stemKey`, WordNet validation, Opinion Lexicon/VADER lookups — happens upstream in the adapter layer built in Task 14; this module treats `stemKey` as an opaque dedup key and never imports `nltk`).
  - `length_band(length: int) -> str | None` (`"short"` 3-5, `"medium"` 6-8, `"long"` 9-12, else `None`)
  - `select_core_set(candidates: list[WordCandidate], seed: int, per_cell: int = 12) -> list[WordCandidate]` — exactly `per_cell` per (length-band, polarity) cell, deduped by `stemKey`, round-robined across `partOfSpeech` for variety, raises `ValueError` naming the short cell if a cell has fewer than `per_cell` eligible candidates after dedup.
  - `has_repeated_letters(word: str) -> bool`, `has_rare_letters(word: str) -> bool` (J/Q/X/Z), `is_vowel_heavy(word: str) -> bool` (>=50% vowels), `is_consonant_heavy(word: str) -> bool` (>=80% consonants)
  - `OrthographicCandidate` dataclass: `word: str, category: str`
  - `select_orthographic_challenge_set(candidates_by_category: dict[str, list[str]], seed: int, per_category: int = 3) -> list[OrthographicCandidate]` — 3 per category (`repeated-letters`, `rare-letters`, `vowel-heavy`, `consonant-heavy`), raises `ValueError` naming the short category otherwise.
- Consumed by: Task 14 (`build_corpus_pipeline.py`).

- [ ] **Step 1: Write the failing test**

Create `python/tests/test_corpus_stimulus.py`:

```python
from __future__ import annotations

import unittest

from lape26.corpus.stimulus import (
    OrthographicCandidate,
    WordCandidate,
    has_rare_letters,
    has_repeated_letters,
    is_consonant_heavy,
    is_vowel_heavy,
    length_band,
    select_core_set,
    select_orthographic_challenge_set,
)

_LENGTH_BY_BAND = {"short": 4, "medium": 7, "long": 10}
_POS_CYCLE = ["noun", "verb", "adjective", "adverb"]
_BANDS = ("short", "medium", "long")
_POLARITIES = ("positive", "negative", "neutral")


def _make_candidates(band: str, polarity: str, count: int) -> list[WordCandidate]:
    return [
        WordCandidate(
            word=f"{band}{polarity}{i}",
            length=_LENGTH_BY_BAND[band],
            partOfSpeech=_POS_CYCLE[i % len(_POS_CYCLE)],
            polarity=polarity,
            vaderCompound=0.5 if polarity == "positive" else (-0.5 if polarity == "negative" else 0.0),
            sourceDataset="opinion-lexicon" if polarity != "neutral" else "words",
            stemKey=f"{band}{polarity}{i}",
        )
        for i in range(count)
    ]


def _full_candidate_pool(count_per_cell: int) -> list[WordCandidate]:
    candidates: list[WordCandidate] = []
    for band in _BANDS:
        for polarity in _POLARITIES:
            candidates.extend(_make_candidates(band, polarity, count_per_cell))
    return candidates


class LengthBandTests(unittest.TestCase):
    def test_bands(self) -> None:
        self.assertEqual(length_band(3), "short")
        self.assertEqual(length_band(5), "short")
        self.assertEqual(length_band(6), "medium")
        self.assertEqual(length_band(8), "medium")
        self.assertEqual(length_band(9), "long")
        self.assertEqual(length_band(12), "long")
        self.assertIsNone(length_band(2))
        self.assertIsNone(length_band(13))


class SelectCoreSetTests(unittest.TestCase):
    def test_selects_exactly_108_with_12_per_cell(self) -> None:
        candidates = _full_candidate_pool(count_per_cell=12)
        selected = select_core_set(candidates, seed=26)
        self.assertEqual(len(selected), 108)
        for band in _BANDS:
            for polarity in _POLARITIES:
                cell_count = sum(
                    1 for c in selected if length_band(c.length) == band and c.polarity == polarity
                )
                self.assertEqual(cell_count, 12, f"{band}/{polarity}")

    def test_selects_12_even_with_surplus_candidates(self) -> None:
        candidates = _full_candidate_pool(count_per_cell=20)
        selected = select_core_set(candidates, seed=26)
        self.assertEqual(len(selected), 108)

    def test_raises_when_a_cell_is_short(self) -> None:
        candidates = _full_candidate_pool(count_per_cell=12)
        candidates = [c for c in candidates if not (c.word.startswith("shortpositive") and c.word != "shortpositive0")]
        with self.assertRaises(ValueError) as ctx:
            select_core_set(candidates, seed=26)
        self.assertIn("short", str(ctx.exception))
        self.assertIn("positive", str(ctx.exception))

    def test_deterministic_given_same_seed(self) -> None:
        candidates = _full_candidate_pool(count_per_cell=15)
        first = select_core_set(candidates, seed=26)
        second = select_core_set(candidates, seed=26)
        self.assertEqual([c.word for c in first], [c.word for c in second])


class OrthographicClassificationTests(unittest.TestCase):
    def test_has_repeated_letters(self) -> None:
        self.assertTrue(has_repeated_letters("BALLOON"))
        self.assertFalse(has_repeated_letters("CAT"))

    def test_has_rare_letters(self) -> None:
        self.assertTrue(has_rare_letters("JAZZ"))
        self.assertTrue(has_rare_letters("QUIZ"))
        self.assertFalse(has_rare_letters("CAT"))

    def test_is_vowel_heavy(self) -> None:
        self.assertTrue(is_vowel_heavy("AI"))
        self.assertFalse(is_vowel_heavy("STRENGTH"))

    def test_is_consonant_heavy(self) -> None:
        self.assertTrue(is_consonant_heavy("STRENGTH"))
        self.assertFalse(is_consonant_heavy("AI"))


class SelectOrthographicChallengeSetTests(unittest.TestCase):
    def _pool(self) -> dict[str, list[str]]:
        return {
            "repeated-letters": ["BALLOON", "ANNOUNCE", "COMMITTEE", "MISSISSIPPI"],
            "rare-letters": ["JAZZ", "QUIZ", "WALTZ", "XYLOPHONE"],
            "vowel-heavy": ["AI", "AREA", "IDEA", "QUEUE"],
            "consonant-heavy": ["STRENGTH", "RHYTHM", "GLYPH", "NYMPH"],
        }

    def test_selects_3_per_category_12_total(self) -> None:
        result = select_orthographic_challenge_set(self._pool(), seed=26)
        self.assertEqual(len(result), 12)
        for category in ("repeated-letters", "rare-letters", "vowel-heavy", "consonant-heavy"):
            self.assertEqual(sum(1 for r in result if r.category == category), 3)

    def test_raises_when_category_is_short(self) -> None:
        pool = self._pool()
        pool["rare-letters"] = ["JAZZ"]
        with self.assertRaises(ValueError) as ctx:
            select_orthographic_challenge_set(pool, seed=26)
        self.assertIn("rare-letters", str(ctx.exception))

    def test_deterministic_given_same_seed(self) -> None:
        first = select_orthographic_challenge_set(self._pool(), seed=26)
        second = select_orthographic_challenge_set(self._pool(), seed=26)
        self.assertEqual([r.word for r in first], [r.word for r in second])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=python python3 -m unittest python.tests.test_corpus_stimulus -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lape26.corpus.stimulus'`

- [ ] **Step 3: Write minimal implementation**

Create `python/lape26/corpus/stimulus.py`:

```python
from __future__ import annotations

import random
from dataclasses import dataclass

_BANDS = ("short", "medium", "long")
_POLARITIES = ("positive", "negative", "neutral")
_VOWELS = set("AEIOU")
_RARE_LETTERS = set("JQXZ")
_ORTHOGRAPHIC_CATEGORIES = ("repeated-letters", "rare-letters", "vowel-heavy", "consonant-heavy")


@dataclass(frozen=True)
class WordCandidate:
    word: str
    length: int
    partOfSpeech: str
    polarity: str
    vaderCompound: float
    sourceDataset: str
    stemKey: str


def length_band(length: int) -> str | None:
    if 3 <= length <= 5:
        return "short"
    if 6 <= length <= 8:
        return "medium"
    if 9 <= length <= 12:
        return "long"
    return None


def select_core_set(
    candidates: list[WordCandidate],
    seed: int,
    per_cell: int = 12,
) -> list[WordCandidate]:
    by_cell: dict[tuple[str, str], list[WordCandidate]] = {
        (band, polarity): [] for band in _BANDS for polarity in _POLARITIES
    }
    for candidate in candidates:
        band = length_band(candidate.length)
        if band is None or candidate.polarity not in _POLARITIES:
            continue
        by_cell[(band, candidate.polarity)].append(candidate)

    selected: list[WordCandidate] = []
    for (band, polarity), pool in by_cell.items():
        cell_seed = seed + (hash((band, polarity)) % 1_000_003)
        selected.extend(_select_cell(pool, per_cell, cell_seed, band, polarity))
    return selected


def _select_cell(
    pool: list[WordCandidate],
    per_cell: int,
    seed: int,
    band: str,
    polarity: str,
) -> list[WordCandidate]:
    deduplicated: dict[str, WordCandidate] = {}
    for candidate in sorted(pool, key=lambda c: c.word):
        deduplicated.setdefault(candidate.stemKey, candidate)
    unique_candidates = list(deduplicated.values())

    rng = random.Random(seed)
    shuffled = unique_candidates[:]
    rng.shuffle(shuffled)

    by_pos: dict[str, list[WordCandidate]] = {}
    for candidate in shuffled:
        by_pos.setdefault(candidate.partOfSpeech, []).append(candidate)

    chosen: list[WordCandidate] = []
    pos_cycle = sorted(by_pos)
    pos_index = 0
    max_attempts = per_cell * max(len(pos_cycle), 1) + len(unique_candidates) + 1
    attempts = 0
    while len(chosen) < per_cell and attempts < max_attempts and pos_cycle:
        pos = pos_cycle[pos_index % len(pos_cycle)]
        if by_pos[pos]:
            chosen.append(by_pos[pos].pop(0))
        pos_index += 1
        attempts += 1

    if len(chosen) < per_cell:
        raise ValueError(
            f"Not enough eligible candidates for {band}/{polarity}: "
            f"need {per_cell}, found {len(chosen)} after dedup"
        )
    return sorted(chosen, key=lambda c: c.word)[:per_cell]


def has_repeated_letters(word: str) -> bool:
    upper = word.upper()
    return any(upper.count(letter) >= 2 for letter in set(upper))


def has_rare_letters(word: str) -> bool:
    upper = word.upper()
    return any(letter in _RARE_LETTERS for letter in upper)


def is_vowel_heavy(word: str) -> bool:
    upper = word.upper()
    if not upper:
        return False
    vowel_count = sum(1 for ch in upper if ch in _VOWELS)
    return vowel_count / len(upper) >= 0.5


def is_consonant_heavy(word: str) -> bool:
    upper = word.upper()
    if not upper:
        return False
    consonant_count = sum(1 for ch in upper if ch not in _VOWELS)
    return consonant_count / len(upper) >= 0.8


@dataclass(frozen=True)
class OrthographicCandidate:
    word: str
    category: str


def select_orthographic_challenge_set(
    candidates_by_category: dict[str, list[str]],
    seed: int,
    per_category: int = 3,
) -> list[OrthographicCandidate]:
    selected: list[OrthographicCandidate] = []
    for category in _ORTHOGRAPHIC_CATEGORIES:
        pool = sorted(set(candidates_by_category.get(category, [])))
        rng = random.Random(seed + (hash(category) % 1_000_003))
        shuffled = pool[:]
        rng.shuffle(shuffled)
        if len(shuffled) < per_category:
            raise ValueError(
                f"Not enough orthographic candidates for {category!r}: "
                f"need {per_category}, found {len(shuffled)}"
            )
        chosen = sorted(shuffled[:per_category])
        selected.extend(OrthographicCandidate(word=word, category=category) for word in chosen)
    return selected
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=python python3 -m unittest python.tests.test_corpus_stimulus -v`
Expected: PASS (12 tests).

- [ ] **Step 5: Commit**

```bash
git add python/lape26/corpus/stimulus.py python/tests/test_corpus_stimulus.py
git commit -m "feat: add deterministic pilot stimulus core-set and orthographic-challenge selection"
```

---

## Task 12: Baseline comparison report (descriptive metrics only)

**Files:**
- Create: `python/lape26/corpus/report.py`
- Test: `python/tests/test_corpus_report.py`

**Interfaces:**
- Consumes: `lape26.core.encode_text`, `lape26.analysis.summarize` (existing).
- Produces:
  - `METRIC_VERSIONS: dict[str, str]` — exactly `register_center_v0.1`, `pitch_span_v0.1`, `interval_contour_v0.1`, `directional_balance_v0.1`, `repetition_index_v0.1`.
  - `REPORT_STATEMENT: str` — the required descriptive-only disclosure sentence.
  - `summarize_word(word: str, mapping_path: str) -> dict[str, object]`
  - `aggregate_summaries(summaries: list[dict[str, object]]) -> dict[str, object]`
  - `build_baseline_comparison_report(*, mapping_paths: dict[str, str], stimulus_words_by_stratum: dict[str, list[str]], provenance: dict[str, object]) -> dict[str, object]` — validates against `data/schemas/baseline-comparison.schema.json` (Task 1).
- Consumed by: Task 14 (`build_corpus_pipeline.py`).

- [ ] **Step 1: Write the failing test**

Create `python/tests/test_corpus_report.py`:

```python
from __future__ import annotations

import json
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator

from lape26.corpus.provenance import build_provenance_block
from lape26.corpus.report import (
    METRIC_VERSIONS,
    REPORT_STATEMENT,
    aggregate_summaries,
    build_baseline_comparison_report,
    summarize_word,
)
from lape26.core import DEFAULT_MAPPING_PATH

ROOT = Path(__file__).resolve().parents[2]


class SummarizeWordTests(unittest.TestCase):
    def test_summarize_hammer_matches_golden_vector(self) -> None:
        summary = summarize_word("HAMMER", str(DEFAULT_MAPPING_PATH))
        self.assertEqual(summary["intervals"], [-9, 3, 0, 1, -7])


class AggregateSummariesTests(unittest.TestCase):
    def test_interval_movement_counts(self) -> None:
        summaries = [
            summarize_word("HAMMER", str(DEFAULT_MAPPING_PATH)),
            summarize_word("MUSIC", str(DEFAULT_MAPPING_PATH)),
        ]
        aggregate = aggregate_summaries(summaries)
        self.assertEqual(aggregate["itemCount"], 2)
        total_intervals = len(summaries[0]["intervals"]) + len(summaries[1]["intervals"])
        movement = aggregate["intervalMovement"]
        self.assertEqual(movement["upward"] + movement["downward"] + movement["repeated"], total_intervals)

    def test_empty_summaries_do_not_crash(self) -> None:
        aggregate = aggregate_summaries([])
        self.assertEqual(aggregate["itemCount"], 0)
        self.assertEqual(aggregate["registerCenterMidi"], {"mean": 0.0, "min": 0.0, "max": 0.0})


class BuildBaselineComparisonReportTests(unittest.TestCase):
    def test_report_validates_against_schema_and_states_boundary(self) -> None:
        schema = json.loads((ROOT / "data" / "schemas" / "baseline-comparison.schema.json").read_text(encoding="utf-8"))
        validator = Draft202012Validator(schema)
        provenance = build_provenance_block(pipeline_source_paths=["python/lape26/corpus/report.py"])
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=python python3 -m unittest python.tests.test_corpus_report -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'lape26.corpus.report'`

- [ ] **Step 3: Write minimal implementation**

Create `python/lape26/corpus/report.py`:

```python
from __future__ import annotations

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

    return {
        "itemCount": len(summaries),
        "registerCenterMidi": _distribution(register_centers),
        "pitchSpanSemitones": _distribution(pitch_spans),
        "directionalBalance": _distribution(directional_balances),
        "repetitionIndex": _distribution(repetition_indices),
        "intervalMovement": {
            "upward": sum(1 for i in all_intervals if i > 0),
            "downward": sum(1 for i in all_intervals if i < 0),
            "repeated": sum(1 for i in all_intervals if i == 0),
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=python python3 -m unittest python.tests.test_corpus_report -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add python/lape26/corpus/report.py python/tests/test_corpus_report.py
git commit -m "feat: add descriptive-only baseline comparison report generator"
```

---

## Task 13: `scripts/setup_corpus.py` CLI (corpus-setup / corpus-lock / corpus-relock / verify)

**Files:**
- Create: `scripts/setup_corpus.py`
- Test: `python/tests/test_setup_corpus_cli.py`

**Interfaces:**
- Consumes: `lape26.corpus.acquire.{download_approved_packages, build_lock_entries, write_lock_file, verify_lock}` (Task 10).
- Produces: `run_setup(download_dir, lock_path) -> None`, `run_lock(download_dir, lock_path) -> None`, `run_verify(download_dir, lock_path) -> bool` — importable and directly testable without going through `argparse`/network. `main()` wires these to the `setup`/`lock`/`relock`/`verify` subcommands with real default paths (`data/raw/nltk_data/`, `data/manifests/corpus-lock.json`).
- Consumed by: `Makefile` targets `corpus-setup`, `corpus-lock`, `corpus-relock` (Task 17); `run_verify` is reused by `scripts/build_corpus_pipeline.py` (Task 14) to refuse to proceed on lock mismatch.

- [ ] **Step 1: Write the failing test**

Create `python/tests/test_setup_corpus_cli.py`:

```python
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "python"))

from setup_corpus import run_lock, run_verify  # noqa: E402
from lape26.corpus.acquire import APPROVED_PACKAGES  # noqa: E402


def _make_fake_download_dir(base: Path) -> Path:
    download_dir = base / "nltk_data"
    for package_id in APPROVED_PACKAGES:
        package_dir = download_dir / "corpora" / package_id
        package_dir.mkdir(parents=True, exist_ok=True)
        (package_dir / "sample.txt").write_text(f"fake contents for {package_id}", encoding="utf-8")
    return download_dir


class SetupCorpusCliTests(unittest.TestCase):
    def test_lock_then_verify_succeeds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            download_dir = _make_fake_download_dir(Path(tmp))
            lock_path = Path(tmp) / "corpus-lock.json"
            run_lock(download_dir, lock_path)
            self.assertTrue(lock_path.exists())
            self.assertTrue(run_verify(download_dir, lock_path))

    def test_verify_fails_without_lock(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            download_dir = _make_fake_download_dir(Path(tmp))
            self.assertFalse(run_verify(download_dir, Path(tmp) / "missing.json"))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=python python3 -m unittest python.tests.test_setup_corpus_cli -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'setup_corpus'`

- [ ] **Step 3: Write minimal implementation**

Create `scripts/setup_corpus.py`:

```python
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from lape26.corpus.acquire import (
    build_lock_entries,
    download_approved_packages,
    verify_lock,
    write_lock_file,
)

DEFAULT_DOWNLOAD_DIR = ROOT / "data" / "raw" / "nltk_data"
DEFAULT_LOCK_PATH = ROOT / "data" / "manifests" / "corpus-lock.json"


def run_setup(download_dir: Path, lock_path: Path) -> None:
    download_approved_packages(download_dir)
    entries = build_lock_entries(download_dir)
    write_lock_file(entries, lock_path)
    print(f"Downloaded and locked {len(entries)} approved corpora to {download_dir}")


def run_lock(download_dir: Path, lock_path: Path) -> None:
    entries = build_lock_entries(download_dir)
    write_lock_file(entries, lock_path)
    print(f"Wrote {lock_path}")


def run_verify(download_dir: Path, lock_path: Path) -> bool:
    is_valid, message = verify_lock(download_dir, lock_path)
    print(message)
    return is_valid


def main() -> None:
    parser = argparse.ArgumentParser(description="Corpus setup/lock tooling (local only, never run in CI)")
    parser.add_argument("command", choices=["setup", "lock", "relock", "verify"])
    parser.add_argument("--download-dir", type=Path, default=DEFAULT_DOWNLOAD_DIR)
    parser.add_argument("--lock-path", type=Path, default=DEFAULT_LOCK_PATH)
    args = parser.parse_args()

    if args.command == "setup":
        run_setup(args.download_dir, args.lock_path)
    elif args.command in ("lock", "relock"):
        run_lock(args.download_dir, args.lock_path)
    elif args.command == "verify":
        if not run_verify(args.download_dir, args.lock_path):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=python python3 -m unittest python.tests.test_setup_corpus_cli -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/setup_corpus.py python/tests/test_setup_corpus_cli.py
git commit -m "feat: add corpus-setup/lock/relock/verify CLI"
```

---

## Task 14: `scripts/build_corpus_pipeline.py` — offline orchestration + NLTK-backed loaders

**Files:**
- Create: `scripts/build_corpus_pipeline.py`
- Test: `python/tests/test_build_corpus_pipeline.py`

**Interfaces:**
- Consumes: everything from Tasks 5–13 (`tokens`, `stats`, `splits`, `provenance`, `controls`, `stimulus`, `report`, `acquire.verify_lock`, `setup_corpus.DEFAULT_DOWNLOAD_DIR/DEFAULT_LOCK_PATH`).
- Produces:
  - `run_pipeline(*, gutenberg_sentences_by_document: dict[str, list[list[str]]], word_candidates: list[WordCandidate], orthographic_candidates_by_category: dict[str, list[str]], seed: int, processed_dir: Path, fixtures_dir: Path, controls_dir: Path) -> dict[str, Path]` — **NLTK-free**, this is the piece tested here and reused (with a tiny fixture) in Task 20's offline test. Writes `corpus-statistics-v0.1.json`, `corpus-splits-v0.1.json`, the 4 control mapping files, `pilot-stimulus-v0.1.json`, `baseline-comparison-v0.1.json`, `baseline-comparison-v0.1.md`, and `artifact-index-v0.1.json`.
  - `load_gutenberg_sentences_by_document() -> dict[str, list[list[str]]]`, `load_word_candidates() -> list[WordCandidate]`, `load_orthographic_candidates() -> dict[str, list[str]]` — the only functions here that import `nltk` (lazily, inside each function body), exercised for real only in Task 21's local run against the actual downloaded corpora.
  - `main()` — verifies the lock (Task 10/13), loads real data via the three loaders above, calls `run_pipeline` with `seed=26` and the real repo output directories.
- Consumed by: `Makefile` target `corpus-pipeline` (Task 17); `run_pipeline` is reused by `scripts/check_corpus_pipeline.py` (Task 15).

**Important limitation to flag explicitly in code and in this task:** `load_word_candidates` filters on WordNet validity, `.isalpha()`, and Opinion Lexicon/VADER polarity, but has **no automated slur/distressing-term filter** — that requires human judgment, not a programmatic check. Task 21 (generating the real committed fixture) includes a mandatory manual review step of the 120 selected words before they're committed, per the spec's exclusion rules (proper nouns, abbreviations, duplicates, morphological variants, highly obscure words, slurs, distressing terms).

- [ ] **Step 1: Write the failing test**

Create `python/tests/test_build_corpus_pipeline.py`:

```python
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


def _synthetic_word_candidates(count_per_cell: int = 15) -> list[WordCandidate]:
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


class RunPipelineTests(unittest.TestCase):
    def test_writes_all_expected_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            written = run_pipeline(
                gutenberg_sentences_by_document=_synthetic_gutenberg_documents(),
                word_candidates=_synthetic_word_candidates(),
                orthographic_candidates_by_category=_synthetic_orthographic_candidates(),
                seed=26,
                processed_dir=base / "processed",
                fixtures_dir=base / "fixtures",
                controls_dir=base / "controls",
            )
            expected_ids = {
                "corpus-statistics-v0.1",
                "corpus-splits-v0.1",
                "sequential-chromatic-v0.1",
                "frequency-ranked-v0.1",
                "circle-of-fifths-v0.1",
                "random-seed-026-v0.1",
                "pilot-stimulus-v0.1",
                "baseline-comparison-v0.1-json",
                "baseline-comparison-v0.1-md",
                "artifact-index-v0.1",
            }
            self.assertEqual(set(written), expected_ids)
            for path in written.values():
                self.assertTrue(path.exists(), f"missing {path}")

    def test_stimulus_fixture_has_108_core_and_12_orthographic(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            written = run_pipeline(
                gutenberg_sentences_by_document=_synthetic_gutenberg_documents(),
                word_candidates=_synthetic_word_candidates(),
                orthographic_candidates_by_category=_synthetic_orthographic_candidates(),
                seed=26,
                processed_dir=base / "processed",
                fixtures_dir=base / "fixtures",
                controls_dir=base / "controls",
            )
            stimulus = json.loads(written["pilot-stimulus-v0.1"].read_text(encoding="utf-8"))
            self.assertEqual(len(stimulus["coreSet"]), 108)
            self.assertEqual(len(stimulus["orthographicChallengeSet"]), 12)

    def test_deterministic_given_same_seed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            kwargs = dict(
                gutenberg_sentences_by_document=_synthetic_gutenberg_documents(),
                word_candidates=_synthetic_word_candidates(),
                orthographic_candidates_by_category=_synthetic_orthographic_candidates(),
                seed=26,
            )
            first = run_pipeline(**kwargs, processed_dir=base / "a" / "processed", fixtures_dir=base / "a" / "fixtures", controls_dir=base / "a" / "controls")
            second = run_pipeline(**kwargs, processed_dir=base / "b" / "processed", fixtures_dir=base / "b" / "fixtures", controls_dir=base / "b" / "controls")
            first_stimulus = json.loads(first["pilot-stimulus-v0.1"].read_text(encoding="utf-8"))
            second_stimulus = json.loads(second["pilot-stimulus-v0.1"].read_text(encoding="utf-8"))
            self.assertEqual(
                [item["word"] for item in first_stimulus["coreSet"]],
                [item["word"] for item in second_stimulus["coreSet"]],
            )

    def test_frequency_ranked_control_source_partition_is_gutenberg_train(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            written = run_pipeline(
                gutenberg_sentences_by_document=_synthetic_gutenberg_documents(),
                word_candidates=_synthetic_word_candidates(),
                orthographic_candidates_by_category=_synthetic_orthographic_candidates(),
                seed=26,
                processed_dir=base / "processed",
                fixtures_dir=base / "fixtures",
                controls_dir=base / "controls",
            )
            control = json.loads(written["frequency-ranked-v0.1"].read_text(encoding="utf-8"))
            self.assertEqual(control["sourcePartition"], "gutenberg-train")

    def test_baseline_report_declares_only_implemented_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            written = run_pipeline(
                gutenberg_sentences_by_document=_synthetic_gutenberg_documents(),
                word_candidates=_synthetic_word_candidates(),
                orthographic_candidates_by_category=_synthetic_orthographic_candidates(),
                seed=26,
                processed_dir=base / "processed",
                fixtures_dir=base / "fixtures",
                controls_dir=base / "controls",
            )
            report = json.loads(written["baseline-comparison-v0.1-json"].read_text(encoding="utf-8"))
            self.assertEqual(set(report["metricVersions"]), {
                "register_center", "pitch_span", "interval_contour",
                "directional_balance", "repetition_index",
            })
            self.assertEqual(set(report["mappingIds"]), {
                "lape-26-en-general-v0.1", "sequential-chromatic-v0.1",
                "frequency-ranked-v0.1", "circle-of-fifths-v0.1", "random-seed-026-v0.1",
            })


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=python python3 -m unittest python.tests.test_build_corpus_pipeline -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'build_corpus_pipeline'`

- [ ] **Step 3: Write minimal implementation**

Create `scripts/build_corpus_pipeline.py`:

```python
#!/usr/bin/env python3
from __future__ import annotations

import json
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
from lape26.corpus.provenance import REPO_ROOT, build_provenance_block, sha256_file
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
    "python/lape26/corpus/tokens.py",
    "python/lape26/corpus/stats.py",
    "python/lape26/corpus/splits.py",
    "python/lape26/corpus/controls.py",
    "python/lape26/corpus/stimulus.py",
    "python/lape26/corpus/report.py",
    "python/lape26/corpus/provenance.py",
]

_POS_NAMES = {"n": "noun", "v": "verb", "a": "adjective", "s": "adjective", "r": "adverb"}


def run_pipeline(
    *,
    gutenberg_sentences_by_document: dict[str, list[list[str]]],
    word_candidates: list[WordCandidate],
    orthographic_candidates_by_category: dict[str, list[str]],
    seed: int,
    processed_dir: Path,
    fixtures_dir: Path,
    controls_dir: Path,
) -> dict[str, Path]:
    provenance = build_provenance_block(pipeline_source_paths=sorted(PIPELINE_SOURCE_PATHS))

    tokenized_by_document = {
        document_id: tokenize_sentences(raw_sentences)
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
        sentence
        for document_id in gutenberg_split.train
        for sentence in tokenized_by_document[document_id]
    ]
    all_sentences = [sentence for sentences in tokenized_by_document.values() for sentence in sentences]

    corpus_statistics = compute_statistics(all_sentences)
    train_statistics = compute_statistics(train_sentences)

    processed_dir.mkdir(parents=True, exist_ok=True)
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    controls_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    statistics_path = processed_dir / "corpus-statistics-v0.1.json"
    _write_json(statistics_path, {
        "artifactId": "corpus-statistics-v0.1",
        "artifactVersion": "0.1.0",
        **asdict(corpus_statistics),
        "provenance": provenance,
    })
    written["corpus-statistics-v0.1"] = statistics_path

    splits_path = processed_dir / "corpus-splits-v0.1.json"
    _write_json(splits_path, {
        "artifactId": "corpus-splits-v0.1",
        "artifactVersion": "0.1.0",
        "wordListSplit": asdict(word_split),
        "gutenbergSplit": asdict(gutenberg_split),
        "provenance": provenance,
    })
    written["corpus-splits-v0.1"] = splits_path

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

    train_word_set = set(word_split.train)
    training_candidates = [c for c in word_candidates if c.word.upper() in train_word_set]
    core_set = select_core_set(training_candidates, seed=seed)
    orthographic_set = select_orthographic_challenge_set(orthographic_candidates_by_category, seed=seed)

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
    stimulus_path = fixtures_dir / "pilot-stimulus-v0.1.json"
    _write_json(stimulus_path, {
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
    })
    written["pilot-stimulus-v0.1"] = stimulus_path

    strata: dict[str, list[str]] = {}
    for item in core_set_items:
        strata.setdefault(f"{item['lengthBand']}_{item['polarity']}", []).append(item["word"])
    strata["orthographic_challenge"] = [o.word for o in orthographic_set]

    mapping_paths = {"lape-26-en-general-v0.1": str(DEFAULT_MAPPING_PATH)}
    mapping_paths.update({control_id: str(path) for control_id, path in control_paths.items()})

    report = build_baseline_comparison_report(
        mapping_paths=mapping_paths,
        stimulus_words_by_stratum=strata,
        provenance=provenance,
    )
    report_json_path = processed_dir / "baseline-comparison-v0.1.json"
    _write_json(report_json_path, report)
    written["baseline-comparison-v0.1-json"] = report_json_path

    report_md_path = processed_dir / "baseline-comparison-v0.1.md"
    report_md_path.write_text(_render_report_markdown(report), encoding="utf-8")
    written["baseline-comparison-v0.1-md"] = report_md_path

    index_path = processed_dir / "artifact-index-v0.1.json"
    _write_json(index_path, _build_artifact_index(written, seed, provenance))
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
        "## Metric versions",
        "",
    ]
    for name, version in report["metricVersions"].items():
        lines.append(f"- `{name}`: `{version}`")
    lines.append("")
    lines.append("## Results by mapping and stratum")
    for mapping_id, by_stratum in report["results"].items():
        lines.append("")
        lines.append(f"### {mapping_id}")
        lines.append("")
        lines.append("| Stratum | Items | Register mean | Pitch span mean | Directional balance mean | Repetition mean |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for stratum_name, aggregate in by_stratum.items():
            lines.append(
                f"| {stratum_name} | {aggregate['itemCount']} "
                f"| {aggregate['registerCenterMidi']['mean']:.2f} "
                f"| {aggregate['pitchSpanSemitones']['mean']:.2f} "
                f"| {aggregate['directionalBalance']['mean']:.3f} "
                f"| {aggregate['repetitionIndex']['mean']:.3f} |"
            )
    return "\n".join(lines) + "\n"


def _relative_path_str(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _build_artifact_index(written: dict[str, Path], seed: int, provenance: dict[str, object]) -> dict[str, object]:
    artifacts = []
    for artifact_id, path in sorted(written.items()):
        artifacts.append({
            "artifactId": artifact_id,
            "artifactVersion": "0.1.0",
            "relativePath": _relative_path_str(path),
            "sha256": sha256_file(path),
            "pipelineVersion": "corpus-pipeline-v0.1",
            "normalizationProfile": provenance["normalization_profile_id"],
            "seed": seed,
            "sourceLockChecksum": provenance["source_tree_digest"],
        })
    return {"artifactIndexVersion": "artifact-index-v0.1", "artifacts": artifacts}


def load_gutenberg_sentences_by_document() -> dict[str, list[list[str]]]:
    from nltk.corpus import gutenberg

    return {
        fileid: [list(sentence) for sentence in gutenberg.sents(fileid)]
        for fileid in gutenberg.fileids()
    }


def load_word_candidates() -> list[WordCandidate]:
    from nltk.corpus import opinion_lexicon, wordnet
    from nltk.corpus import words as words_corpus
    from nltk.sentiment import SentimentIntensityAnalyzer

    positive_words = set(opinion_lexicon.positive())
    negative_words = set(opinion_lexicon.negative())
    analyzer = SentimentIntensityAnalyzer()

    candidates: list[WordCandidate] = []
    seen_words: set[str] = set()
    for raw_word in words_corpus.words():
        word = raw_word.lower()
        if not word.isalpha() or word in seen_words:
            continue
        seen_words.add(word)

        synsets = wordnet.synsets(word)
        if not synsets:
            continue

        part_of_speech = _POS_NAMES.get(synsets[0].pos(), "other")
        stem_key = wordnet.morphy(word) or word
        compound = analyzer.polarity_scores(word)["compound"]

        if word in positive_words:
            polarity = "positive"
        elif word in negative_words:
            polarity = "negative"
        elif abs(compound) < 0.05:
            polarity = "neutral"
        else:
            continue

        candidates.append(
            WordCandidate(
                word=word.upper(),
                length=len(word),
                partOfSpeech=part_of_speech,
                polarity=polarity,
                vaderCompound=compound,
                sourceDataset="opinion-lexicon+vader" if polarity != "neutral" else "words+wordnet+vader",
                stemKey=stem_key,
            )
        )
    return candidates


def load_orthographic_candidates() -> dict[str, list[str]]:
    from nltk.corpus import words as words_corpus

    from lape26.corpus.stimulus import (
        has_rare_letters,
        has_repeated_letters,
        is_consonant_heavy,
        is_vowel_heavy,
    )

    candidates: dict[str, list[str]] = {
        "repeated-letters": [], "rare-letters": [], "vowel-heavy": [], "consonant-heavy": [],
    }
    seen: set[str] = set()
    for raw_word in words_corpus.words():
        word = raw_word.lower()
        if not word.isalpha() or len(word) < 3 or word in seen:
            continue
        seen.add(word)
        if has_repeated_letters(word):
            candidates["repeated-letters"].append(word.upper())
        if has_rare_letters(word):
            candidates["rare-letters"].append(word.upper())
        if is_vowel_heavy(word):
            candidates["vowel-heavy"].append(word.upper())
        if is_consonant_heavy(word):
            candidates["consonant-heavy"].append(word.upper())
    return candidates


def main() -> None:
    from setup_corpus import DEFAULT_DOWNLOAD_DIR, DEFAULT_LOCK_PATH

    is_valid, message = verify_lock(DEFAULT_DOWNLOAD_DIR, DEFAULT_LOCK_PATH)
    print(message)
    if not is_valid:
        raise SystemExit(1)

    written = run_pipeline(
        gutenberg_sentences_by_document=load_gutenberg_sentences_by_document(),
        word_candidates=load_word_candidates(),
        orthographic_candidates_by_category=load_orthographic_candidates(),
        seed=26,
        processed_dir=REPO_ROOT / "data" / "processed" / "corpus",
        fixtures_dir=REPO_ROOT / "data" / "fixtures",
        controls_dir=REPO_ROOT / "mappings" / "controls",
    )
    for artifact_id, path in sorted(written.items()):
        print(f"Wrote {artifact_id}: {path}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=python python3 -m unittest python.tests.test_build_corpus_pipeline -v`
Expected: PASS (5 tests). Note: this exercises `run_pipeline` only — `load_gutenberg_sentences_by_document`, `load_word_candidates`, `load_orthographic_candidates`, and `main()` require real NLTK data and are verified for real in Task 21.

- [ ] **Step 5: Commit**

```bash
git add scripts/build_corpus_pipeline.py python/tests/test_build_corpus_pipeline.py
git commit -m "feat: add corpus-pipeline orchestration (offline-testable) and NLTK-backed loaders"
```

---

## Task 15: `scripts/check_corpus_pipeline.py` (corpus-check, local-only)

**Files:**
- Create: `scripts/check_corpus_pipeline.py`
- Test: `python/tests/test_check_corpus_pipeline.py`

**Interfaces:**
- Consumes: `run_pipeline` and the three loaders from Task 14.
- Produces: `compare_artifact(committed_path: Path, regenerated_path: Path) -> str | None` — returns a mismatch description or `None` if equivalent. For JSON artifacts, compares content with `provenance.pipeline_source_commit` and `provenance.working_tree_dirty` excluded (those legitimately differ run-to-run); for `.md` it's a plain text compare. `check(tmp_root: Path) -> list[str]` — regenerates into `tmp_root` and returns all mismatches (empty list = clean). `main()` — runs `check()` against a real temp dir and exits non-zero with a printed report if anything differs.
- **Not run in CI** (per design decision) — local-only maintainer safeguard, invoked by `make corpus-check`.

- [ ] **Step 1: Write the failing test**

Create `python/tests/test_check_corpus_pipeline.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=python python3 -m unittest python.tests.test_check_corpus_pipeline -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'check_corpus_pipeline'`

- [ ] **Step 3: Write minimal implementation**

Create `scripts/check_corpus_pipeline.py`:

```python
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
    from build_corpus_pipeline import (
        load_gutenberg_sentences_by_document,
        load_orthographic_candidates,
        load_word_candidates,
        run_pipeline,
    )
    from lape26.corpus.acquire import verify_lock
    from setup_corpus import DEFAULT_DOWNLOAD_DIR, DEFAULT_LOCK_PATH

    is_valid, message = verify_lock(DEFAULT_DOWNLOAD_DIR, DEFAULT_LOCK_PATH)
    if not is_valid:
        raise SystemExit(f"corpus-check aborted: {message}")

    regenerated = run_pipeline(
        gutenberg_sentences_by_document=load_gutenberg_sentences_by_document(),
        word_candidates=load_word_candidates(),
        orthographic_candidates_by_category=load_orthographic_candidates(),
        seed=26,
        processed_dir=tmp_root / "processed",
        fixtures_dir=tmp_root / "fixtures",
        controls_dir=tmp_root / "controls",
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=python python3 -m unittest python.tests.test_check_corpus_pipeline -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/check_corpus_pipeline.py python/tests/test_check_corpus_pipeline.py
git commit -m "feat: add local-only corpus-check regeneration diff (provenance-aware)"
```

---

## Task 16: `scripts/check_corpus_provenance.py` (CI-wired, network-free)

**Files:**
- Create: `scripts/check_corpus_provenance.py`
- Test: `python/tests/test_check_corpus_provenance.py`

**Interfaces:**
- Consumes: `lape26.corpus.provenance` constants/`sha256_file` (Task 8), `jsonschema`, `data/schemas/*.json` (Task 1).
- Produces: `check_schema_validation(root: Path, artifacts: dict[Path, str], errors: list[str]) -> None`, `check_provenance(root: Path, artifact_paths: list[Path], errors: list[str]) -> None`, `check_corpus_lock_structure(root: Path, errors: list[str]) -> None`, `check_artifact_index(root: Path, errors: list[str]) -> None`, `main()`. Every `check_*` function is parameterized by `root: Path` so tests can point it at a fake fixture tree instead of the real repo.
- **This is the script CI actually runs** (Task 18) — no network, only reads committed files.

- [ ] **Step 1: Write the failing test**

Create `python/tests/test_check_corpus_provenance.py`:

```python
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
    sha256_file,
)


def _make_fake_root(base: Path) -> Path:
    (base / "python" / "lape26").mkdir(parents=True)
    (base / "python" / "lape26" / "normalize.py").write_text("def normalize_text(t): return t\n", encoding="utf-8")
    (base / "data" / "manifests").mkdir(parents=True)
    (base / "data" / "processed" / "corpus").mkdir(parents=True)
    return base


class CheckProvenanceTests(unittest.TestCase):
    def test_passes_when_provenance_matches_live_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _make_fake_root(Path(tmp))
            live_sha = sha256_file(root / NORMALIZATION_IMPLEMENTATION_PATH)
            artifact_path = root / "data" / "processed" / "corpus" / "artifact.json"
            artifact_path.write_text(json.dumps({
                "provenance": {
                    "normalization_profile_id": NORMALIZATION_PROFILE_ID,
                    "normalization_status": NORMALIZATION_STATUS,
                    "normalization_spec_path": NORMALIZATION_SPEC_PATH,
                    "normalization_implementation": NORMALIZATION_IMPLEMENTATION_PATH,
                    "normalization_implementation_sha256": live_sha,
                    "working_tree_dirty": False,
                }
            }), encoding="utf-8")
            errors: list[str] = []
            check_provenance(root, [artifact_path], errors)
            self.assertEqual(errors, [])

    def test_fails_when_checksum_stale(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _make_fake_root(Path(tmp))
            artifact_path = root / "data" / "processed" / "corpus" / "artifact.json"
            artifact_path.write_text(json.dumps({
                "provenance": {
                    "normalization_profile_id": NORMALIZATION_PROFILE_ID,
                    "normalization_status": NORMALIZATION_STATUS,
                    "normalization_spec_path": NORMALIZATION_SPEC_PATH,
                    "normalization_implementation": NORMALIZATION_IMPLEMENTATION_PATH,
                    "normalization_implementation_sha256": "stale-checksum",
                    "working_tree_dirty": False,
                }
            }), encoding="utf-8")
            errors: list[str] = []
            check_provenance(root, [artifact_path], errors)
            self.assertEqual(len(errors), 1)
            self.assertIn("does not match", errors[0])

    def test_fails_when_working_tree_dirty_true(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _make_fake_root(Path(tmp))
            live_sha = sha256_file(root / NORMALIZATION_IMPLEMENTATION_PATH)
            artifact_path = root / "data" / "processed" / "corpus" / "artifact.json"
            artifact_path.write_text(json.dumps({
                "provenance": {
                    "normalization_profile_id": NORMALIZATION_PROFILE_ID,
                    "normalization_status": NORMALIZATION_STATUS,
                    "normalization_spec_path": NORMALIZATION_SPEC_PATH,
                    "normalization_implementation": NORMALIZATION_IMPLEMENTATION_PATH,
                    "normalization_implementation_sha256": live_sha,
                    "working_tree_dirty": True,
                }
            }), encoding="utf-8")
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
                        "package_id": pkg,
                        "source_version": "x",
                        "archive_sha256": "a" * 64,
                        "installed_content_sha256": "a" * 64,
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
                "packages": [
                    {
                        "package_id": "gutenberg",
                        "source_version": "x",
                        "archive_sha256": "a" * 64,
                        "installed_content_sha256": "a" * 64,
                        "retrieved_at": "2026-07-29",
                    }
                ]
            }), encoding="utf-8")
            errors: list[str] = []
            check_corpus_lock_structure(root, errors)
            self.assertEqual(len(errors), 1)


class CheckArtifactIndexTests(unittest.TestCase):
    def test_passes_when_checksums_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _make_fake_root(Path(tmp))
            artifact_path = root / "data" / "processed" / "corpus" / "thing.json"
            artifact_path.write_text('{"x": 1}', encoding="utf-8")
            index_path = root / "data" / "processed" / "corpus" / "artifact-index-v0.1.json"
            index_path.write_text(json.dumps({
                "artifacts": [{
                    "relativePath": "data/processed/corpus/thing.json",
                    "sha256": sha256_file(artifact_path),
                }]
            }), encoding="utf-8")
            errors: list[str] = []
            check_artifact_index(root, errors)
            self.assertEqual(errors, [])

    def test_fails_when_checksum_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = _make_fake_root(Path(tmp))
            artifact_path = root / "data" / "processed" / "corpus" / "thing.json"
            artifact_path.write_text('{"x": 1}', encoding="utf-8")
            index_path = root / "data" / "processed" / "corpus" / "artifact-index-v0.1.json"
            index_path.write_text(json.dumps({
                "artifacts": [{"relativePath": "data/processed/corpus/thing.json", "sha256": "wrong"}]
            }), encoding="utf-8")
            errors: list[str] = []
            check_artifact_index(root, errors)
            self.assertEqual(len(errors), 1)
            self.assertIn("mismatch", errors[0])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=python python3 -m unittest python.tests.test_check_corpus_provenance -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'check_corpus_provenance'`

- [ ] **Step 3: Write minimal implementation**

Create `scripts/check_corpus_provenance.py`:

```python
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
    sha256_file,
)

ARTIFACT_SCHEMAS = {
    "data/processed/corpus/corpus-statistics-v0.1.json": "corpus-statistics.schema.json",
    "data/processed/corpus/corpus-splits-v0.1.json": "corpus-splits.schema.json",
    "data/processed/corpus/baseline-comparison-v0.1.json": "baseline-comparison.schema.json",
    "data/fixtures/pilot-stimulus-v0.1.json": "pilot-stimulus.schema.json",
    "mappings/controls/sequential-chromatic-v0.1.json": "control-mapping.schema.json",
    "mappings/controls/frequency-ranked-v0.1.json": "control-mapping.schema.json",
    "mappings/controls/circle-of-fifths-v0.1.json": "control-mapping.schema.json",
    "mappings/controls/random-seed-026-v0.1.json": "control-mapping.schema.json",
}


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
        provenance = payload.get("provenance", {})
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
        if provenance.get("normalization_implementation_sha256") != live_sha256:
            errors.append(
                f"{artifact_path}: normalization_implementation_sha256 does not match "
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
    required_fields = ("package_id", "source_version", "archive_sha256", "installed_content_sha256", "retrieved_at")
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
        actual_sha256 = sha256_file(artifact_path)
        if entry.get("sha256") != actual_sha256:
            errors.append(
                f"artifact-index checksum mismatch for {relative_path}: "
                f"recorded {entry.get('sha256')}, actual {actual_sha256}"
            )


def main() -> None:
    errors: list[str] = []
    check_schema_validation(ROOT, errors)
    check_provenance(ROOT, [ROOT / p for p in ARTIFACT_SCHEMAS], errors)
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=python python3 -m unittest python.tests.test_check_corpus_provenance -v`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add scripts/check_corpus_provenance.py python/tests/test_check_corpus_provenance.py
git commit -m "feat: add CI-wired provenance/schema/lock/artifact-index checker"
```

---

## Task 17: Makefile targets

**Files:**
- Modify: `Makefile`

**Interfaces:**
- Produces: `make corpus-setup`, `make corpus-lock`, `make corpus-relock`, `make corpus-pipeline`, `make corpus-check`, `make corpus-provenance-check` — thin wrappers around Tasks 13–16's scripts.

- [ ] **Step 1: Add the new targets**

Modify `Makefile` — add to the `.PHONY` line and append the new targets, keeping every existing target unchanged:

```makefile
.PHONY: test validate test-explorer test-python test-ts test-parity \
        corpus-setup corpus-lock corpus-relock corpus-pipeline corpus-check corpus-provenance-check

test: validate test-explorer test-python test-ts test-parity

validate:
	python3 scripts/validate_mapping.py

test-explorer:
	python3 scripts/check_word_explorer_mapping.py

test-python:
	PYTHONPATH=python python3 -m unittest discover -s python/tests -v

test-ts:
	node --experimental-strip-types --test packages/core-ts/test/core.test.ts

test-parity:
	node --experimental-strip-types scripts/check_cross_runtime.ts

corpus-setup:
	python3 scripts/setup_corpus.py setup

corpus-lock:
	python3 scripts/setup_corpus.py lock

corpus-relock:
	python3 scripts/setup_corpus.py relock

corpus-pipeline:
	python3 scripts/build_corpus_pipeline.py

corpus-check:
	python3 scripts/check_corpus_pipeline.py

corpus-provenance-check:
	python3 scripts/check_corpus_provenance.py
```

- [ ] **Step 2: Verify the new targets are wired correctly**

Run: `make -n corpus-pipeline`
Expected: prints `python3 scripts/build_corpus_pipeline.py` without executing it (dry run).

Run: `make -n corpus-setup corpus-lock corpus-relock corpus-check corpus-provenance-check`
Expected: prints the five corresponding commands without executing them.

- [ ] **Step 3: Commit**

```bash
git add Makefile
git commit -m "build: add corpus-setup/lock/relock/pipeline/check/provenance-check Makefile targets"
```

---

## Task 18: CI wiring (no network)

**Files:**
- Modify: `.github/workflows/ci.yml`

**Interfaces:**
- Adds two steps to the existing `validate-and-test` job: installing the pinned research dependencies (needed because several new tests import `jsonschema`/`yaml`) and running `scripts/check_corpus_provenance.py`. No new workflow file, no `nltk.download()`, no network access — the existing "Run Python tests" step already picks up every new `python/tests/test_corpus_*.py` / `test_*_cli.py` / `test_schemas_valid.py` / `test_dataset_manifests.py` file via `unittest discover`, since they're all network-free by construction.

- [ ] **Step 1: Add the two new steps**

Modify `.github/workflows/ci.yml` — insert an "Install research dependencies" step right after "Set up Node" and before "Validate mapping", and add a "Check corpus provenance and schema integrity" step after "Check Python/TypeScript parity":

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:

permissions:
  contents: read

jobs:
  validate-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Set up Node
        uses: actions/setup-node@v4
        with:
          node-version: "22"

      - name: Install research dependencies
        run: pip3 install -r requirements-research.txt

      - name: Validate mapping
        run: python3 scripts/validate_mapping.py

      - name: Check Word Explorer mapping
        run: python3 scripts/check_word_explorer_mapping.py

      - name: Run Python tests
        run: PYTHONPATH=python python3 -m unittest discover -s python/tests -v

      - name: Run TypeScript tests
        run: node --experimental-strip-types --test packages/core-ts/test/core.test.ts

      - name: Check Python/TypeScript parity
        run: node --experimental-strip-types scripts/check_cross_runtime.ts

      - name: Check corpus provenance and schema integrity
        run: python3 scripts/check_corpus_provenance.py
```

- [ ] **Step 2: Verify the workflow file is valid YAML**

Run: `python3 -c "import yaml; yaml.safe_load(open('.github/workflows/ci.yml'))" && echo "valid YAML"`
Expected: prints `valid YAML`.

- [ ] **Step 3: Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: install research deps and check corpus provenance/schema integrity"
```

---

## Task 19: Documentation updates (DATA_SOURCES.md, THIRD_PARTY_NOTICES.md, ROADMAP.md)

**Files:**
- Modify: `DATA_SOURCES.md`
- Modify: `THIRD_PARTY_NOTICES.md`
- Modify: `ROADMAP.md`

**Interfaces:** none (documentation only); no code depends on these.

- [ ] **Step 1: Add the approved-datasets index to DATA_SOURCES.md**

Append to `DATA_SOURCES.md` (after the existing "## Listener data" section):

```markdown

## Approved corpus datasets (Phase 2)

Five datasets are approved for the corpus and stimulus pipeline. Each has a
manifest under `data/manifests/`. Brown, Names, and CMUdict are explicitly
excluded from this pipeline.

| Dataset | Role | License | Redistribution | Manifest |
|---|---|---|---|---|
| Gutenberg (NLTK sample) | Natural text, sentences, boundaries, character/bigram frequency | Public Domain (US) | Yes — derived stats only | `data/manifests/gutenberg.yaml` |
| words | Candidate vocabulary, orthographic stress cases | Public Domain / unrestricted | Yes — selected words only | `data/manifests/words.yaml` |
| WordNet | Dictionary validation, part-of-speech, morphological grouping | WordNet License (permissive) | Yes — per-item validation results only | `data/manifests/wordnet.yaml` |
| Opinion Lexicon | Positive/negative candidate labels | Research/academic use (Hu & Liu 2004) | Yes, per project decision — derived per-item metadata only | `data/manifests/opinion-lexicon.yaml` |
| VADER | Sentiment scoring / polarity confirmation | MIT License | Yes | `data/manifests/vader.yaml` |

See `data/corpus/README.md` for dataset roles and exclusion rules in detail.
```

- [ ] **Step 2: Add attribution notices to THIRD_PARTY_NOTICES.md**

Append to `THIRD_PARTY_NOTICES.md` (after the existing paragraph):

```markdown

## Corpus pipeline datasets (Phase 2)

- **NLTK Gutenberg Sample Corpus** — Project Gutenberg texts, Public Domain (US), bundled via NLTK (`nltk.download('gutenberg')`).
- **NLTK Words Corpus** — derived from the Unix words list, public domain / unrestricted, bundled via NLTK (`nltk.download('words')`).
- **Princeton WordNet** (via NLTK) — WordNet License. "This software and database is being provided to you, the LICENSEE, by Princeton University under the following license. ... Princeton University makes no representations about the suitability of the licensed software, database or documentation for any purpose." Bundled via NLTK (`nltk.download('wordnet')`).
- **Opinion Lexicon** — Minqing Hu and Bing Liu, "Mining and Summarizing Customer Reviews", KDD 2004. Published for academic/research use; bundled via NLTK (`nltk.download('opinion_lexicon')`).
- **VADER Sentiment Lexicon** — Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. ICWSM-14. MIT License. Bundled via NLTK (`nltk.download('vader_lexicon')`).

Full licensing and redistribution details for each are in `data/manifests/`.
```

- [ ] **Step 3: Restructure ROADMAP.md Phase 2 and flip the Phase 0 checkbox**

In `ROADMAP.md`, change:

```markdown
- [ ] Create public GitHub repository
```

to:

```markdown
- [x] Create public GitHub repository
```

(Branch protection and the `v0.1.0-experimental` tag stay unchecked — neither exists yet.)

Replace the entire existing "## Phase 2 — Evaluation framework" section:

```markdown
## Phase 2 — Evaluation framework

- [ ] Implement sequential chromatic control
- [ ] Implement frequency-ranked control
- [ ] Implement circle-of-fifths control
- [ ] Generate 100 seeded random controls
- [ ] Add interval, register, tonality, chord, identity, and fatigue reports
- [ ] Publish first controlled comparison report
```

with:

```markdown
## Phase 2 — Evaluation framework

### Corpus and stimulus foundation

- [x] Add license-aware dataset manifests
- [x] Add explicit corpus download/setup tooling
- [x] Normalize approved corpus inputs using the documented LAPE normalization profile
- [x] Generate character, bigram, boundary, word-length, and sentence-length statistics
- [x] Create deterministic train, validation, and holdout partitions
- [x] Generate the versioned 120-item controlled pilot stimulus fixture
- [x] Record selection seed, source metadata, exclusions, and checksums
- [x] Add reproducibility tests

### Mapping evaluation and controls

- [x] Implement sequential chromatic control
- [x] Implement frequency-ranked control
- [x] Implement circle-of-fifths control
- [x] Generate one seeded random control (`random-seed-026-v0.1`)
- [ ] Generate 100 seeded random controls
- [x] Add descriptive baseline comparison metrics (register, pitch span, interval, directional balance, repetition index)
- [ ] Add experimental heuristic comparison metrics (tonality, chord, identity, fatigue, composite) — future Phase 2 metric-design PR
- [x] Publish first descriptive baseline comparison report
```

- [ ] **Step 4: Verify no other ROADMAP content changed**

Run: `git diff ROADMAP.md`
Expected: only the Phase 0 checkbox line and the Phase 2 section are changed; Phase 1 and Phases 3–6 are untouched.

- [ ] **Step 5: Commit**

```bash
git add DATA_SOURCES.md THIRD_PARTY_NOTICES.md ROADMAP.md
git commit -m "docs: index approved datasets, add attributions, restructure ROADMAP Phase 2"
```

---

## Task 20: Offline-safety static check (no network calls reachable)

**Files:**
- Create: `python/tests/test_corpus_offline.py`

**Interfaces:**
- Produces: `find_module_level_nltk_imports(source_path: Path) -> list[str]` — parses a `.py` file with `ast` and returns a description for every `import nltk` / `from nltk import ...` found at **module scope** (top-level `tree.body`), ignoring imports nested inside function/method bodies. Used to assert that `python/lape26/corpus/tokens.py`, `stats.py`, `splits.py`, `provenance.py`, `controls.py`, `stimulus.py`, and `report.py` never import `nltk` at module scope — only `acquire.py`'s `download_approved_packages` function body may.

- [ ] **Step 1: Write the failing test**

Create `python/tests/test_corpus_offline.py`:

```python
from __future__ import annotations

import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CORPUS_DIR = ROOT / "python" / "lape26" / "corpus"

# acquire.py is allowed nltk usage, but only inside a function body, never
# at module scope — that's checked separately, not exempted here.
ALL_CORPUS_MODULES = [
    "__init__.py", "tokens.py", "stats.py", "splits.py", "provenance.py",
    "controls.py", "acquire.py", "stimulus.py", "report.py",
]


def find_module_level_nltk_imports(source_path: Path) -> list[str]:
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    findings: list[str] = []
    for node in tree.body:  # top-level only — does not recurse into functions
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "nltk" or alias.name.startswith("nltk."):
                    findings.append(f"{source_path}:{node.lineno}: module-level `import {alias.name}`")
        elif isinstance(node, ast.ImportFrom):
            if node.module and (node.module == "nltk" or node.module.startswith("nltk.")):
                findings.append(f"{source_path}:{node.lineno}: module-level `from {node.module} import ...`")
    return findings


class NoModuleLevelNltkImportTests(unittest.TestCase):
    def test_no_corpus_module_imports_nltk_at_module_scope(self) -> None:
        all_findings: list[str] = []
        for filename in ALL_CORPUS_MODULES:
            path = CORPUS_DIR / filename
            self.assertTrue(path.exists(), f"missing {path}")
            all_findings.extend(find_module_level_nltk_imports(path))
        self.assertEqual(all_findings, [], "\n".join(all_findings))

    def test_acquire_module_still_imports_nltk_somewhere(self) -> None:
        # Sanity check that the finder isn't trivially passing because nltk
        # usage was removed entirely — acquire.py must still reference nltk,
        # just not at module scope.
        source = (CORPUS_DIR / "acquire.py").read_text(encoding="utf-8")
        self.assertIn("import nltk", source)

    def test_every_corpus_module_is_importable_without_nltk_installed_check(self) -> None:
        # This test documents the intent; actual "nltk not installed" is
        # exercised by CI only installing requirements-research.txt (which
        # does include nltk) — see Task 18. The AST check above is what
        # actually guarantees module-scope safety regardless of whether
        # nltk happens to be installed in a given environment.
        for filename in ALL_CORPUS_MODULES:
            self.assertTrue((CORPUS_DIR / filename).exists())


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=python python3 -m unittest python.tests.test_corpus_offline -v`
Expected: PASS or FAIL depending on whether Tasks 5–11 already avoided module-scope `nltk` imports — since those tasks were written to import `nltk` only inside `download_approved_packages` in `acquire.py`, this should already be GREEN the first time it's run. If so, skip Step 3 and go straight to Step 4 to confirm; this is the one task in the plan where RED may not reproduce because earlier tasks already did the work correctly — that is expected and fine, not a plan defect.

- [ ] **Step 3: Fix any violations found**

If Step 2 reported findings, move the offending `import nltk` line from module scope into the specific function body that needs it, matching the pattern already used in `acquire.py`'s `download_approved_packages`.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=python python3 -m unittest python.tests.test_corpus_offline -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add python/tests/test_corpus_offline.py
git commit -m "test: statically verify nltk is never imported at module scope in corpus/"
```

---

## Task 21: Generate and commit the real artifacts (with mandatory manual review)

**Files:**
- Modify: `scripts/build_corpus_pipeline.py` (add a manual-exclusion filter)
- Create: `data/manifests/stimulus-exclusions.yaml`
- Create (generated, not hand-written): `data/manifests/corpus-lock.json`, `data/processed/corpus/corpus-statistics-v0.1.json`, `data/processed/corpus/corpus-splits-v0.1.json`, `data/processed/corpus/baseline-comparison-v0.1.json`, `data/processed/corpus/baseline-comparison-v0.1.md`, `data/processed/corpus/artifact-index-v0.1.json`, `data/fixtures/pilot-stimulus-v0.1.json`, `mappings/controls/sequential-chromatic-v0.1.json`, `mappings/controls/frequency-ranked-v0.1.json`, `mappings/controls/circle-of-fifths-v0.1.json`, `mappings/controls/random-seed-026-v0.1.json`

**Why a manual step is required here (not automatable):** `load_word_candidates` (Task 14) filters on WordNet validity, `.isalpha()`, and Opinion Lexicon/VADER polarity, but has no way to detect slurs, distressing terms, or the subtler proper-noun/abbreviation cases WordNet's common-noun senses don't catch. The spec requires excluding these; only a human reviewing the actual 120 selected words can do that reliably.

- [ ] **Step 1: Add a manual-exclusion filter to the loaders**

In `scripts/build_corpus_pipeline.py`, add this function near the top (after `_POS_NAMES`):

```python
def _load_manual_exclusions() -> set[str]:
    import yaml

    exclusions_path = ROOT / "data" / "manifests" / "stimulus-exclusions.yaml"
    if not exclusions_path.exists():
        return set()
    data = yaml.safe_load(exclusions_path.read_text(encoding="utf-8")) or {}
    return {str(word).upper() for word in data.get("excluded_words", [])}
```

In `load_word_candidates`, right after `seen_words: set[str] = set()`, add:

```python
    excluded_words = _load_manual_exclusions()
```

and change the loop's skip condition from:

```python
        if not word.isalpha() or word in seen_words:
            continue
```

to:

```python
        if not word.isalpha() or word in seen_words or word.upper() in excluded_words:
            continue
```

In `load_orthographic_candidates`, right after `seen: set[str] = set()`, add the same `excluded_words = _load_manual_exclusions()` line, and change:

```python
        if not word.isalpha() or len(word) < 3 or word in seen:
            continue
```

to:

```python
        if not word.isalpha() or len(word) < 3 or word in seen or word.upper() in excluded_words:
            continue
```

Create `data/manifests/stimulus-exclusions.yaml` (starts empty — populated only if manual review in Step 4 finds a problem):

```yaml
# Words to exclude from pilot stimulus candidate generation, keyed by
# manual review (Task 21). Populate only after finding a real problem in
# a generated fixture — do not pre-populate speculatively.
excluded_words: []
```

- [ ] **Step 2: Run corpus-setup (real network access, local only, never in CI)**

Run: `pip3 install -r requirements-research.txt && python3 -c "import nltk; nltk.download('punkt_tab')" && make corpus-setup`

(`punkt_tab` is NLTK's sentence/word tokenizer data — not one of the 5 approved *content* corpora, but is required by some NLTK corpus readers' internal tokenization; if `make corpus-setup` completes without needing it, this extra download can be skipped. Verify by running `make corpus-setup` first and only adding the `punkt_tab` download if it errors.)

Expected: `Downloaded and locked 5 approved corpora to .../data/raw/nltk_data` and a new `data/manifests/corpus-lock.json` file.

- [ ] **Step 3: Run corpus-pipeline**

Run: `make corpus-pipeline`
Expected: 10 "Wrote ..." lines, one per artifact listed in this task's file list above.

- [ ] **Step 4: Manually review the 120-item stimulus fixture**

Run: `python3 -c "import json; data = json.load(open('data/fixtures/pilot-stimulus-v0.1.json')); print('\n'.join(f\"{i['word']} ({i['polarity']}, {i['partOfSpeech']})\" for i in data['coreSet'])); print('---'); print('\n'.join(f\"{i['word']} ({i['category']})\" for i in data['orthographicChallengeSet']))"`

Read through all 120 words. For each, check against the spec's exclusion rules: proper nouns, abbreviations, duplicates, close morphological variants, highly obscure words, slurs, potentially distressing terms. If any word fails this review:

1. Add it (uppercase) to `excluded_words` in `data/manifests/stimulus-exclusions.yaml`.
2. Re-run `make corpus-pipeline` (deterministic given the same seed, so only the excluded word's cell changes — a different candidate fills that slot).
3. Repeat this step until the full 120-item list passes review.

- [ ] **Step 5: Run corpus-check to confirm the pipeline is stable**

Run: `make corpus-check`
Expected: `corpus-check: regenerated artifacts match committed artifacts.`

- [ ] **Step 6: Commit the generated artifacts**

```bash
git add scripts/build_corpus_pipeline.py data/manifests/stimulus-exclusions.yaml \
        data/manifests/corpus-lock.json data/processed/corpus/ data/fixtures/pilot-stimulus-v0.1.json \
        mappings/controls/
git commit -m "feat: generate and commit corpus statistics, splits, controls, stimulus fixture, and baseline report"
```

---

## Task 22: Final verification

**Files:** none modified — this task only runs checks.

- [ ] **Step 1: Run the full existing test suite (must still pass unchanged)**

Run: `make test`
Expected: `validate`, `test-explorer`, `test-python`, `test-ts`, `test-parity` all pass, exactly as before this branch existed.

- [ ] **Step 2: Run the full Python test suite explicitly (covers every new corpus test)**

Run: `PYTHONPATH=python python3 -m unittest discover -s python/tests -v`
Expected: every test from Tasks 1–20 passes — `test_normalize`, `test_schemas_valid`, `test_dataset_manifests`, `test_corpus_tokens`, `test_corpus_stats`, `test_corpus_splits`, `test_corpus_provenance`, `test_corpus_controls`, `test_corpus_lock`, `test_corpus_stimulus`, `test_corpus_report`, `test_setup_corpus_cli`, `test_build_corpus_pipeline`, `test_check_corpus_pipeline`, `test_check_corpus_provenance`, `test_corpus_offline`, plus the pre-existing `test_core`.

- [ ] **Step 3: Run the CI-equivalent corpus provenance check against the real committed artifacts**

Run: `python3 scripts/check_corpus_provenance.py`
Expected: `check_corpus_provenance: all committed corpus artifacts are valid.`

- [ ] **Step 4: Run corpus-check one more time**

Run: `make corpus-check`
Expected: `corpus-check: regenerated artifacts match committed artifacts.`

- [ ] **Step 5: Confirm the canonical mapping is byte-identical to before this branch**

Run: `git diff main -- mappings/lape-26-en-general-v0.1.json`
Expected: empty output (no diff).

- [ ] **Step 6: Confirm mappings/mapping.schema.json was never touched**

Run: `git diff main -- mappings/mapping.schema.json`
Expected: empty output (no diff).

- [ ] **Step 7: Confirm specs/text-normalization.md was never touched**

Run: `git diff main -- specs/text-normalization.md`
Expected: empty output (no diff).

- [ ] **Step 8: Review the full diff for anything unexpected**

Run: `git diff main --stat`
Expected: only files listed across Tasks 1–21's file manifests appear — no unrelated project files.

- [ ] **Step 9: Confirm the working tree is clean before any future artifact regeneration**

Run: `git status --short`
Expected: empty (everything from this branch is committed).

No commit for this task — it's verification-only. If any step fails, fix the underlying issue in the task that introduced it and re-run this task from Step 1.

