# Phase 1 (branch) / ROADMAP Phase 2 — Corpus Evaluation Foundation

- **Branch:** `phase-1/corpus-evaluation-foundation` (informal branch name only — see Roadmap alignment below)
- **Date:** 2026-07-29
- **Status:** Draft for review

## 1. Scope & goal

Build a reproducible, license-aware corpus and stimulus pipeline for LAPE-26: documented dataset manifests, normalized-corpus statistics, deterministic train/validation/holdout splits, a versioned 120-item pilot stimulus fixture, four control mappings, and a descriptive baseline comparison report — without modifying the canonical `mappings/lape-26-en-general-v0.1.json`.

**Deliverable framing:** this produces a reproducible linguistic benchmark and pilot stimulus set. It is explicitly **not** a claim that the LAPE-26 mapping has been trained or validated musically, and the baseline report must not rank mappings by "quality," "musicality," or similar.

## 2. Roadmap alignment

ROADMAP.md's phase numbering is **not** renumbered. `phase-1/corpus-evaluation-foundation` is informal branch naming; the work lands under the existing **Phase 2 — Evaluation framework**, restructured into two subsections:

**Corpus and stimulus foundation**
- License-aware dataset manifests
- Explicit corpus download/setup tooling
- Normalization of approved corpus inputs via the documented profile
- Character, bigram, boundary, word-length, and sentence-length statistics
- Deterministic train/validation/holdout partitions
- Versioned 120-item controlled pilot stimulus fixture
- Recorded selection seed, source metadata, exclusions, checksums
- Reproducibility tests

**Mapping evaluation and controls**
- Sequential chromatic, frequency-ranked, circle-of-fifths controls
- One seeded random control (`random-seed-026-v0.1`) — **not** the full 100 required by the existing roadmap line, which stays unchecked
- Descriptive baseline comparison metrics (register, pitch span, interval, directional balance, repetition index) — **checked**
- Experimental heuristic metrics (tonality, chord, identity, fatigue, composite) — **added as a new unchecked line**, explicitly deferred to a future Phase 2 metric-design PR
- First descriptive baseline comparison report — **checked**

Also: flip Phase 0's "Create public GitHub repository" checkbox to done (verified via `gh api repos/datateamsix/lape-26` → `private: false`). Branch protection and the `v0.1.0-experimental` tag stay unchecked — neither exists yet.

## 3. Normalization dependency

`specs/text-normalization.md` is not modified and not formally "finalized" in this workstream (ROADMAP Phase 1's "Finalize normalization profile v0.1" stays unchecked — out of scope here). Instead:

- `normalize_text()` is extracted from `python/lape26/core.py` into its own `python/lape26/normalize.py` (mechanical extraction, `core.py` re-imports it; behavior unchanged — `encode_text` and existing tests are unaffected).
- `data/corpus/README.md` carries this notice verbatim:

  > This pipeline treats `lape-text-normalization-v0.1` as a provisional frozen profile for corpus-generation purposes. It has not been formally finalized under ROADMAP Phase 1. Any change to the profile or its implementation requires regeneration and re-versioning of all dependent corpus statistics, splits, stimulus fixtures, control mappings derived from corpus statistics, and comparison reports.

- Every generated artifact carries a machine-readable `provenance` block (schema in §9):
  ```yaml
  normalization_profile_id: lape-text-normalization-v0.1
  normalization_status: provisional-frozen
  normalization_spec_path: specs/text-normalization.md
  normalization_implementation: python/lape26/normalize.py
  normalization_implementation_sha256: <content checksum>
  normalization_git_blob_sha: <git blob SHA of normalize.py>
  pipeline_source_commit: <commit used to generate, informational>
  working_tree_dirty: false
  source_tree_digest: <deterministic digest of pipeline source files>
  ```
- `scripts/check_corpus_provenance.py` (run in CI) requires: profile ID matches, implementation path matches, live-file SHA-256 matches, schema version supported, and `working_tree_dirty` is not `true` on any committed artifact. It does **not** require `pipeline_source_commit == current HEAD` — unrelated later commits must not invalidate valid artifacts.
- **Git workflow implication:** the pipeline implementation is committed first (clean tree); generated artifacts are produced and committed in a second commit, so their provenance block reflects a clean source state.

## 4. Structure preservation & bigram correctness

Corpus text is parsed into an intermediate `CorpusToken` representation *before* collapsing to the pitch-bearing A–Z stream, preserving: word boundaries, sentence boundaries, whitespace boundaries, punctuation metadata, and original source indices. Bigram counting never crosses a word boundary undetected. The following statistics are generated (the bigram matrix is split into within-word and cross-word components rather than one flat matrix):

- Character frequencies
- Within-word character bigrams
- Cross-word boundary transitions
- Word-initial letter frequencies
- Word-final letter frequencies
- Sentence-initial letter frequencies
- Sentence-final letter frequencies
- Word-length statistics
- Sentence-length statistics

## 5. Dataset roles (documented in `data/corpus/README.md`)

| Dataset | Role | Must NOT be used for |
|---|---|---|
| Gutenberg (NLTK sample, ~18 texts) | Natural text, sentences, boundaries, character transitions, sentence-length stats, **primary source of natural bigram/character frequency** | — |
| `words` | Candidate vocabulary, orthographic stress cases | Natural usage frequency |
| WordNet | Dictionary validation, part-of-speech, semantic filtering | Usage-frequency inference |
| Opinion Lexicon | Positive/negative candidate word labels | Directly influencing the musical objective |
| VADER | Sentiment scoring / polarity confirmation | Directly influencing the musical objective |

Excluded from the pipeline entirely: Brown, Names, CMUdict.

## 6. Dependency management

- `requirements-research.txt` — exact-pinned versions (`nltk==...`, `pyyaml==...`), the authoritative install source used by `make corpus-setup`.
- `pyproject.toml` gets a loose `[project.optional-dependencies].research` entry for standard packaging discoverability only; the exact pins in `requirements-research.txt` are what's actually installed.

## 7. Directory layout

```text
data/
  manifests/
    gutenberg.yaml, words.yaml, wordnet.yaml,
    opinion-lexicon.yaml, vader.yaml        (committed; extended schema — see §9)
    corpus-lock.json                        (committed; generated by corpus-setup)
  corpus/README.md                          (committed; roles, freeze notice, directory map)
  schemas/                                  (committed; see §9)
  raw/nltk_data/                            (gitignored — local download cache)
  processed/
    .tmp/, .check/                          (gitignored — scratch dirs for corpus-check)
    corpus/
      corpus-statistics-v0.1.json           (committed)
      corpus-splits-v0.1.json               (committed — words split + Gutenberg book split)
      artifact-index-v0.1.json              (committed)
      baseline-comparison-v0.1.json         (committed)
      baseline-comparison-v0.1.md           (committed)
  fixtures/
    pilot-stimulus-v0.1.json                (committed — frozen, independent of corpus splits)
mappings/controls/
  sequential-chromatic-v0.1.json
  frequency-ranked-v0.1.json                (built from Gutenberg TRAIN partition only)
  circle-of-fifths-v0.1.json
  random-seed-026-v0.1.json
```

## 8. Corpus lock file & command behavior

`data/manifests/corpus-lock.json` records, per installed NLTK package: `package_id`, `source_version` (`documented-or-unknown` where NLTK doesn't expose one), `archive_sha256`, `installed_content_sha256`, `retrieved_at`.

- **`make corpus-setup`** — downloads exactly the 5 approved NLTK packages into `data/raw/nltk_data/`, verifies package IDs, writes/verifies `corpus-lock.json`. Never runs in CI. Never downloads undeclared packages.
- **`make corpus-relock`** — explicit, intentional lock regeneration from the current local cache (the only sanctioned way to accept a changed upstream package).
- **`make corpus-pipeline`** — no network. Verifies the local cache against `corpus-lock.json` and refuses to proceed on mismatch (unless `corpus-relock` was run first). Builds `CorpusToken`s, generates statistics, source-aware splits, the stimulus fixture, control mappings, and the baseline report. Deterministic/byte-identical given unchanged inputs, seed, code, and environment.
- **`make corpus-check`** — **local-only maintainer safeguard, not run in CI.** Regenerates the full pipeline into a temporary directory and byte-for-byte diffs against committed artifacts, without overwriting them. Requires the verified local NLTK cache, same as `corpus-pipeline`.

NLTK is imported only inside the acquisition/adapter layer (`python/lape26/corpus/acquire.py`); `stats.py`, `splits.py`, `stimulus.py`, `controls.py`, and `report.py` operate on plain Python data structures so they're exercisable with the tiny fixture, with no NLTK import at module load time elsewhere.

## 9. Schemas & artifact index

New `data/schemas/`:
- `dataset-manifest.schema.json` (extends the existing manifest fields with preprocessing steps + redistribution detail)
- `corpus-provenance.schema.json` (the block described in §3)
- `corpus-statistics.schema.json`
- `corpus-splits.schema.json`
- `pilot-stimulus.schema.json`
- `baseline-comparison.schema.json`
- `control-mapping.schema.json` — **new, separate** from `mappings/mapping.schema.json` (which is left untouched, since it has `additionalProperties: false` and sits adjacent to the canonical mapping). Reuses the same `tuning`/`range`/`letters` shape and adds `controlType`, `generationMethod`, `seed`, `sourcePartition`, and the provenance block.

`data/processed/corpus/artifact-index-v0.1.json` — one authoritative inventory of every committed generated artifact: artifact ID, artifact version, relative path, SHA-256, pipeline version, normalization profile, generation seed (where applicable), source lock-file checksum.

Independent artifact versioning: `corpus-statistics-v0.1`, `corpus-splits-v0.1`, `pilot-stimulus-v0.1`, `control-sequential-chromatic-v0.1`, `control-frequency-ranked-v0.1`, `control-circle-of-fifths-v0.1`, `control-random-seed-026-v0.1`, `baseline-comparison-v0.1`, `pipeline_version: corpus-pipeline-v0.1`.

## 10. Splits (leakage prevention)

Two independent split artifacts inside `corpus-splits-v0.1.json`:

- **Word-list split**: NLTK `words`, deduplicated + normalized, shuffled with recorded seed, 80/10/10 train/validation/holdout.
- **Gutenberg split**: by whole source document (book), not by scattering sentences — a deterministic seeded assignment of the ~18 books to train/validation/holdout, balanced by word-count toward 80/10/10 while keeping each book intact in exactly one partition.

**Leakage rule:** `frequency-ranked-v0.1` is built from Gutenberg **train**-partition character/bigram frequency only. It is then evaluated (in the baseline report) against validation, holdout, and the pilot fixture — never fit and scored on the same material.

**Pilot fixture boundary:** the 120-item stimulus fixture is **not** a member of the train/validation/holdout split. It's a separate, frozen, independently versioned evaluation fixture. Its candidate words are drawn only from the training-side vocabulary (to avoid contaminating validation/holdout), but the fixture itself carries no train/val/holdout label.

## 11. Pilot stimulus fixture (`pilot-stimulus-v0.1.json`, 120 items)

**Core set — 108 items.** 3×3 matrix, 12 words per cell:
- Length: short (3–5 letters) / medium (6–8) / long (9–12)
- Polarity: positive / negative / neutral

Selection rules:
- Every item validated as a real English word via WordNet.
- Positive/negative candidates sourced from Opinion Lexicon, confirmed/scored via VADER.
- Neutral = WordNet-valid, absent from both lexicons, VADER compound score near zero.
- Excluded: proper nouns, abbreviations, duplicates, close morphological variants, highly obscure words, slurs, potentially distressing terms.
- Reasonable POS mix (noun/verb/adjective/adverb) within each cell where available.
- Per item, recorded: source dataset, polarity label, VADER compound score, length, part of speech, selection seed.

**Orthographic challenge set — 12 items**, 3 each: repeated letters, rare letters (J/Q/X/Z), vowel-heavy, consonant-heavy. No sentiment balancing required; purpose is stress-testing the deterministic mapping against unusual letter patterns.

**Interpretation boundary (recorded in the fixture and its docs):** sentiment labels are sampling metadata only. The pipeline must not assume or encode that positive words should sound consonant/major/pleasant/high-pitched or that negative words should sound dissonant/minor/low-pitched — the point is to leave that an open question for later listening-study phases.

**Representativeness boundary:** this is a controlled evaluation set, not a natural-language frequency sample — must not be described as representative of English without reliable usage-frequency balancing.

## 12. Baseline comparison report (`baseline-comparison-v0.1`)

Uses only already-implemented descriptive metrics from `python/lape26/analysis.py`:

```yaml
metric_versions:
  register_center: register_center_v0.1
  pitch_span: pitch_span_v0.1
  interval_contour: interval_contour_v0.1
  directional_balance: directional_balance_v0.1
  repetition_index: repetition_index_v0.1
mapping_ids:
  - lape-26-en-general-v0.1
  - sequential-chromatic-v0.1
  - frequency-ranked-v0.1
  - circle-of-fifths-v0.1
  - random-seed-026-v0.1
pipeline_version: corpus-pipeline-v0.1
```

`tonal_fit_v0.1` and other experimental heuristic metrics (`specs/analysis-metrics.md`) are **not** implemented in this workstream — they need their own design pass (exact mathematical definition, candidate key/scale set, repeated-note and chromatic-distance treatment, short/ambiguous-sequence handling, output range/interpretation, tests, evidence of usefulness, limitations). Note count is recorded as descriptive metadata, not a scored metric.

Report contents: mean/distribution of register center and pitch span, adjacent-interval distributions, upward/downward/repeated movement counts, directional balance, repetition index — broken out by mapping, by stimulus length×polarity strata, and separately for the orthographic challenge subset. Two outputs: `baseline-comparison-v0.1.json` (machine-readable) and `baseline-comparison-v0.1.md` (human-readable, for contributors/reviewers). Both must state:

> `baseline-comparison-v0.1` compares deterministic mappings using implemented descriptive metrics only. It does not measure objective musicality, consonance, emotional fit, or listener preference.

No language like "best mapping," "most musical," or "highest quality."

## 13. Tests

Network-free, run against `python/tests/fixtures/tiny-corpus-sample.txt` and small synthetic lexicons — never real NLTK data, never in CI beyond this:

- `test_corpus_provenance.py` — provenance block fields match live implementation; `working_tree_dirty` never `true` in committed artifacts.
- `test_corpus_splits.py` — determinism (same seed → identical split); Gutenberg split keeps whole books intact per partition.
- `test_corpus_stats.py` — within-word vs. cross-word bigram separation is correct on a hand-computed tiny example.
- `test_corpus_lock.py` — lock-mismatch correctly blocks `corpus-pipeline` before generation; structure/checksum validation.
- `test_corpus_stimulus.py` — exactly 120 unique items; 108 core + 12 orthographic; exactly 12 per length×polarity cell; no duplicate/morphological-variant overlap; correct orthographic category counts; deterministic given the fixed seed.
- `test_corpus_controls.py` — every control mapping has exactly 26 unique valid A–Z↔MIDI entries and passes `control-mapping.schema.json`; `frequency-ranked` provably built from Gutenberg train partition only.
- `test_corpus_report.py` — report declares only implemented metric versions; no unimplemented-metric or ranking language.
- `test_corpus_offline.py` — no NLTK import/network call reachable from `stats.py`, `splits.py`, `stimulus.py`, `controls.py`, `report.py` at module scope or during a fixture-driven pipeline run.
- Existing mapping-integrity check (`validate_mapping.py`) extended/reused to assert `mappings/lape-26-en-general-v0.1.json` is unchanged.

CI (`ci.yml`, extended in place — no new workflow file) runs: full pipeline determinism against the tiny fixture; provenance validation; schema validation; corpus-lock structural/checksum validation; artifact-index checksum verification; control-mapping integrity; pilot-fixture structural tests; canonical mapping integrity. **No `nltk.download()` call anywhere in CI.**

## 14. Documentation updates

- `DATA_SOURCES.md` — index of all 5 approved datasets: role, license, redistribution policy, manifest path.
- `THIRD_PARTY_NOTICES.md` — attribution/license notices, particularly for non-public-domain resources (WordNet, Opinion Lexicon, VADER).
- `ROADMAP.md` — restructured Phase 2 (§2), Phase 0 checkbox flip. No renumbering, no other phase content changes.
- `.gitignore` — add `data/raw/nltk_data/`, `data/processed/.tmp/`, `data/processed/.check/`, `__pycache__/`, `*.pyc`. Committed statistics/fixtures/controls/lock/report are never ignored.

## 15. Non-goals (explicitly out of scope)

- Changing `mappings/lape-26-en-general-v0.1.json` (must remain byte-identical — verified by test).
- Finalizing `specs/text-normalization.md` under ROADMAP Phase 1.
- Implementing `tonal_fit_v0.1` or other experimental heuristic metrics.
- Generating the full 100 seeded random controls (roadmap line stays unchecked).
- Any CI job that performs `nltk.download()` or otherwise touches the network.
- Renumbering or otherwise restructuring ROADMAP.md phases beyond Phase 2's two subsections and the one Phase 0 checkbox.
