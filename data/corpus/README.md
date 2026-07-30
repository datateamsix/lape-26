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

## Token preservation scope (narrow, by design)

The `CorpusToken` model in this phase preserves: document ID, sentence
index, source token index (before punctuation is dropped), normalized
word index (after punctuation is dropped), and sentence-boundary flags.
It does **not** retain punctuation as metadata, character-level source
offsets, or whitespace metadata — nothing in this phase consumes those,
and claiming otherwise would overstate what the pipeline actually does.

## Dataset roles

| Dataset | Role | Must NOT be used for |
|---|---|---|
| Gutenberg | Natural text, sentences, boundaries, character/bigram transitions, sentence-length stats, primary source of natural character/bigram frequency | — |
| words | Candidate vocabulary, orthographic stress cases | Natural usage frequency |
| WordNet | Dictionary validation, part-of-speech, morphological grouping | Usage-frequency inference |
| Opinion Lexicon | Positive/negative candidate labels, confirmed against VADER | Directly influencing the musical objective |
| VADER | Sentiment scoring / polarity confirmation | Directly influencing the musical objective |

Excluded entirely: Brown corpus, Names corpus, CMUdict, and `punkt`/`punkt_tab`
(sentence splitting uses this project's own deterministic splitter instead —
see `python/lape26/corpus/nltk_adapter.py` — specifically to avoid adding a
6th, unreviewed NLTK resource to the approved-dataset boundary).

## Directory map

```text
data/
  manifests/          Dataset manifests + corpus-lock.json + stimulus-exclusions.yaml
  corpus/README.md    This file
  schemas/            JSON Schemas for every generated artifact
  raw/nltk_data/       Downloaded corpora (gitignored, local only)
  processed/corpus/    Statistics, splits, comparison report, artifact index (committed)
  fixtures/pilot-stimulus-v0.1.json   120-item fixture (committed)
mappings/controls/     Generated control mapping JSON files (committed)
```

## Commands

- `make corpus-setup` — downloads exactly the 5 approved NLTK packages, creates `data/manifests/corpus-lock.json` if absent (verifies if present). Local only, never run in CI, never touches undeclared packages.
- `make corpus-relock` — explicit, intentional lock regeneration from the current local cache; unlike `corpus-lock`, always overwrites and reports what changed.
- `make corpus-pipeline` — offline; verifies the lock, then generates statistics, splits, the stimulus fixture, control mappings, and the baseline report.
- `make corpus-check` — local-only maintainer safeguard; regenerates into a temp directory and diffs against committed artifacts. Not run in CI.

## Leakage prevention

The `frequency-ranked` control mapping is built from the Gutenberg
**train** partition only, then evaluated (in the baseline report) against
Gutenberg validation/holdout, word-list validation/holdout, and the pilot
fixture. The 120-item pilot stimulus fixture is not a member of the
train/validation/holdout split — it is a separate, frozen, independently
versioned evaluation fixture. Every word in it (both the core set and the
orthographic-challenge set) is drawn only from training-side vocabulary,
and a single shared exclusion set guarantees no word appears twice
anywhere in the 120 items.
