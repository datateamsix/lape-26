# Roadmap

## Phase 0 — Open-source foundation

**Goal:** Publish a credible, runnable, and inspectable research project.

- [x] Define project purpose and scientific boundary
- [x] Publish Apache-2.0 license and governance files
- [x] Publish canonical experimental mapping JSON
- [x] Publish mapping and event schemas
- [x] Include Word Explorer prototype
- [x] Include Python and TypeScript reference encoders
- [x] Include golden test vectors
- [x] Include mapping validation and CI
- [x] Create public GitHub repository
- [ ] Enable branch protection and required checks
- [ ] Tag `v0.1.0-experimental`

## Phase 1 — Deterministic core

- [ ] Finalize normalization profile v0.1
- [ ] Finalize canonical musical-event schema v0.1
- [ ] Add punctuation and sentence event fixtures
- [ ] Publish npm and Python packages experimentally
- [ ] Add MIDI export reference
- [ ] Add checksum generation for mappings

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
- [x] Add descriptive baseline comparison metrics (register, pitch span, interval, directional balance, repetition index), evaluated against the pilot fixture, Gutenberg validation/holdout, and word-list validation/holdout
- [ ] Add experimental heuristic comparison metrics (tonality, chord, identity, fatigue, composite) — future Phase 2 metric-design PR
- [x] Publish first descriptive baseline comparison report

## Phase 3 — Candidate optimization

- [ ] Implement all 325 two-letter swaps
- [ ] Implement seeded simulated annealing
- [ ] Define objective-function versioning
- [ ] Produce computational candidate shortlist

## Phase 4 — Listener study

- [ ] Finalize consent and privacy language
- [ ] Build blinded A/B study application
- [ ] Lock instrument, tempo, and loudness per study
- [ ] Collect pilot responses
- [ ] Publish aggregate results and limitations

## Phase 5 — Preference model

- [ ] Fit Bradley–Terry baseline
- [ ] Fit interpretable logistic or tree model
- [ ] Hold out words and listeners
- [ ] Re-optimize objective weights
- [ ] Conduct confirmatory study

## Phase 6 — Version 1.0 proposal

A `1.0` mapping proposal must outperform simple and random controls, generalize to held-out data, remain comfortable across longer text, and include reproducible human-evaluation evidence.
