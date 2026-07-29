# Contributing to LAPE-26

Thank you for helping explore the musicality of language through an open, reproducible, and appropriately skeptical research process.

## Before contributing

1. Read `README.md`, `GOVERNANCE.md`, and the relevant specification.
2. Search existing issues and proposals.
3. Open an issue before large architectural, mapping, metric, or study-protocol changes.
4. Keep creative hypotheses clearly separate from established physical or statistical facts.

## Development checks

Run:

```bash
make test
```

All pull requests should preserve:

- Deterministic output
- Mapping-schema validity
- Python/TypeScript golden-vector parity
- Existing released mapping files
- Versioned metric behavior
- Dataset license metadata

## Contribution types

- Core specification
- TypeScript runtime
- Python research tooling
- Audio rendering
- Mapping analysis
- Dataset manifests
- Listening-study methods
- Accessibility
- Documentation
- Visual or musical examples

## Canonical mapping changes

Do not directly overwrite a released mapping. Create a new candidate mapping and an LMP document under:

```text
specs/proposals/LMP-NNNN-short-title.md
```

A mapping proposal must include:

- Motivation
- Complete proposed mapping
- Mapping checksum
- Generation algorithm and seed
- Dataset manifests
- Objective and metric versions
- Baseline comparisons
- Listener evidence, when available
- Reproduction instructions
- Limitations and likely failure modes
- Backward-compatibility impact

## Data contributions

Do not commit copyrighted lyrics, personal listener data, or restricted datasets unless redistribution is explicitly permitted. Add a manifest and license notes for every data source.

## Commit guidance

Prefer focused commits such as:

```text
feat(core): add punctuation event metadata
research(metrics): add interval distribution report
fix(mapping): reject duplicate MIDI assignments
docs(method): clarify heuristic musicality boundary
```
