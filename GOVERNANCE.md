# Governance

## Current model

LAPE-26 begins as a maintainer-led open-source creative research project. The founding maintainer is responsible for release integrity, research boundaries, mapping-version decisions, and protection of participant data.

## Decision principles

Decisions should favor:

1. Reproducibility
2. Determinism in released mappings
3. Transparent formulas and assumptions
4. Honest separation of evidence and interpretation
5. Backward-compatible versioning
6. Accessible participation by musicians and non-specialists
7. Respect for dataset licenses and listener privacy

## Maintainer responsibilities

- Review and merge contributions
- Approve releases
- Assign mapping and metric versions
- Maintain schemas and golden vectors
- Resolve conduct and security reports
- Document conflicts and rejected alternatives

## Mapping releases

Mapping states are:

```text
experimental → candidate → stable → deprecated
```

A stable mapping is never changed in place. Corrections require a new mapping ID and changelog entry.

## Research proposals

Substantial changes use a lightweight proposal process. Proposals remain public even when rejected so design history is preserved.

## Future governance

When the project has sustained independent contributors, governance may expand to include domain maintainers for music theory, linguistics, data, software, and listener research.
