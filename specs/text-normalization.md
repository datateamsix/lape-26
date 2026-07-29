# Text Normalization Profile v0.1

Profile ID: `lape-text-normalization-v0.1`

## Canonical pitch normalization

1. Apply Unicode compatibility decomposition where the runtime supports it.
2. Convert text to uppercase.
3. Keep ASCII letters `A` through `Z` as pitch-bearing symbols.
4. Preserve source indices for traceability.
5. Treat whitespace as phrase metadata and optional rests.
6. Preserve punctuation as metadata but do not assign it a pitch in v0.1.
7. Ignore unsupported symbols for pitch generation.

## Examples

```text
"Hammer"       → HAMMER
"co-operate"   → COOPERATE
"don't"        → DONT
"café"         → CAFE when diacritic decomposition is available
"26 tones"     → TONES
```

The normalization profile is versioned. Future number, punctuation, phonemic, and multilingual profiles must use new IDs rather than changing this behavior silently.
