# Analysis Metrics v0.1

All metrics are descriptive or heuristic. They do not establish objective beauty or emotional truth.

## Canonical descriptive metrics

### Register center

Arithmetic mean of MIDI pitches, converted to frequency only after averaging MIDI values.

### Pitch span

Maximum MIDI pitch minus minimum MIDI pitch.

### Interval contour

Signed adjacent MIDI differences.

### Directional balance

```text
sum(intervals) / (sum(abs(intervals)) + epsilon)
```

### Repetition index

Proportion of adjacent intervals equal to zero.

## Experimental heuristic metrics

- `adjacent_consonance_v0.1`
- `tonal_fit_v0.1`
- `melodic_flow_v0.1`
- `chord_consonance_v0.1`
- `chord_clarity_v0.1`
- `musicality_composite_v0.1`

Every implementation must expose the metric version and contributing components. Composite scores should be presented with explanations, not as authoritative judgments.
