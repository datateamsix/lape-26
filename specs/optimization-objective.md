# Mapping Optimization Objective — Draft v0.1

The mapping problem is a constrained permutation search over 26 unique MIDI pitches.

Candidate mappings may be evaluated using versioned components such as:

```text
language transition cost
interval distribution cost
register comfort cost
tonal-fit cost
word-identity cost
fatigue proxy
listener preference prediction
```

The objective weights are experimental parameters. Search may use local swaps, simulated annealing, or other reproducible algorithms with recorded seeds.

Optimization happens offline. A published mapping remains a fixed JSON asset and never changes dynamically during user playback.
