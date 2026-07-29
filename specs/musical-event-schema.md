# Musical Event Schema v0.1

A canonical letter event contains:

```json
{
  "sourceCharacter": "H",
  "normalizedCharacter": "H",
  "sourceIndex": 0,
  "normalizedIndex": 0,
  "pitch": "A4",
  "midi": 69,
  "frequencyHz": 440.0,
  "startBeat": 0,
  "durationBeats": 1,
  "velocity": 0.8,
  "mappingVersion": "lape-26-en-general-v0.1",
  "normalizationProfile": "lape-text-normalization-v0.1"
}
```

Pitch identity is canonical. Timing, velocity, articulation, instrument, chord voicing, and effects are rendering parameters unless a later specification explicitly promotes them into the canonical encoding.
