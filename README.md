# LAPE-26

**LAPE-26 is an open-source creative research project intended to explore the musicality of language.**

It asks a simple but unusual question:

> What happens when the 26 letters of written English are assigned to 26 unique pitches from the existing 12-tone chromatic system, and words, names, and sentences are rendered as reproducible musical structures?

LAPE-26 provides a deterministic musical alphabet, reference encoders, a browser-based Word Explorer, analysis metrics, and a research framework for comparing candidate mappings through corpus analysis and blind listening studies.

## Project status

**Version:** `0.1.0-experimental`  
**Mapping:** `lape-26-en-general-v0.1`  
**Status:** Creative research prototype—not a frozen or scientifically proven natural alphabet.

The current mapping is a reasoned experimental hypothesis. It is expected to be tested against simple controls, language corpora, names, emotional word lists, interval distributions, listening fatigue, and listener preferences before any `1.0` mapping is proposed.

## What LAPE-26 is

- A deterministic text-to-pitch encoding system
- A creative instrument for exploring how words and names sound under one fixed musical alphabet
- An open research framework for evaluating and refining that alphabet
- A reference implementation in TypeScript and Python
- A foundation for melody, chord, sentence, phonemic, and hybrid language-music experiments

## What LAPE-26 is not

- A claim that words possess objectively correct natural melodies
- A replacement for phonetics, music theory, linguistics, or psychoacoustics
- A black-box generative music model
- An AI system that changes letter assignments from one performance to another
- A claim that musical preference or emotional meaning can be reduced to one score

## Core principle

The mapping may be optimized during research, but a released mapping is fixed and versioned:

```text
same text
+ same normalization profile
+ same mapping version
+ same rendering settings
= same canonical note events
```

## Current mapping

The experimental mapping assigns every letter to one unique MIDI pitch from C3 through C#5 using 12-tone equal temperament with A4 = 440 Hz.

| Letter | Pitch | MIDI | Frequency |
|---|---:|---:|---:|
| A | C4 | 60 | 261.626 Hz |
| B | G#4 | 68 | 415.305 Hz |
| C | F4 | 65 | 349.228 Hz |
| D | C#4 | 61 | 277.183 Hz |
| E | E4 | 64 | 329.628 Hz |
| F | C3 | 48 | 130.813 Hz |
| G | F#4 | 66 | 369.994 Hz |
| H | A4 | 69 | 440.000 Hz |
| I | G4 | 67 | 391.995 Hz |
| J | D#3 | 51 | 155.563 Hz |
| K | C5 | 72 | 523.251 Hz |
| L | G#3 | 56 | 207.652 Hz |
| M | D#4 | 63 | 311.127 Hz |
| N | A#3 | 58 | 233.082 Hz |
| O | G3 | 55 | 195.998 Hz |
| P | F#3 | 54 | 184.997 Hz |
| Q | E3 | 52 | 164.814 Hz |
| R | A3 | 57 | 220.000 Hz |
| S | B3 | 59 | 246.942 Hz |
| T | D4 | 62 | 293.665 Hz |
| U | D3 | 50 | 146.832 Hz |
| V | C#3 | 49 | 138.591 Hz |
| W | F3 | 53 | 174.614 Hz |
| X | C#5 | 73 | 554.365 Hz |
| Y | B4 | 71 | 493.883 Hz |
| Z | A#4 | 70 | 466.164 Hz |


The canonical machine-readable file is:

```text
mappings/lape-26-en-general-v0.1.json
```

## Try the Word Explorer

Open:

```text
apps/word-explorer/index.html
```

The current prototype supports:

- Words, names, and short phrases
- Sequential melody playback
- Simultaneous “Together (Chord)” playback
- Note and frequency inspection
- Melodic contour rendering
- Register, span, direction, and interval analysis
- Transparent heuristic musicality analysis

The prototype currently loads Tone.js from a CDN, so audio playback requires an internet connection. The deterministic note mapping and displayed analysis remain local to the page.

## Repository map

```text
mappings/           Canonical and control mappings
specs/              Versioned methodology and behavioral specifications
packages/core-ts/   TypeScript reference encoder
packages/audio-web/ Browser audio integration boundary
python/lape26/      Python reference encoder and research utilities
apps/word-explorer/ Public creative exploration tool
apps/listener-study/Controlled listening-study starter
scripts/            Mapping validation and reporting utilities
tests/              Shared golden vectors and regression fixtures
data/               Data manifests, tiny redistributable fixtures, and licensing notes
experiments/        Reproducible experiment parameters and reports
.github/            CI, issue templates, and contribution workflow
```

## Quick start

### Run all dependency-free checks

```bash
make test
```

Or run them separately:

```bash
python3 scripts/validate_mapping.py
PYTHONPATH=python python3 -m unittest discover -s python/tests -v
node --experimental-strip-types --test packages/core-ts/test/core.test.ts
node --experimental-strip-types scripts/check_cross_runtime.ts
```

The initial checks use only Python and Node.js standard-library functionality.

### Encode text with Python

```bash
PYTHONPATH=python python3 -m lape26.cli encode "HAMMER"
```

### Encode text with TypeScript

```bash
node --experimental-strip-types packages/core-ts/src/cli.ts "HAMMER"
```

## Research roadmap

1. **Phase 0 — Open-source foundation**  
   Publish the mapping, specifications, reference encoders, Word Explorer, governance, tests, and CI.

2. **Phase 1 — Deterministic core**  
   Freeze schemas, normalization behavior, event generation, and cross-runtime parity.

3. **Phase 2 — Evaluation framework**  
   Compare the current mapping with sequential, frequency-ranked, circle-of-fifths, and random controls.

4. **Phase 3 — Candidate optimization**  
   Use transparent local-swap and seeded search methods to produce candidate fixed alphabets.

5. **Phase 4 — Blind listening studies**  
   Test musicality, word fit, distinctiveness, repeat-listening preference, and fatigue.

6. **Phase 5 — Preference modeling**  
   Fit small interpretable ranking models to listener judgments and use them to refine offline objective weights.

7. **Phase 6 — LAPE-26 v1.0 proposal**  
   Publish a reproducible mapping proposal with methodology, comparisons, listener evidence, and known limitations.

See [`ROADMAP.md`](ROADMAP.md) for acceptance criteria.

## Scientific and creative boundary

The frequencies are defined by standard 12-tone equal temperament:

```text
f(m) = 440 × 2^((m - 69) / 12)
```

The assignment of letters to those pitches is a creative and empirical design problem. Physics determines the frequency of a selected pitch; it does not determine which pitch belongs to the letter `A`, `B`, or `Z`.

Analysis terms such as *musicality*, *tension*, *flow*, or *word fit* must therefore remain:

- explicitly defined;
- versioned;
- reproducible;
- open to criticism;
- separated from claims of objective emotional truth.

## Contributing

Contributions are welcome from musicians, composers, linguists, creative coders, psychoacoustics researchers, statisticians, accessibility specialists, and curious listeners.

Start with:

- [`CONTRIBUTING.md`](CONTRIBUTING.md)
- [`GOVERNANCE.md`](GOVERNANCE.md)
- [`specs/`](specs/)

Changes to a canonical mapping require a versioned LAPE Mapping Proposal and reproducible evaluation evidence. Existing mappings are never silently overwritten.

## Data and licensing

Project code, mapping schemas, and original specifications are licensed under Apache-2.0. External datasets retain their own licenses and must be described in a manifest. Copyrighted lyric text must not be committed unless redistribution is explicitly permitted.

See [`DATA_SOURCES.md`](DATA_SOURCES.md) and [`data/README.md`](data/README.md).

## Citation

Citation metadata is provided in [`CITATION.cff`](CITATION.cff).

## License

Apache License 2.0. See [`LICENSE`](LICENSE).
