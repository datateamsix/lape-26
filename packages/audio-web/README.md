# @lape-26/audio-web

This package boundary will translate canonical LAPE events into browser audio playback.

Phase 0 keeps the working Tone.js implementation inside `apps/word-explorer/index.html`. Phase 1 should extract:

- Transport scheduling
- Melody playback
- Simultaneous chord playback
- Voice limiting
- Duplicate-note reinforcement
- Stop and cleanup behavior

The audio layer must not own letter mappings or normalization.
