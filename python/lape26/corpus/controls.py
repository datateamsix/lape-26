from __future__ import annotations

import random

from ..core import midi_to_frequency
from .provenance import NORMALIZATION_PROFILE_ID

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
POOL_LOWEST_MIDI = 48
POOL_HIGHEST_MIDI = 73
POOL_MIDI = list(range(POOL_LOWEST_MIDI, POOL_HIGHEST_MIDI + 1))

_PITCH_CLASS_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def midi_to_pitch_name(midi: int) -> str:
    octave = midi // 12 - 1
    name = _PITCH_CLASS_NAMES[midi % 12]
    return f"{name}{octave}"


def generate_sequential_chromatic_control(pool_midi: list[int]) -> dict[str, int]:
    ordered_pool = sorted(pool_midi)
    return dict(zip(LETTERS, ordered_pool))


def generate_frequency_ranked_control(
    letter_frequency: dict[str, int],
    pool_midi: list[int],
) -> dict[str, int]:
    ranked_letters = sorted(LETTERS, key=lambda letter: (-letter_frequency.get(letter, 0), letter))
    center = (min(pool_midi) + max(pool_midi)) / 2
    ranked_pitches = sorted(pool_midi, key=lambda midi: (abs(midi - center), midi))
    return dict(zip(ranked_letters, ranked_pitches))


def _circle_of_fifths_pitch_classes(letter_count: int) -> list[int]:
    return [(i * 7) % 12 for i in range(letter_count)]


def generate_circle_of_fifths_control(pool_midi: list[int]) -> dict[str, int]:
    """Advance pitch class by a perfect fifth (7 semitones) per letter,
    starting at C (pitch class 0). The fixed 26-note pool (2 octaves + 2
    semitones) does not contain 3 instances of every pitch class visited 3
    times by the 26-letter fifths cycle, so when a letter's ideal pitch
    class has no unused pool slot left, it takes the closest unused MIDI
    value instead (ties broken by lower value). This keeps the result a
    valid bijection onto the fixed pool while remaining fully deterministic.
    """
    pitch_classes = _circle_of_fifths_pitch_classes(len(pool_midi))
    available = set(pool_midi)
    assignment: dict[str, int] = {}
    for letter, pitch_class in zip(LETTERS, pitch_classes):
        candidates = sorted(midi for midi in available if midi % 12 == pitch_class)
        if candidates:
            chosen = candidates[0]
        else:
            reference = min(pool_midi) + pitch_class
            chosen = min(available, key=lambda midi: (abs(midi - reference), midi))
        assignment[letter] = chosen
        available.discard(chosen)
    return assignment


def generate_random_seed_control(pool_midi: list[int], seed: int) -> dict[str, int]:
    rng = random.Random(seed)
    shuffled = pool_midi[:]
    rng.shuffle(shuffled)
    return dict(zip(LETTERS, shuffled))


def build_control_mapping_document(
    *,
    control_id: str,
    control_type: str,
    generation_method: str,
    assignment: dict[str, int],
    seed: int | None,
    source_partition: str | None,
    provenance: dict[str, object],
) -> dict[str, object]:
    if set(assignment) != set(LETTERS):
        raise ValueError("Control assignment must cover exactly A-Z")
    if len(set(assignment.values())) != 26:
        raise ValueError("Control assignment must use 26 unique MIDI values")

    letters = {
        letter: {
            "pitch": midi_to_pitch_name(midi),
            "midi": midi,
            "frequencyHz": round(midi_to_frequency(midi), 6),
        }
        for letter, midi in assignment.items()
    }
    return {
        "$schema": "../../data/schemas/control-mapping.schema.json",
        "mappingId": control_id,
        "version": "0.1.0-experimental",
        "status": "experimental",
        "controlType": control_type,
        "generationMethod": generation_method,
        "seed": seed,
        "sourcePartition": source_partition,
        "alphabet": LETTERS,
        "normalizationProfile": NORMALIZATION_PROFILE_ID,
        "tuning": {
            "system": "12-TET",
            "referencePitch": "A4",
            "referenceMidi": 69,
            "referenceFrequencyHz": 440.0,
            "frequencyFormula": "f(m)=440*2^((m-69)/12)",
        },
        "range": {
            "lowestPitch": midi_to_pitch_name(POOL_LOWEST_MIDI),
            "lowestMidi": POOL_LOWEST_MIDI,
            "highestPitch": midi_to_pitch_name(POOL_HIGHEST_MIDI),
            "highestMidi": POOL_HIGHEST_MIDI,
        },
        "letters": letters,
        "provenance": provenance,
    }
