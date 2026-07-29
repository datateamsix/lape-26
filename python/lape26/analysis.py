from __future__ import annotations

from math import inf
from typing import Iterable


def midi_sequence(events: Iterable[dict]) -> list[int]:
    return [int(event["midi"]) for event in events]


def intervals(midi: list[int]) -> list[int]:
    return [b - a for a, b in zip(midi, midi[1:])]


def register_center(midi: list[int]) -> float | None:
    return sum(midi) / len(midi) if midi else None


def pitch_span(midi: list[int]) -> int:
    return max(midi) - min(midi) if midi else 0


def directional_balance(midi: list[int]) -> float:
    movement = intervals(midi)
    denominator = sum(abs(value) for value in movement)
    return sum(movement) / denominator if denominator else 0.0


def repetition_index(midi: list[int]) -> float:
    movement = intervals(midi)
    return sum(value == 0 for value in movement) / len(movement) if movement else 0.0


def summarize(events: list[dict]) -> dict:
    midi = midi_sequence(events)
    return {
        "noteCount": len(midi),
        "midi": midi,
        "intervals": intervals(midi),
        "registerCenterMidi": register_center(midi),
        "pitchSpanSemitones": pitch_span(midi),
        "directionalBalance": directional_balance(midi),
        "repetitionIndex": repetition_index(midi),
    }
