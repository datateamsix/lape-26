from __future__ import annotations

import json
import math
import unicodedata
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

DEFAULT_MAPPING_PATH = Path(__file__).resolve().parents[2] / "mappings" / "lape-26-en-general-v0.1.json"


@dataclass(frozen=True)
class LetterEvent:
    sourceCharacter: str
    normalizedCharacter: str
    sourceIndex: int
    normalizedIndex: int
    pitch: str
    midi: int
    frequencyHz: float
    mappingVersion: str
    normalizationProfile: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def midi_to_frequency(midi: int, reference_hz: float = 440.0) -> float:
    if not 0 <= midi <= 127:
        raise ValueError("MIDI pitch must be between 0 and 127")
    return reference_hz * (2.0 ** ((midi - 69) / 12.0))


def normalize_text(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed.upper() if "A" <= ch <= "Z")


def load_mapping(path: str | Path = DEFAULT_MAPPING_PATH) -> dict[str, Any]:
    mapping_path = Path(path)
    data = json.loads(mapping_path.read_text(encoding="utf-8"))
    letters = data.get("letters", {})
    if set(letters) != set("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
        raise ValueError("Mapping must define exactly A-Z")
    midi_values = [entry["midi"] for entry in letters.values()]
    if len(midi_values) != len(set(midi_values)):
        raise ValueError("Mapping MIDI assignments must be unique")
    return data


def encode_text(text: str, mapping_path: str | Path = DEFAULT_MAPPING_PATH) -> list[dict[str, Any]]:
    mapping = load_mapping(mapping_path)
    normalized = normalize_text(text)
    events: list[dict[str, Any]] = []
    normalized_index = 0
    for source_index, source_character in enumerate(unicodedata.normalize("NFKD", text)):
        normalized_character = source_character.upper()
        if not ("A" <= normalized_character <= "Z"):
            continue
        entry = mapping["letters"][normalized_character]
        event = LetterEvent(
            sourceCharacter=source_character,
            normalizedCharacter=normalized_character,
            sourceIndex=source_index,
            normalizedIndex=normalized_index,
            pitch=entry["pitch"],
            midi=int(entry["midi"]),
            frequencyHz=midi_to_frequency(int(entry["midi"])),
            mappingVersion=mapping["mappingId"],
            normalizationProfile=mapping["normalizationProfile"],
        )
        events.append(event.to_dict())
        normalized_index += 1

    if "".join(event["normalizedCharacter"] for event in events) != normalized:
        raise RuntimeError("Normalization and encoding paths diverged")
    return events
