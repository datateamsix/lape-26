from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "mappings" / "lape-26-en-general-v0.1.json"


def expected_frequency(midi: int) -> float:
    return 440.0 * 2.0 ** ((midi - 69) / 12.0)


def main() -> None:
    raw = PATH.read_bytes()
    mapping = json.loads(raw)
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    letters = mapping["letters"]

    assert "".join(sorted(letters)) == alphabet, "mapping must contain exactly A-Z"
    midi = [letters[letter]["midi"] for letter in alphabet]
    assert len(set(midi)) == 26, "MIDI assignments must be unique"
    assert sorted(midi) == list(range(48, 74)), "mapping must use every MIDI pitch 48 through 73"

    for letter in alphabet:
        entry = letters[letter]
        expected = expected_frequency(entry["midi"])
        assert math.isclose(entry["frequencyHz"], expected, rel_tol=0, abs_tol=0.000001), (
            f"{letter} frequency mismatch: stored={entry['frequencyHz']} expected={expected}"
        )

    digest = hashlib.sha256(raw).hexdigest()
    checksum_path = PATH.with_suffix(PATH.suffix + ".sha256")
    checksum_path.write_text(f"{digest}  {PATH.name}\n", encoding="utf-8")
    print(f"Mapping valid: {mapping['mappingId']}")
    print(f"SHA-256: {digest}")


if __name__ == "__main__":
    main()
