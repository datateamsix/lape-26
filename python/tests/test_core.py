from __future__ import annotations

import json
import math
import unittest
from pathlib import Path

from lape26.analysis import summarize
from lape26.core import DEFAULT_MAPPING_PATH, encode_text, load_mapping, midi_to_frequency, normalize_text

ROOT = Path(__file__).resolve().parents[2]


class CoreTests(unittest.TestCase):
    def test_mapping_has_unique_a_to_z_assignments(self) -> None:
        mapping = load_mapping()
        self.assertEqual(set(mapping["letters"]), set("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
        midi = [entry["midi"] for entry in mapping["letters"].values()]
        self.assertEqual(len(midi), 26)
        self.assertEqual(len(set(midi)), 26)
        self.assertEqual(min(midi), 48)
        self.assertEqual(max(midi), 73)

    def test_reference_frequency(self) -> None:
        self.assertAlmostEqual(midi_to_frequency(69), 440.0, places=9)
        self.assertAlmostEqual(midi_to_frequency(60), 261.625565, places=5)

    def test_normalization(self) -> None:
        self.assertEqual(normalize_text("Café, don't! 26"), "CAFEDONT")

    def test_hammer_golden_vector(self) -> None:
        events = encode_text("HAMMER")
        self.assertEqual([event["midi"] for event in events], [69, 60, 63, 63, 64, 57])
        self.assertEqual(summarize(events)["intervals"], [-9, 3, 0, 1, -7])

    def test_shared_golden_vectors(self) -> None:
        cases = json.loads((ROOT / "tests" / "golden-vectors.json").read_text(encoding="utf-8"))["cases"]
        for case in cases:
            with self.subTest(case=case["id"]):
                events = encode_text(case["input"])
                self.assertEqual([event["normalizedCharacter"] for event in events], list(case["normalized"]))
                self.assertEqual([event["midi"] for event in events], case["midi"])
                self.assertEqual(summarize(events)["intervals"], case["intervals"])


if __name__ == "__main__":
    unittest.main()
