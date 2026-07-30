from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator

from lape26.core import encode_text
from lape26.corpus.controls import (
    POOL_MIDI,
    build_control_mapping_document,
    generate_circle_of_fifths_control,
    generate_frequency_ranked_control,
    generate_random_seed_control,
    generate_sequential_chromatic_control,
    midi_to_pitch_name,
)
from lape26.corpus.provenance import NORMALIZATION_PROFILE_ID, build_provenance_block

ROOT = Path(__file__).resolve().parents[2]
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _assert_valid_control(test_case: unittest.TestCase, assignment: dict[str, int]) -> None:
    test_case.assertEqual(set(assignment), set(LETTERS))
    test_case.assertEqual(len(set(assignment.values())), 26)
    for midi in assignment.values():
        test_case.assertIn(midi, POOL_MIDI)


class MidiToPitchNameTests(unittest.TestCase):
    def test_known_values_match_canonical_mapping(self) -> None:
        self.assertEqual(midi_to_pitch_name(60), "C4")
        self.assertEqual(midi_to_pitch_name(69), "A4")
        self.assertEqual(midi_to_pitch_name(48), "C3")
        self.assertEqual(midi_to_pitch_name(73), "C#5")


class SequentialChromaticTests(unittest.TestCase):
    def test_ascends_alphabet_to_pitch(self) -> None:
        result = generate_sequential_chromatic_control(POOL_MIDI)
        _assert_valid_control(self, result)
        self.assertEqual(result["A"], 48)
        self.assertEqual(result["Z"], 73)


class FrequencyRankedTests(unittest.TestCase):
    def test_most_frequent_letter_gets_pitch_nearest_center(self) -> None:
        result = generate_frequency_ranked_control({"Z": 100}, POOL_MIDI)
        _assert_valid_control(self, result)
        self.assertEqual(result["Z"], 60)


class CircleOfFifthsTests(unittest.TestCase):
    def test_produces_valid_bijection(self) -> None:
        result = generate_circle_of_fifths_control(POOL_MIDI)
        _assert_valid_control(self, result)

    def test_first_two_letters_follow_fifths_where_slots_available(self) -> None:
        result = generate_circle_of_fifths_control(POOL_MIDI)
        self.assertEqual(result["A"], 48)
        self.assertEqual(result["B"], 55)


class RandomSeedControlTests(unittest.TestCase):
    def test_deterministic_given_same_seed(self) -> None:
        first = generate_random_seed_control(POOL_MIDI, seed=26)
        second = generate_random_seed_control(POOL_MIDI, seed=26)
        self.assertEqual(first, second)
        _assert_valid_control(self, first)

    def test_different_seed_differs(self) -> None:
        a = generate_random_seed_control(POOL_MIDI, seed=26)
        b = generate_random_seed_control(POOL_MIDI, seed=27)
        self.assertNotEqual(a, b)


def _provenance() -> dict[str, object]:
    return build_provenance_block(
        pipeline_source_paths=["python/lape26/corpus/controls.py"],
        input_data_paths=[],
    )


class BuildControlMappingDocumentTests(unittest.TestCase):
    def test_document_validates_against_schema_and_includes_normalization_profile(self) -> None:
        schema = json.loads((ROOT / "data" / "schemas" / "control-mapping.schema.json").read_text(encoding="utf-8"))
        validator = Draft202012Validator(schema)
        assignment = generate_sequential_chromatic_control(POOL_MIDI)
        document = build_control_mapping_document(
            control_id="sequential-chromatic-v0.1",
            control_type="sequential-chromatic",
            generation_method="alphabetical-ascending-chromatic",
            assignment=assignment,
            seed=None,
            source_partition=None,
            provenance=_provenance(),
        )
        validator.validate(document)
        self.assertEqual(document["normalizationProfile"], NORMALIZATION_PROFILE_ID)

    def test_rejects_incomplete_assignment(self) -> None:
        with self.assertRaises(ValueError):
            build_control_mapping_document(
                control_id="broken",
                control_type="sequential-chromatic",
                generation_method="test",
                assignment={"A": 48},
                seed=None,
                source_partition=None,
                provenance={},
            )

    def test_all_four_control_types_are_encodable_by_encode_text(self) -> None:
        # This is the integration test that would have caught the missing
        # normalizationProfile field: schema validation alone did not
        # exercise the actual encode_text() code path.
        generators = {
            "sequential-chromatic": generate_sequential_chromatic_control(POOL_MIDI),
            "frequency-ranked": generate_frequency_ranked_control({"E": 10}, POOL_MIDI),
            "circle-of-fifths": generate_circle_of_fifths_control(POOL_MIDI),
            "random-seed": generate_random_seed_control(POOL_MIDI, seed=26),
        }
        with tempfile.TemporaryDirectory() as tmp:
            for control_type, assignment in generators.items():
                document = build_control_mapping_document(
                    control_id=f"{control_type}-v0.1",
                    control_type=control_type,
                    generation_method="test",
                    assignment=assignment,
                    seed=26 if control_type == "random-seed" else None,
                    source_partition=None,
                    provenance=_provenance(),
                )
                path = Path(tmp) / f"{control_type}.json"
                path.write_text(json.dumps(document), encoding="utf-8")
                events = encode_text("HAMMER", mapping_path=path)
                self.assertEqual(len(events), 6)


if __name__ == "__main__":
    unittest.main()
