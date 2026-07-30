from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

from lape26.corpus.stimulus import (
    OrthographicCandidate,
    WordCandidate,
    has_rare_letters,
    has_repeated_letters,
    is_consonant_heavy,
    is_vowel_heavy,
    length_band,
    select_core_set,
    select_orthographic_challenge_set,
    stable_seed,
)

ROOT = Path(__file__).resolve().parents[2]
_LENGTH_BY_BAND = {"short": 4, "medium": 7, "long": 10}
_POS_CYCLE = ["noun", "verb", "adjective", "adverb"]
_BANDS = ("short", "medium", "long")
_POLARITIES = ("positive", "negative", "neutral")


def _make_candidates(band: str, polarity: str, count: int) -> list[WordCandidate]:
    return [
        WordCandidate(
            word=f"{band}{polarity}{i}".upper(),
            length=_LENGTH_BY_BAND[band],
            partOfSpeech=_POS_CYCLE[i % len(_POS_CYCLE)],
            polarity=polarity,
            vaderCompound=0.5 if polarity == "positive" else (-0.5 if polarity == "negative" else 0.0),
            sourceDataset="opinion-lexicon" if polarity != "neutral" else "words",
            stemKey=f"{band}{polarity}{i}".upper(),
        )
        for i in range(count)
    ]


def _full_candidate_pool(count_per_cell: int) -> list[WordCandidate]:
    candidates: list[WordCandidate] = []
    for band in _BANDS:
        for polarity in _POLARITIES:
            candidates.extend(_make_candidates(band, polarity, count_per_cell))
    return candidates


class StableSeedTests(unittest.TestCase):
    def test_deterministic_within_process(self) -> None:
        self.assertEqual(stable_seed(26, "short", "positive"), stable_seed(26, "short", "positive"))

    def test_different_parts_give_different_seeds(self) -> None:
        self.assertNotEqual(stable_seed(26, "short", "positive"), stable_seed(26, "short", "negative"))

    def test_stable_across_separate_interpreter_processes_regardless_of_pythonhashseed(self) -> None:
        # This is the release-blocking case: hash() is salted per-process
        # in CPython, so a naive hash()-based seed would differ here even
        # though stable_seed must not.
        script = (
            f"import sys; sys.path.insert(0, {str(ROOT / 'python')!r}); "
            "from lape26.corpus.stimulus import stable_seed; "
            "print(stable_seed(26, 'short', 'positive'))"
        )
        results = []
        for hash_seed in ("1", "42"):
            completed = subprocess.run(
                [sys.executable, "-c", script],
                env={**os.environ, "PYTHONHASHSEED": hash_seed},
                capture_output=True,
                text=True,
                check=True,
            )
            results.append(completed.stdout.strip())
        self.assertEqual(results[0], results[1])


class LengthBandTests(unittest.TestCase):
    def test_bands(self) -> None:
        self.assertEqual(length_band(3), "short")
        self.assertEqual(length_band(5), "short")
        self.assertEqual(length_band(6), "medium")
        self.assertEqual(length_band(8), "medium")
        self.assertEqual(length_band(9), "long")
        self.assertEqual(length_band(12), "long")
        self.assertIsNone(length_band(2))
        self.assertIsNone(length_band(13))


class SelectCoreSetTests(unittest.TestCase):
    def test_selects_exactly_108_with_12_per_cell(self) -> None:
        candidates = _full_candidate_pool(count_per_cell=12)
        selected = select_core_set(candidates, seed=26, used_words=set(), used_stems=set())
        self.assertEqual(len(selected), 108)
        for band in _BANDS:
            for polarity in _POLARITIES:
                cell_count = sum(
                    1 for c in selected if length_band(c.length) == band and c.polarity == polarity
                )
                self.assertEqual(cell_count, 12, f"{band}/{polarity}")

    def test_selects_12_even_with_surplus_candidates(self) -> None:
        candidates = _full_candidate_pool(count_per_cell=20)
        selected = select_core_set(candidates, seed=26, used_words=set(), used_stems=set())
        self.assertEqual(len(selected), 108)

    def test_raises_when_a_cell_is_short(self) -> None:
        candidates = _full_candidate_pool(count_per_cell=12)
        candidates = [c for c in candidates if not (c.word.startswith("SHORTPOSITIVE") and c.word != "SHORTPOSITIVE0")]
        with self.assertRaises(ValueError) as ctx:
            select_core_set(candidates, seed=26, used_words=set(), used_stems=set())
        self.assertIn("short", str(ctx.exception))
        self.assertIn("positive", str(ctx.exception))

    def test_deterministic_given_same_seed(self) -> None:
        candidates = _full_candidate_pool(count_per_cell=15)
        first = select_core_set(candidates, seed=26, used_words=set(), used_stems=set())
        second = select_core_set(candidates, seed=26, used_words=set(), used_stems=set())
        self.assertEqual([c.word for c in first], [c.word for c in second])

    def test_populates_used_words_and_stems_so_a_second_call_excludes_them(self) -> None:
        candidates = _full_candidate_pool(count_per_cell=12)  # exactly enough, no surplus
        used_words: set[str] = set()
        used_stems: set[str] = set()
        first = select_core_set(candidates, seed=26, used_words=used_words, used_stems=used_stems)
        self.assertEqual(len(used_words), 108)
        self.assertEqual(len(used_stems), 108)
        # A second call against the same (now-exhausted) pool must fail —
        # proves used_words/used_stems are actually being respected.
        with self.assertRaises(ValueError):
            select_core_set(candidates, seed=99, used_words=used_words, used_stems=used_stems)


class OrthographicClassificationTests(unittest.TestCase):
    def test_has_repeated_letters(self) -> None:
        self.assertTrue(has_repeated_letters("BALLOON"))
        self.assertFalse(has_repeated_letters("CAT"))

    def test_has_rare_letters(self) -> None:
        self.assertTrue(has_rare_letters("JAZZ"))
        self.assertTrue(has_rare_letters("QUIZ"))
        self.assertFalse(has_rare_letters("CAT"))

    def test_is_vowel_heavy(self) -> None:
        self.assertTrue(is_vowel_heavy("AI"))
        self.assertFalse(is_vowel_heavy("STRENGTH"))

    def test_is_consonant_heavy(self) -> None:
        self.assertTrue(is_consonant_heavy("STRENGTH"))
        self.assertFalse(is_consonant_heavy("AI"))


class SelectOrthographicChallengeSetTests(unittest.TestCase):
    def _pool(self) -> dict[str, list[str]]:
        return {
            "repeated-letters": ["BALLOON", "ANNOUNCE", "COMMITTEE", "MISSISSIPPI"],
            "rare-letters": ["JAZZ", "QUIZ", "WALTZ", "XYLOPHONE"],
            "vowel-heavy": ["AI", "AREA", "IDEA", "QUEUE"],
            "consonant-heavy": ["STRENGTH", "RHYTHM", "GLYPH", "NYMPH"],
        }

    def test_selects_3_per_category_12_total(self) -> None:
        result = select_orthographic_challenge_set(self._pool(), seed=26, used_words=set())
        self.assertEqual(len(result), 12)
        for category in ("repeated-letters", "rare-letters", "vowel-heavy", "consonant-heavy"):
            self.assertEqual(sum(1 for r in result if r.category == category), 3)

    def test_no_word_appears_in_two_categories(self) -> None:
        pool = self._pool()
        # Deliberately overlapping: JAZZ has a repeated-ish letter pattern
        # AND a rare letter — add it to both pools with enough surplus
        # elsewhere that exclusion doesn't starve either category.
        pool["repeated-letters"].append("JAZZ")
        result = select_orthographic_challenge_set(pool, seed=26, used_words=set())
        words = [r.word for r in result]
        self.assertEqual(len(words), len(set(words)))

    def test_raises_when_category_is_short(self) -> None:
        pool = self._pool()
        pool["rare-letters"] = ["JAZZ"]
        with self.assertRaises(ValueError) as ctx:
            select_orthographic_challenge_set(pool, seed=26, used_words=set())
        self.assertIn("rare-letters", str(ctx.exception))

    def test_deterministic_given_same_seed(self) -> None:
        first = select_orthographic_challenge_set(self._pool(), seed=26, used_words=set())
        second = select_orthographic_challenge_set(self._pool(), seed=26, used_words=set())
        self.assertEqual([r.word for r in first], [r.word for r in second])

    def test_excludes_words_already_used_by_core_set(self) -> None:
        pool = self._pool()
        pre_used = {"BALLOON"}  # pretend the core set already took this word
        result = select_orthographic_challenge_set(pool, seed=26, used_words=pre_used)
        self.assertNotIn("BALLOON", [r.word for r in result])


if __name__ == "__main__":
    unittest.main()
