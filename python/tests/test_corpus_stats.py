from __future__ import annotations

import unittest

from lape26.corpus.stats import compute_statistics
from lape26.corpus.tokens import tokenize_sentences


class ComputeStatisticsTests(unittest.TestCase):
    def test_within_word_bigrams_single_word(self) -> None:
        stats = compute_statistics(tokenize_sentences("doc-one", [["cat"]]))
        self.assertEqual(stats.withinWordBigrams, {"AT": 1, "CA": 1})
        self.assertEqual(stats.characterFrequency, {"A": 1, "C": 1, "T": 1})
        self.assertEqual(stats.crossWordBoundaryBigrams, {})

    def test_cross_word_transition_not_counted_as_within_word(self) -> None:
        stats = compute_statistics(tokenize_sentences("doc-one", [["cat", "dog"]]))
        self.assertEqual(stats.withinWordBigrams, {"AT": 1, "CA": 1, "DO": 1, "OG": 1})
        self.assertEqual(stats.crossWordBoundaryBigrams, {"TD": 1})
        self.assertNotIn("TD", stats.withinWordBigrams)

    def test_word_and_sentence_boundary_letters(self) -> None:
        stats = compute_statistics(tokenize_sentences("doc-one", [["cat", "dog"]]))
        self.assertEqual(stats.wordInitialLetters, {"C": 1, "D": 1})
        self.assertEqual(stats.wordFinalLetters, {"G": 1, "T": 1})
        self.assertEqual(stats.sentenceInitialLetters, {"C": 1})
        self.assertEqual(stats.sentenceFinalLetters, {"G": 1})

    def test_word_and_sentence_length_stats(self) -> None:
        stats = compute_statistics(tokenize_sentences("doc-one", [["cat", "dog"], ["a"]]))
        self.assertEqual(stats.wordCount, 3)
        self.assertEqual(stats.sentenceCount, 2)
        self.assertEqual(stats.wordLengthStats, {"min": 1.0, "max": 3.0, "mean": 7 / 3, "median": 3.0})
        self.assertEqual(stats.sentenceLengthStats, {"min": 1.0, "max": 2.0, "mean": 1.5, "median": 1.5})

    def test_no_cross_word_bigram_across_sentence_boundary(self) -> None:
        stats = compute_statistics(tokenize_sentences("doc-one", [["cat"], ["dog"]]))
        self.assertEqual(stats.crossWordBoundaryBigrams, {})


if __name__ == "__main__":
    unittest.main()
