from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from lape26.corpus.nltk_adapter import (
    configure_nltk_data_path,
    split_sentences,
    split_words,
    tokenize_raw_text,
)
from lape26.corpus.stats import compute_statistics
from lape26.corpus.tokens import tokenize_sentences

ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = ROOT / "python" / "tests" / "fixtures" / "tiny-corpus-sample.txt"


class SplitSentencesTests(unittest.TestCase):
    def test_splits_on_terminal_punctuation(self) -> None:
        sentences = split_sentences("The cat sat. A dog ran!")
        self.assertEqual(sentences, ["The cat sat.", "A dog ran!"])

    def test_collapses_whitespace_and_blank_lines(self) -> None:
        sentences = split_sentences("One.\n\n\nTwo.")
        self.assertEqual(sentences, ["One.", "Two."])

    def test_empty_input_returns_empty_list(self) -> None:
        self.assertEqual(split_sentences(""), [])
        self.assertEqual(split_sentences("   \n  "), [])


class SplitWordsTests(unittest.TestCase):
    def test_extracts_alphabetic_tokens_only(self) -> None:
        self.assertEqual(split_words('"Yes," she said.'), ["Yes", "she", "said"])

    def test_keeps_apostrophes_within_words(self) -> None:
        self.assertEqual(split_words("don't stop"), ["don't", "stop"])


class TokenizeRawTextAndFixtureTests(unittest.TestCase):
    def test_fixture_file_exists_and_is_nonempty(self) -> None:
        self.assertTrue(FIXTURE_PATH.exists())
        self.assertGreater(len(FIXTURE_PATH.read_text(encoding="utf-8").strip()), 0)

    def test_fixture_tokenizes_into_multiple_sentences(self) -> None:
        raw_sentences = tokenize_raw_text(FIXTURE_PATH.read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(raw_sentences), 4)
        self.assertIn(["The", "cat", "sat", "on", "the", "mat"], raw_sentences)

    def test_fixture_flows_end_to_end_through_tokenize_and_stats(self) -> None:
        raw_sentences = tokenize_raw_text(FIXTURE_PATH.read_text(encoding="utf-8"))
        sentences = tokenize_sentences("tiny-corpus-sample", raw_sentences)
        stats = compute_statistics(sentences)
        self.assertGreater(stats.wordCount, 0)
        self.assertGreater(stats.sentenceCount, 0)
        self.assertIn("C", stats.characterFrequency)  # from "cat"


class ConfigureNltkDataPathTests(unittest.TestCase):
    def test_inserts_download_dir_into_nltk_data_path_once(self) -> None:
        import nltk

        with tempfile.TemporaryDirectory() as tmp:
            download_dir = Path(tmp) / "nltk_data"
            download_dir.mkdir()
            original_path = list(nltk.data.path)
            try:
                configure_nltk_data_path(download_dir)
                configure_nltk_data_path(download_dir)  # calling twice must not duplicate
                resolved = str(download_dir.resolve())
                self.assertEqual(nltk.data.path.count(resolved), 1)
            finally:
                nltk.data.path[:] = original_path


if __name__ == "__main__":
    unittest.main()
