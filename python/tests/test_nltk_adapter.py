from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from lape26.corpus.nltk_adapter import (
    _is_proper_noun_like,
    _is_taxonomic_name_like,
    configure_nltk_data_path,
    split_sentences,
    split_words,
    tokenize_raw_text,
)
from lape26.corpus.stats import compute_statistics
from lape26.corpus.tokens import tokenize_sentences

ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = ROOT / "python" / "tests" / "fixtures" / "tiny-corpus-sample.txt"
DOWNLOAD_DIR = ROOT / "data" / "raw" / "nltk_data"


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


def _wordnet_available() -> bool:
    try:
        configure_nltk_data_path(DOWNLOAD_DIR)
        from nltk.corpus import wordnet

        wordnet.synsets("test")
        return True
    except LookupError:
        return False


@unittest.skipUnless(
    _wordnet_available(), "requires local NLTK wordnet data (run `make corpus-setup` first) — CI is network-free"
)
class ProperNounAndTaxonomicFilterTests(unittest.TestCase):
    """Covers the stimulus-candidate proper-noun/taxonomic-name filters
    added after manual Task 23 review repeatedly found NLTK's `words`
    corpus and WordNet's neutral-polarity/orthographic candidates
    contaminated with surnames, place names, and taxonomic Latin
    classification terms (e.g. "reid", "arizona", "reptilia") — see
    data/manifests/stimulus-exclusions.yaml for the manually-found cases
    this is meant to catch automatically going forward.
    """

    def setUp(self) -> None:
        configure_nltk_data_path(DOWNLOAD_DIR)
        from nltk.corpus import wordnet

        self.wordnet = wordnet

    def _synsets(self, word: str) -> list:
        return self.wordnet.synsets(word)

    def test_proper_nouns_are_filtered(self) -> None:
        for word in ("arizona", "jerusalem", "reid", "yellowknife", "china", "turkey", "victor"):
            with self.subTest(word=word):
                self.assertTrue(_is_proper_noun_like(self._synsets(word)), f"{word!r} should be proper-noun-like")

    def test_taxonomic_names_are_filtered(self) -> None:
        for word in ("reptilia", "triopidae", "crassula", "myroxylon", "periploca", "syzygium", "erethizontidae"):
            with self.subTest(word=word):
                self.assertTrue(
                    _is_taxonomic_name_like(self._synsets(word)), f"{word!r} should be taxonomic-name-like"
                )

    def test_ordinary_species_common_nouns_are_not_taxonomic(self) -> None:
        # Species common nouns route through animal/organism/vertebrate,
        # not through genus/family/class-level classification lemmas.
        for word in ("manatee", "natterjack", "skunk"):
            with self.subTest(word=word):
                self.assertFalse(_is_taxonomic_name_like(self._synsets(word)), f"{word!r} should not be taxonomic")

    def test_most_ordinary_words_are_not_flagged_proper_noun(self) -> None:
        for word in ("happy", "sad", "warm", "joy", "honor", "faith", "flank", "cheese", "mossy"):
            with self.subTest(word=word):
                self.assertFalse(_is_proper_noun_like(self._synsets(word)), f"{word!r} should not be proper-noun-like")

    def test_known_accepted_false_positives_documented(self) -> None:
        # "love"/"skunk" each carry one incidental noun.person WordNet
        # sense (a term of endearment; an informal insult) alongside their
        # dominant senses. The proper-noun filter checks ANY sense rather
        # than the dominant one, so it also excludes these from stimulus
        # candidacy. This is an accepted, deliberate trade-off (see
        # _is_proper_noun_like's docstring): manual review remains the
        # real safety net, so under-filtering (missing a genuine proper
        # noun) is worse than over-filtering (losing a good word from
        # candidacy). This test exists so that trade-off is visible and
        # intentional, not a silent surprise if WordNet data changes.
        for word in ("love", "skunk"):
            with self.subTest(word=word):
                self.assertTrue(_is_proper_noun_like(self._synsets(word)))


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
