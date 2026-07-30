from __future__ import annotations

import unittest

from lape26.corpus.tokens import tokenize_sentences


class TokenizeSentencesTests(unittest.TestCase):
    def test_marks_sentence_initial_and_final(self) -> None:
        sentences = tokenize_sentences("doc-one", [["The", "cat", "sat", "."]])
        self.assertEqual(len(sentences), 1)
        words = sentences[0].words
        self.assertEqual([w.normalizedText for w in words], ["THE", "CAT", "SAT"])
        self.assertTrue(words[0].isSentenceInitial)
        self.assertFalse(words[0].isSentenceFinal)
        self.assertTrue(words[-1].isSentenceFinal)
        self.assertFalse(words[-1].isSentenceInitial)

    def test_pure_punctuation_tokens_are_dropped(self) -> None:
        sentences = tokenize_sentences("doc-one", [["Hello", ",", "world", "!"]])
        self.assertEqual([w.normalizedText for w in sentences[0].words], ["HELLO", "WORLD"])

    def test_all_punctuation_sentence_is_dropped_entirely(self) -> None:
        sentences = tokenize_sentences("doc-one", [["..."], ["Real", "sentence", "."]])
        self.assertEqual(len(sentences), 1)
        self.assertEqual([w.normalizedText for w in sentences[0].words], ["REAL", "SENTENCE"])

    def test_single_word_sentence_is_both_initial_and_final(self) -> None:
        sentences = tokenize_sentences("doc-one", [["Stop", "."]])
        word = sentences[0].words[0]
        self.assertTrue(word.isSentenceInitial)
        self.assertTrue(word.isSentenceFinal)

    def test_sentence_index_is_preserved_across_dropped_sentences(self) -> None:
        sentences = tokenize_sentences("doc-one", [["..."], ["One", "."], ["Two", "."]])
        self.assertEqual([s.sentenceIndex for s in sentences], [1, 2])

    def test_document_id_is_attached_to_every_sentence_and_word(self) -> None:
        sentences = tokenize_sentences("doc-42", [["Hi", "."]])
        self.assertEqual(sentences[0].documentId, "doc-42")
        self.assertEqual(sentences[0].words[0].documentId, "doc-42")

    def test_source_token_index_reflects_pre_drop_position(self) -> None:
        # "Well" is index 0, "," is index 1 (dropped), "hello" is index 2
        sentences = tokenize_sentences("doc-one", [["Well", ",", "hello"]])
        words = sentences[0].words
        self.assertEqual(words[0].sourceTokenIndex, 0)
        self.assertEqual(words[1].sourceTokenIndex, 2)
        self.assertEqual(words[0].normalizedWordIndex, 0)
        self.assertEqual(words[1].normalizedWordIndex, 1)


if __name__ == "__main__":
    unittest.main()
