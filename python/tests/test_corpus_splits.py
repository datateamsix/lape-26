from __future__ import annotations

import unittest

from lape26.corpus.splits import split_documents_by_word_count, split_word_list


class SplitWordListTests(unittest.TestCase):
    def test_no_overlap_and_full_coverage(self) -> None:
        words = [f"word{i}" for i in range(20)]
        result = split_word_list(words, seed=26)
        self.assertEqual(set(result.train) & set(result.validation), set())
        self.assertEqual(set(result.train) & set(result.holdout), set())
        self.assertEqual(set(result.validation) & set(result.holdout), set())
        self.assertEqual(
            set(result.train) | set(result.validation) | set(result.holdout),
            set(words),
        )

    def test_approximate_80_10_10_proportions(self) -> None:
        words = [f"word{i}" for i in range(20)]
        result = split_word_list(words, seed=26)
        self.assertEqual(len(result.train), 16)
        self.assertEqual(len(result.validation), 2)
        self.assertEqual(len(result.holdout), 2)

    def test_deterministic_given_same_seed(self) -> None:
        words = [f"word{i}" for i in range(20)]
        first = split_word_list(words, seed=26)
        second = split_word_list(words, seed=26)
        self.assertEqual(first, second)

    def test_deduplicates_input(self) -> None:
        result = split_word_list(["cat", "cat", "dog"], seed=26)
        total = len(result.train) + len(result.validation) + len(result.holdout)
        self.assertEqual(total, 2)


class SplitDocumentsByWordCountTests(unittest.TestCase):
    def _sample_counts(self) -> dict[str, int]:
        return {f"doc{i}": 1000 + i * 137 for i in range(18)}

    def test_every_document_assigned_exactly_once(self) -> None:
        counts = self._sample_counts()
        result = split_documents_by_word_count(counts, seed=26)
        all_assigned = list(result.train) + list(result.validation) + list(result.holdout)
        self.assertEqual(sorted(all_assigned), sorted(counts))
        self.assertEqual(len(all_assigned), len(set(all_assigned)))

    def test_deterministic_given_same_seed(self) -> None:
        counts = self._sample_counts()
        first = split_documents_by_word_count(counts, seed=26)
        second = split_documents_by_word_count(counts, seed=26)
        self.assertEqual(first, second)

    def test_train_partition_captures_majority_of_word_count(self) -> None:
        counts = self._sample_counts()
        result = split_documents_by_word_count(counts, seed=26)
        train_words = sum(counts[doc] for doc in result.train)
        total_words = sum(counts.values())
        self.assertGreater(train_words / total_words, 0.6)


if __name__ == "__main__":
    unittest.main()
