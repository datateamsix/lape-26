from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from statistics import mean, median

from .tokens import SentenceToken


@dataclass(frozen=True)
class CorpusStatistics:
    characterFrequency: dict[str, int]
    withinWordBigrams: dict[str, int]
    crossWordBoundaryBigrams: dict[str, int]
    wordInitialLetters: dict[str, int]
    wordFinalLetters: dict[str, int]
    sentenceInitialLetters: dict[str, int]
    sentenceFinalLetters: dict[str, int]
    wordLengthStats: dict[str, float]
    sentenceLengthStats: dict[str, float]
    wordCount: int
    sentenceCount: int


def compute_statistics(sentences: list[SentenceToken]) -> CorpusStatistics:
    character_frequency: Counter[str] = Counter()
    within_word_bigrams: Counter[str] = Counter()
    cross_word_bigrams: Counter[str] = Counter()
    word_initial: Counter[str] = Counter()
    word_final: Counter[str] = Counter()
    sentence_initial: Counter[str] = Counter()
    sentence_final: Counter[str] = Counter()
    word_lengths: list[int] = []
    sentence_lengths: list[int] = []

    for sentence in sentences:
        sentence_lengths.append(len(sentence.words))
        previous_final_letter: str | None = None
        for word in sentence.words:
            text = word.normalizedText
            character_frequency.update(text)
            for a, b in zip(text, text[1:]):
                within_word_bigrams[f"{a}{b}"] += 1
            word_lengths.append(len(text))
            word_initial[text[0]] += 1
            word_final[text[-1]] += 1
            if word.isSentenceInitial:
                sentence_initial[text[0]] += 1
            if word.isSentenceFinal:
                sentence_final[text[-1]] += 1
            if previous_final_letter is not None:
                cross_word_bigrams[f"{previous_final_letter}{text[0]}"] += 1
            previous_final_letter = text[-1]

    return CorpusStatistics(
        characterFrequency=dict(sorted(character_frequency.items())),
        withinWordBigrams=dict(sorted(within_word_bigrams.items())),
        crossWordBoundaryBigrams=dict(sorted(cross_word_bigrams.items())),
        wordInitialLetters=dict(sorted(word_initial.items())),
        wordFinalLetters=dict(sorted(word_final.items())),
        sentenceInitialLetters=dict(sorted(sentence_initial.items())),
        sentenceFinalLetters=dict(sorted(sentence_final.items())),
        wordLengthStats=_length_stats(word_lengths),
        sentenceLengthStats=_length_stats(sentence_lengths),
        wordCount=len(word_lengths),
        sentenceCount=len(sentences),
    )


def _length_stats(lengths: list[int]) -> dict[str, float]:
    if not lengths:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "median": 0.0}
    return {
        "min": float(min(lengths)),
        "max": float(max(lengths)),
        "mean": float(mean(lengths)),
        "median": float(median(lengths)),
    }
