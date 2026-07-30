from __future__ import annotations

from dataclasses import dataclass

from ..normalize import normalize_text


@dataclass(frozen=True)
class WordToken:
    sourceText: str
    normalizedText: str
    documentId: str
    sentenceIndex: int
    sourceTokenIndex: int
    normalizedWordIndex: int
    isSentenceInitial: bool
    isSentenceFinal: bool


@dataclass(frozen=True)
class SentenceToken:
    words: tuple[WordToken, ...]
    documentId: str
    sentenceIndex: int


def tokenize_sentences(document_id: str, raw_sentences: list[list[str]]) -> list[SentenceToken]:
    sentences: list[SentenceToken] = []
    for sentence_index, raw_words in enumerate(raw_sentences):
        kept: list[tuple[int, str, str]] = []
        for source_token_index, raw_word in enumerate(raw_words):
            normalized = normalize_text(raw_word)
            if normalized:
                kept.append((source_token_index, raw_word, normalized))
        if not kept:
            continue

        last_index = len(kept) - 1
        words = tuple(
            WordToken(
                sourceText=raw_word,
                normalizedText=normalized,
                documentId=document_id,
                sentenceIndex=sentence_index,
                sourceTokenIndex=source_token_index,
                normalizedWordIndex=normalized_word_index,
                isSentenceInitial=normalized_word_index == 0,
                isSentenceFinal=normalized_word_index == last_index,
            )
            for normalized_word_index, (source_token_index, raw_word, normalized) in enumerate(kept)
        )
        sentences.append(SentenceToken(words=words, documentId=document_id, sentenceIndex=sentence_index))
    return sentences
