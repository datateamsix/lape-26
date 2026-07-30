from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .stimulus import WordCandidate

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'])")
_WORD_PATTERN = re.compile(r"[A-Za-z']+")

_POS_NAMES = {"n": "noun", "v": "verb", "a": "adjective", "s": "adjective", "r": "adverb"}
_POLARITY_CONFIRM_THRESHOLD = 0.05


def configure_nltk_data_path(download_dir: Path) -> None:
    import nltk

    resolved = str(download_dir.resolve())
    if resolved not in nltk.data.path:
        nltk.data.path.insert(0, resolved)


def split_sentences(raw_text: str) -> list[str]:
    """Project-owned deterministic sentence splitter. Deliberately avoids
    NLTK's punkt/punkt_tab tokenizer resource — not one of this project's
    5 approved packages, and flagged by NLTK's own data inventory as
    having ambiguous licensing. This is a simple regex splitter adequate
    for character/bigram/length statistics, not a linguistically complete
    sentence boundary detector.
    """
    normalized = re.sub(r"\s+", " ", raw_text.strip())
    if not normalized:
        return []
    return [s.strip() for s in _SENTENCE_BOUNDARY.split(normalized) if s.strip()]


def split_words(sentence: str) -> list[str]:
    return _WORD_PATTERN.findall(sentence)


def tokenize_raw_text(raw_text: str) -> list[list[str]]:
    return [split_words(sentence) for sentence in split_sentences(raw_text)]


def load_gutenberg_sentences_by_document(download_dir: Path) -> dict[str, list[list[str]]]:
    configure_nltk_data_path(download_dir)
    from nltk.corpus import gutenberg

    return {fileid: tokenize_raw_text(gutenberg.raw(fileid)) for fileid in gutenberg.fileids()}


def _load_manual_exclusions(exclusions_path: Path) -> set[str]:
    import yaml

    if not exclusions_path.exists():
        return set()
    data = yaml.safe_load(exclusions_path.read_text(encoding="utf-8")) or {}
    return {str(word).upper() for word in data.get("excluded_words", [])}


def load_word_candidates(download_dir: Path, exclusions_path: Path) -> list["WordCandidate"]:
    configure_nltk_data_path(download_dir)
    from nltk.corpus import opinion_lexicon, wordnet
    from nltk.corpus import words as words_corpus
    from nltk.sentiment import SentimentIntensityAnalyzer

    from .stimulus import WordCandidate

    positive_words = set(opinion_lexicon.positive())
    negative_words = set(opinion_lexicon.negative())
    dual_listed = positive_words & negative_words
    analyzer = SentimentIntensityAnalyzer()
    excluded_words = _load_manual_exclusions(exclusions_path)

    candidates: list[WordCandidate] = []
    seen_words: set[str] = set()
    for raw_word in words_corpus.words():
        word = raw_word.lower()
        if (
            not word.isalpha()
            or word in seen_words
            or word.upper() in excluded_words
            or word in dual_listed
        ):
            continue
        seen_words.add(word)

        synsets = wordnet.synsets(word)
        if not synsets:
            continue

        part_of_speech = _POS_NAMES.get(synsets[0].pos(), "other")
        stem_key = wordnet.morphy(word) or word
        compound = analyzer.polarity_scores(word)["compound"]

        if word in positive_words and compound >= _POLARITY_CONFIRM_THRESHOLD:
            polarity = "positive"
        elif word in negative_words and compound <= -_POLARITY_CONFIRM_THRESHOLD:
            polarity = "negative"
        elif (
            word not in positive_words
            and word not in negative_words
            and abs(compound) < _POLARITY_CONFIRM_THRESHOLD
        ):
            polarity = "neutral"
        else:
            continue  # Opinion Lexicon and VADER disagree, or ambiguous — drop rather than guess

        candidates.append(
            WordCandidate(
                word=word.upper(),
                length=len(word),
                partOfSpeech=part_of_speech,
                polarity=polarity,
                vaderCompound=compound,
                sourceDataset=(
                    "opinion-lexicon+vader-confirmed" if polarity != "neutral" else "words+wordnet+vader"
                ),
                stemKey=stem_key,
            )
        )
    return candidates


def load_orthographic_candidates(download_dir: Path, exclusions_path: Path) -> dict[str, list[str]]:
    configure_nltk_data_path(download_dir)
    from nltk.corpus import words as words_corpus

    from .stimulus import has_rare_letters, has_repeated_letters, is_consonant_heavy, is_vowel_heavy

    excluded_words = _load_manual_exclusions(exclusions_path)
    candidates: dict[str, list[str]] = {
        "repeated-letters": [], "rare-letters": [], "vowel-heavy": [], "consonant-heavy": [],
    }
    seen: set[str] = set()
    for raw_word in words_corpus.words():
        word = raw_word.lower()
        if not word.isalpha() or len(word) < 3 or word in seen or word.upper() in excluded_words:
            continue
        seen.add(word)
        if has_repeated_letters(word):
            candidates["repeated-letters"].append(word.upper())
        if has_rare_letters(word):
            candidates["rare-letters"].append(word.upper())
        if is_vowel_heavy(word):
            candidates["vowel-heavy"].append(word.upper())
        if is_consonant_heavy(word):
            candidates["consonant-heavy"].append(word.upper())
    return candidates
