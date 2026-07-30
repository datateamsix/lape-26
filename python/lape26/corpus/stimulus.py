from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass

_BANDS = ("short", "medium", "long")
_POLARITIES = ("positive", "negative", "neutral")
_VOWELS = set("AEIOU")
_RARE_LETTERS = set("JQXZ")
_ORTHOGRAPHIC_CATEGORIES = ("repeated-letters", "rare-letters", "vowel-heavy", "consonant-heavy")


def stable_seed(base_seed: int, *parts: str) -> int:
    """Deterministic seed derivation stable across separate Python
    interpreter processes. CPython salts string hash() per process by
    default (PYTHONHASHSEED), so hash() must never be used for seeding —
    this uses a fixed SHA-256 digest instead.
    """
    payload = "|".join(parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return base_seed + int.from_bytes(digest[:8], "big")


@dataclass(frozen=True)
class WordCandidate:
    word: str
    length: int
    partOfSpeech: str
    polarity: str
    vaderCompound: float
    sourceDataset: str
    stemKey: str


def length_band(length: int) -> str | None:
    if 3 <= length <= 5:
        return "short"
    if 6 <= length <= 8:
        return "medium"
    if 9 <= length <= 12:
        return "long"
    return None


def select_core_set(
    candidates: list[WordCandidate],
    seed: int,
    used_words: set[str],
    used_stems: set[str],
    per_cell: int = 12,
) -> list[WordCandidate]:
    by_cell: dict[tuple[str, str], list[WordCandidate]] = {
        (band, polarity): [] for band in _BANDS for polarity in _POLARITIES
    }
    for candidate in candidates:
        band = length_band(candidate.length)
        if band is None or candidate.polarity not in _POLARITIES:
            continue
        by_cell[(band, candidate.polarity)].append(candidate)

    selected: list[WordCandidate] = []
    for (band, polarity), pool in by_cell.items():
        cell_seed = stable_seed(seed, band, polarity)
        selected.extend(_select_cell(pool, per_cell, cell_seed, band, polarity, used_words, used_stems))
    return selected


def _select_cell(
    pool: list[WordCandidate],
    per_cell: int,
    seed: int,
    band: str,
    polarity: str,
    used_words: set[str],
    used_stems: set[str],
) -> list[WordCandidate]:
    eligible = [c for c in pool if c.word not in used_words and c.stemKey not in used_stems]

    deduplicated: dict[str, WordCandidate] = {}
    for candidate in sorted(eligible, key=lambda c: c.word):
        deduplicated.setdefault(candidate.stemKey, candidate)
    unique_candidates = list(deduplicated.values())

    rng = random.Random(seed)
    shuffled = unique_candidates[:]
    rng.shuffle(shuffled)

    by_pos: dict[str, list[WordCandidate]] = {}
    for candidate in shuffled:
        by_pos.setdefault(candidate.partOfSpeech, []).append(candidate)

    chosen: list[WordCandidate] = []
    pos_cycle = sorted(by_pos)
    pos_index = 0
    max_attempts = per_cell * max(len(pos_cycle), 1) + len(unique_candidates) + 1
    attempts = 0
    while len(chosen) < per_cell and attempts < max_attempts and pos_cycle:
        pos = pos_cycle[pos_index % len(pos_cycle)]
        if by_pos[pos]:
            candidate = by_pos[pos].pop(0)
            chosen.append(candidate)
            used_words.add(candidate.word)
            used_stems.add(candidate.stemKey)
        pos_index += 1
        attempts += 1

    if len(chosen) < per_cell:
        raise ValueError(
            f"Not enough eligible candidates for {band}/{polarity}: "
            f"need {per_cell}, found {len(chosen)} after dedup and exclusion"
        )
    return sorted(chosen, key=lambda c: c.word)[:per_cell]


def has_repeated_letters(word: str) -> bool:
    upper = word.upper()
    return any(upper.count(letter) >= 2 for letter in set(upper))


def has_rare_letters(word: str) -> bool:
    upper = word.upper()
    return any(letter in _RARE_LETTERS for letter in upper)


def is_vowel_heavy(word: str) -> bool:
    upper = word.upper()
    if not upper:
        return False
    vowel_count = sum(1 for ch in upper if ch in _VOWELS)
    return vowel_count / len(upper) >= 0.5


def is_consonant_heavy(word: str) -> bool:
    upper = word.upper()
    if not upper:
        return False
    consonant_count = sum(1 for ch in upper if ch not in _VOWELS)
    return consonant_count / len(upper) >= 0.8


@dataclass(frozen=True)
class OrthographicCandidate:
    word: str
    category: str


def select_orthographic_challenge_set(
    candidates_by_category: dict[str, list[str]],
    seed: int,
    used_words: set[str],
    per_category: int = 3,
) -> list[OrthographicCandidate]:
    selected: list[OrthographicCandidate] = []
    for category in _ORTHOGRAPHIC_CATEGORIES:
        pool = sorted(set(candidates_by_category.get(category, [])) - used_words)
        rng = random.Random(stable_seed(seed, category))
        shuffled = pool[:]
        rng.shuffle(shuffled)
        if len(shuffled) < per_category:
            raise ValueError(
                f"Not enough orthographic candidates for {category!r} after exclusions: "
                f"need {per_category}, found {len(shuffled)}"
            )
        chosen = sorted(shuffled[:per_category])
        for word in chosen:
            used_words.add(word)
        selected.extend(OrthographicCandidate(word=word, category=category) for word in chosen)
    return selected
