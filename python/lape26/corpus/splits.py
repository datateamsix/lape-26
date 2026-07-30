from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class WordListSplit:
    train: tuple[str, ...]
    validation: tuple[str, ...]
    holdout: tuple[str, ...]
    seed: int


def split_word_list(
    words: list[str],
    seed: int,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
) -> WordListSplit:
    deduplicated = sorted(set(words))
    rng = random.Random(seed)
    shuffled = deduplicated[:]
    rng.shuffle(shuffled)

    total = len(shuffled)
    train_end = round(total * train_ratio)
    validation_end = train_end + round(total * validation_ratio)

    return WordListSplit(
        train=tuple(sorted(shuffled[:train_end])),
        validation=tuple(sorted(shuffled[train_end:validation_end])),
        holdout=tuple(sorted(shuffled[validation_end:])),
        seed=seed,
    )


@dataclass(frozen=True)
class DocumentSplit:
    train: tuple[str, ...]
    validation: tuple[str, ...]
    holdout: tuple[str, ...]
    seed: int


def split_documents_by_word_count(
    document_word_counts: dict[str, int],
    seed: int,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
) -> DocumentSplit:
    """Assign whole documents to train/validation/holdout, keeping every
    document intact in exactly one partition, while approximating the
    target word-count ratios via deterministic greedy bin-balancing:
    documents are shuffled with `seed`, then each is assigned to whichever
    partition is currently furthest below its target word-count share.
    """
    document_ids = sorted(document_word_counts)
    rng = random.Random(seed)
    shuffled_ids = document_ids[:]
    rng.shuffle(shuffled_ids)

    total_words = sum(document_word_counts.values())
    holdout_ratio = 1 - train_ratio - validation_ratio
    targets = {
        "train": total_words * train_ratio,
        "validation": total_words * validation_ratio,
        "holdout": total_words * holdout_ratio,
    }
    assigned: dict[str, list[str]] = {"train": [], "validation": [], "holdout": []}
    running: dict[str, int] = {"train": 0, "validation": 0, "holdout": 0}

    for document_id in shuffled_ids:
        deficits = {name: targets[name] - running[name] for name in targets}
        chosen = max(deficits, key=lambda name: deficits[name])
        assigned[chosen].append(document_id)
        running[chosen] += document_word_counts[document_id]

    return DocumentSplit(
        train=tuple(sorted(assigned["train"])),
        validation=tuple(sorted(assigned["validation"])),
        holdout=tuple(sorted(assigned["holdout"])),
        seed=seed,
    )
