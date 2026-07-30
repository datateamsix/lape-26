from __future__ import annotations

import ast
import socket
import sys
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CORPUS_DIR = ROOT / "python" / "lape26" / "corpus"

ALL_CORPUS_MODULES = [
    "__init__.py", "tokens.py", "stats.py", "splits.py", "provenance.py",
    "controls.py", "acquire.py", "nltk_adapter.py", "stimulus.py", "report.py",
]
NLTK_IMPORTING_MODULES = {"acquire.py", "nltk_adapter.py"}


def find_module_level_nltk_imports(source_path: Path) -> list[str]:
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    findings: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "nltk" or alias.name.startswith("nltk."):
                    findings.append(f"{source_path}:{node.lineno}: module-level `import {alias.name}`")
        elif isinstance(node, ast.ImportFrom):
            if node.module and (node.module == "nltk" or node.module.startswith("nltk.")):
                findings.append(f"{source_path}:{node.lineno}: module-level `from {node.module} import ...`")
    return findings


class NoModuleLevelNltkImportTests(unittest.TestCase):
    def test_no_corpus_module_imports_nltk_at_module_scope(self) -> None:
        all_findings: list[str] = []
        for filename in ALL_CORPUS_MODULES:
            path = CORPUS_DIR / filename
            self.assertTrue(path.exists(), f"missing {path}")
            all_findings.extend(find_module_level_nltk_imports(path))
        self.assertEqual(all_findings, [], "\n".join(all_findings))

    def test_only_acquire_and_nltk_adapter_reference_nltk_at_all(self) -> None:
        for filename in ALL_CORPUS_MODULES:
            source = (CORPUS_DIR / filename).read_text(encoding="utf-8")
            references_nltk = "nltk" in source
            if filename in NLTK_IMPORTING_MODULES:
                self.assertTrue(references_nltk, f"{filename} should still reference nltk somewhere")
            else:
                self.assertFalse(references_nltk, f"{filename} should not reference nltk at all")


_LENGTH_BY_BAND = {"short": 4, "medium": 7, "long": 10}
_POS_CYCLE = ["noun", "verb", "adjective", "adverb"]


class _NetworkBlockedError(Exception):
    pass


@contextmanager
def _network_blocked():
    original_connect = socket.socket.connect

    def _blocked_connect(self, *args, **kwargs):
        raise _NetworkBlockedError("A network connection was attempted during an offline test run")

    socket.socket.connect = _blocked_connect
    try:
        yield
    finally:
        socket.socket.connect = original_connect


def _tiny_word_candidates(count_per_cell: int = 25):
    from lape26.corpus.stimulus import WordCandidate

    candidates = []
    for band in ("short", "medium", "long"):
        for polarity in ("positive", "negative", "neutral"):
            for i in range(count_per_cell):
                # polarity[:2] would collide ("negative"/"neutral" both ->
                # "ne"), giving those two cells identical word strings and
                # starving whichever is processed second once run_pipeline's
                # training-partition filter is applied (see Task 15); [:3]
                # is unique across all three polarities.
                word = f"{band[:2]}{polarity[:3]}{i:02d}".upper()
                candidates.append(
                    WordCandidate(
                        word=word,
                        length=_LENGTH_BY_BAND[band],
                        partOfSpeech=_POS_CYCLE[i % len(_POS_CYCLE)],
                        polarity=polarity,
                        vaderCompound=0.5 if polarity == "positive" else (-0.5 if polarity == "negative" else 0.0),
                        sourceDataset="synthetic-offline-test",
                        stemKey=word,
                    )
                )
    return candidates


_ORTHOGRAPHIC_BASE_WORDS = {
    "repeated-letters": ["BALLOON", "ANNOUNCE", "COMMITTEE"],
    "rare-letters": ["JAZZ", "QUIZ", "WALTZ"],
    "vowel-heavy": ["AI", "AREA", "IDEA"],
    "consonant-heavy": ["STRENGTH", "RHYTHM", "GLYPH"],
}


def _tiny_orthographic_candidates(count_per_category: int = 12):
    # run_pipeline filters orthographic candidates to the training word-list
    # partition before selection (leakage prevention, Task 15), so each
    # category needs enough members for >=3 to survive an ~80% keep rate.
    candidates: dict[str, list[str]] = {}
    for category, base_words in _ORTHOGRAPHIC_BASE_WORDS.items():
        prefix = "".join(part[0] for part in category.split("-")).upper()
        filler = [f"{prefix}SYN{i:02d}" for i in range(max(count_per_category - len(base_words), 0))]
        candidates[category] = base_words + filler
    return candidates


def _orthographic_membership_candidates():
    from lape26.corpus.stimulus import WordCandidate

    # Phantom WordCandidate entries so the orthographic words above are
    # members of word_candidates (and therefore eligible for the
    # training-partition split), without ever being eligible for core-set
    # selection: length=20 falls outside every length_band.
    candidates = []
    for category, words in _tiny_orthographic_candidates().items():
        for word in words:
            candidates.append(
                WordCandidate(
                    word=word,
                    length=20,
                    partOfSpeech="noun",
                    polarity="neutral",
                    vaderCompound=0.0,
                    sourceDataset="synthetic-orthographic-membership",
                    stemKey=f"phantom-{word}",
                )
            )
    return candidates


def _tiny_gutenberg_documents():
    return {
        "doc-one.txt": [["The", "cat", "sat", "."], ["A", "cat", "ran", "fast", "."]] * 10,
        "doc-two.txt": [["Dogs", "bark", "loudly", "."], ["Birds", "sing", "."]] * 5,
    }


class NoNetworkCallsDuringOfflinePipelineTests(unittest.TestCase):
    def test_run_pipeline_with_synthetic_data_makes_no_network_calls(self) -> None:
        sys.path.insert(0, str(ROOT / "scripts"))
        sys.path.insert(0, str(ROOT / "python"))
        from build_corpus_pipeline import run_pipeline

        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            lock_path = base / "fake-lock.json"
            lock_path.write_text("{}", encoding="utf-8")
            with _network_blocked():
                written = run_pipeline(
                    gutenberg_sentences_by_document=_tiny_gutenberg_documents(),
                    word_candidates=_tiny_word_candidates() + _orthographic_membership_candidates(),
                    orthographic_candidates_by_category=_tiny_orthographic_candidates(),
                    seed=26,
                    processed_dir=base / "processed",
                    fixtures_dir=base / "fixtures",
                    controls_dir=base / "controls",
                    corpus_lock_path=lock_path,
                )
            self.assertTrue(all(path.exists() for path in written.values()))


if __name__ == "__main__":
    unittest.main()
