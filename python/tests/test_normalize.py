from __future__ import annotations

import unittest

from lape26.normalize import normalize_text


class NormalizeTests(unittest.TestCase):
    def test_uppercases_and_strips_non_letters(self) -> None:
        self.assertEqual(normalize_text("Café, don't! 26"), "CAFEDONT")

    def test_keeps_only_ascii_letters_after_decomposition(self) -> None:
        self.assertEqual(normalize_text("naïve"), "NAIVE")

    def test_empty_string(self) -> None:
        self.assertEqual(normalize_text(""), "")


if __name__ == "__main__":
    unittest.main()
