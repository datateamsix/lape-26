from __future__ import annotations

import unicodedata


def normalize_text(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed.upper() if "A" <= ch <= "Z")
