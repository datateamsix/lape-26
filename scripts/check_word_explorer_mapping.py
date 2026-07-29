from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HTML_PATH = ROOT / "apps" / "word-explorer" / "index.html"
MAPPING_PATH = ROOT / "mappings" / "lape-26-en-general-v0.1.json"

ENTRY_RE = re.compile(
    r'^\s*([A-Z]):\s*\{\s*note:\s*"([^"]+)",\s*midi:\s*(\d+),\s*hz:\s*([0-9.]+)\s*\}',
    re.MULTILINE,
)


def main() -> None:
    html = HTML_PATH.read_text(encoding="utf-8")
    canonical = json.loads(MAPPING_PATH.read_text(encoding="utf-8"))["letters"]
    embedded = {
        letter: {"pitch": pitch, "midi": int(midi), "frequencyHz": float(hz)}
        for letter, pitch, midi, hz in ENTRY_RE.findall(html)
    }

    if set(embedded) != set(canonical):
        raise SystemExit("Word Explorer embedded mapping does not define exactly the canonical A-Z set")

    for letter, expected in canonical.items():
        actual = embedded[letter]
        if actual["pitch"] != expected["pitch"] or actual["midi"] != expected["midi"]:
            raise SystemExit(f"Word Explorer mapping mismatch for {letter}: {actual} != {expected}")
        if abs(actual["frequencyHz"] - expected["frequencyHz"]) > 0.000001:
            raise SystemExit(f"Word Explorer frequency mismatch for {letter}: {actual} != {expected}")

    print(f"Word Explorer mapping matches canonical JSON for {len(embedded)} letters.")


if __name__ == "__main__":
    main()
