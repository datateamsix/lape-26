from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from lape26.analysis import summarize
from lape26.core import encode_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a small deterministic LAPE-26 evaluation report")
    parser.add_argument("text", nargs="+", help="One or more words or quoted phrases")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = []
    for text in args.text:
        events = encode_text(text)
        report.append({"input": text, "events": events, "analysis": summarize(events)})

    rendered = json.dumps({"mapping": "lape-26-en-general-v0.1", "results": report}, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
