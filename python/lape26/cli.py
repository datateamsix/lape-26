from __future__ import annotations

import argparse
import json

from .analysis import summarize
from .core import encode_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode text with LAPE-26")
    subparsers = parser.add_subparsers(dest="command", required=True)

    encode_parser = subparsers.add_parser("encode", help="Encode text as LAPE-26 events")
    encode_parser.add_argument("text")
    encode_parser.add_argument("--analysis", action="store_true")

    args = parser.parse_args()
    if args.command == "encode":
        events = encode_text(args.text)
        payload = {"input": args.text, "events": events}
        if args.analysis:
            payload["analysis"] = summarize(events)
        print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
