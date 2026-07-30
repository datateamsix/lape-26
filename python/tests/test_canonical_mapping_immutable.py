from __future__ import annotations

import unittest

from lape26.core import DEFAULT_MAPPING_PATH
from lape26.corpus.provenance import sha256_file

# Recorded once, from `python3 scripts/validate_mapping.py`'s own output
# against the current committed mappings/lape-26-en-general-v0.1.json.
EXPECTED_SHA256 = "dba13125a69eb815917611c655d7384668c26958d44fbfef93c03079765b4bc1"


class CanonicalMappingImmutableTests(unittest.TestCase):
    def test_canonical_mapping_matches_recorded_checksum(self) -> None:
        actual = sha256_file(DEFAULT_MAPPING_PATH)
        self.assertEqual(
            actual,
            EXPECTED_SHA256,
            "mappings/lape-26-en-general-v0.1.json has changed since this checksum "
            "was recorded. If this is an intentional, approved new mapping version, "
            "update EXPECTED_SHA256 here explicitly as its own reviewed change — "
            "never let this test silently start passing again on its own.",
        )


if __name__ == "__main__":
    unittest.main()
