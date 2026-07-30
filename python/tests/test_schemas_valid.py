from __future__ import annotations

import json
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator
from jsonschema.validators import validator_for

ROOT = Path(__file__).resolve().parents[2]
SCHEMAS_DIR = ROOT / "data" / "schemas"

EXPECTED_SCHEMAS = [
    "dataset-manifest.schema.json",
    "corpus-provenance.schema.json",
    "corpus-statistics.schema.json",
    "corpus-splits.schema.json",
    "pilot-stimulus.schema.json",
    "baseline-comparison.schema.json",
    "control-mapping.schema.json",
    "artifact-index.schema.json",
]


class SchemaFilesTests(unittest.TestCase):
    def test_all_expected_schemas_exist_parse_and_are_well_formed(self) -> None:
        for filename in EXPECTED_SCHEMAS:
            path = SCHEMAS_DIR / filename
            with self.subTest(filename=filename):
                self.assertTrue(path.exists(), f"missing {path}")
                data = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(data["$schema"], "https://json-schema.org/draft/2020-12/schema")
                self.assertIn("required", data)
                self.assertIn("properties", data)
                validator_cls = validator_for(data)
                validator_cls.check_schema(data)  # raises if the schema itself is malformed

    def test_dataset_manifest_schema_permits_optional_citation(self) -> None:
        schema = json.loads((SCHEMAS_DIR / "dataset-manifest.schema.json").read_text(encoding="utf-8"))
        self.assertIn("citation", schema["properties"])

    def test_generated_artifact_schemas_reference_provenance_schema(self) -> None:
        for filename in (
            "corpus-statistics.schema.json",
            "corpus-splits.schema.json",
            "pilot-stimulus.schema.json",
            "baseline-comparison.schema.json",
            "control-mapping.schema.json",
        ):
            schema = json.loads((SCHEMAS_DIR / filename).read_text(encoding="utf-8"))
            provenance_property = schema["properties"]["provenance"]
            self.assertIn("$ref", provenance_property, f"{filename} does not $ref the provenance schema")

    def test_dataset_manifest_date_format_is_enforced_with_format_checker(self) -> None:
        schema = json.loads((SCHEMAS_DIR / "dataset-manifest.schema.json").read_text(encoding="utf-8"))
        validator = Draft202012Validator(schema, format_checker=Draft202012Validator.FORMAT_CHECKER)
        manifest = {
            "dataset_id": "x", "name": "x", "version": "x", "source_url": "x",
            "retrieved_at": "not-a-date", "license": "x", "redistribution_allowed": True,
            "content_type": "x", "language": "en", "locale": "general",
            "checksum_sha256": "", "role": "x", "preparation_steps": ["x"], "known_biases": [],
        }
        errors = list(validator.iter_errors(manifest))
        self.assertTrue(any("date" in e.message.lower() or "not-a-date" in str(e.instance) for e in errors))


if __name__ == "__main__":
    unittest.main()
