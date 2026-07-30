from __future__ import annotations

import json
import unittest
from pathlib import Path

import yaml
from jsonschema import Draft202012Validator

ROOT = Path(__file__).resolve().parents[2]
MANIFESTS_DIR = ROOT / "data" / "manifests"
SCHEMA_PATH = ROOT / "data" / "schemas" / "dataset-manifest.schema.json"

EXPECTED_MANIFESTS = {
    "gutenberg.yaml": "gutenberg",
    "words.yaml": "words",
    "wordnet.yaml": "wordnet",
    "opinion-lexicon.yaml": "opinion_lexicon",
    "vader.yaml": "vader_lexicon",
}

FORBIDDEN_DATASET_IDS = {"brown", "names", "cmudict", "punkt", "punkt_tab"}


class DatasetManifestTests(unittest.TestCase):
    def setUp(self) -> None:
        self.validator = Draft202012Validator(json.loads(SCHEMA_PATH.read_text(encoding="utf-8")))

    def test_all_five_manifests_exist_and_validate(self) -> None:
        for filename, expected_package_id in EXPECTED_MANIFESTS.items():
            path = MANIFESTS_DIR / filename
            with self.subTest(filename=filename):
                self.assertTrue(path.exists(), f"missing {path}")
                manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
                self.validator.validate(manifest)
                self.assertEqual(manifest["dataset_id"], expected_package_id)

    def test_no_forbidden_datasets_present(self) -> None:
        for path in MANIFESTS_DIR.glob("*.yaml"):
            manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
            self.assertNotIn(manifest.get("dataset_id"), FORBIDDEN_DATASET_IDS)

    def test_opinion_lexicon_and_vader_roles_exclude_musical_objective(self) -> None:
        for filename in ("opinion-lexicon.yaml", "vader.yaml"):
            manifest = yaml.safe_load((MANIFESTS_DIR / filename).read_text(encoding="utf-8"))
            self.assertIn(
                "musical objective",
                " ".join(manifest.get("must_not_be_used_for", [])).lower(),
            )

    def test_opinion_lexicon_license_is_cc_by_4_0(self) -> None:
        manifest = yaml.safe_load((MANIFESTS_DIR / "opinion-lexicon.yaml").read_text(encoding="utf-8"))
        self.assertIn("CC BY 4.0", manifest["license"])
        self.assertTrue(manifest["redistribution_allowed"])
        self.assertIn("citation", manifest)

    def test_version_fields_do_not_conflate_library_and_resource_version(self) -> None:
        for path in MANIFESTS_DIR.glob("*.yaml"):
            manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
            self.assertNotIn("nltk-3.9.1", manifest["version"])


if __name__ == "__main__":
    unittest.main()
