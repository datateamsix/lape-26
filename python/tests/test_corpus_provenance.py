from __future__ import annotations

import re
import unittest
from pathlib import Path

from lape26.corpus.provenance import (
    NORMALIZATION_IMPLEMENTATION_PATH,
    NORMALIZATION_PROFILE_ID,
    NORMALIZATION_SPEC_PATH,
    NORMALIZATION_STATUS,
    REPO_ROOT,
    build_provenance_block,
    git_committed_blob_sha,
    sha256_file,
)

HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")


class ProvenanceBlockTests(unittest.TestCase):
    def test_provenance_block_has_all_required_fields(self) -> None:
        block = build_provenance_block(
            pipeline_source_paths=[NORMALIZATION_IMPLEMENTATION_PATH],
            input_data_paths=["data/manifests/stimulus-exclusions.yaml"],
        )
        self.assertEqual(block["normalization_profile_id"], NORMALIZATION_PROFILE_ID)
        self.assertEqual(block["normalization_status"], NORMALIZATION_STATUS)
        self.assertEqual(block["normalization_spec_path"], NORMALIZATION_SPEC_PATH)
        self.assertEqual(block["normalization_implementation"], NORMALIZATION_IMPLEMENTATION_PATH)
        self.assertRegex(block["normalization_live_sha256"], HEX64)
        self.assertRegex(block["normalization_committed_blob_sha"], HEX40)
        self.assertRegex(block["pipeline_source_commit"], HEX40)
        self.assertIsInstance(block["working_tree_dirty"], bool)
        self.assertRegex(block["source_tree_digest"], HEX64)
        self.assertRegex(block["input_data_sha256"], HEX64)

    def test_live_sha256_matches_sha256_file(self) -> None:
        path = REPO_ROOT / NORMALIZATION_IMPLEMENTATION_PATH
        block = build_provenance_block(pipeline_source_paths=[NORMALIZATION_IMPLEMENTATION_PATH], input_data_paths=[])
        self.assertEqual(block["normalization_live_sha256"], sha256_file(path))

    def test_committed_blob_sha_matches_git_plumbing_directly(self) -> None:
        # Distinct code path from sha256_file: goes through `git rev-parse
        # HEAD:<path>` rather than reading the working-tree file — this is
        # what makes it "the committed blob" rather than "the live file".
        block = build_provenance_block(pipeline_source_paths=[NORMALIZATION_IMPLEMENTATION_PATH], input_data_paths=[])
        self.assertEqual(
            block["normalization_committed_blob_sha"],
            git_committed_blob_sha(NORMALIZATION_IMPLEMENTATION_PATH),
        )

    def test_source_tree_digest_is_deterministic_for_same_paths(self) -> None:
        first = build_provenance_block(pipeline_source_paths=[NORMALIZATION_IMPLEMENTATION_PATH], input_data_paths=[])
        second = build_provenance_block(pipeline_source_paths=[NORMALIZATION_IMPLEMENTATION_PATH], input_data_paths=[])
        self.assertEqual(first["source_tree_digest"], second["source_tree_digest"])

    def test_input_data_digest_tolerates_missing_path(self) -> None:
        block = build_provenance_block(
            pipeline_source_paths=[NORMALIZATION_IMPLEMENTATION_PATH],
            input_data_paths=["data/manifests/this-file-does-not-exist-yet.yaml"],
        )
        self.assertRegex(block["input_data_sha256"], HEX64)

    def test_working_tree_dirty_is_scoped_to_given_paths(self) -> None:
        from lape26.corpus.provenance import git_is_dirty

        # A path that has never existed and is never tracked has no git
        # status output, so a *scoped* dirty-check must report False for
        # it regardless of whatever else is going on elsewhere in the
        # repo's working tree (including, notably, this very generation
        # run's own not-yet-committed output artifacts, which is exactly
        # the case this scoping exists to handle — see Task 23).
        self.assertFalse(git_is_dirty(["this/path/has/never/existed.txt"]))


if __name__ == "__main__":
    unittest.main()
