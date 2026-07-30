from __future__ import annotations

import copy
import hashlib
import json
import subprocess
from pathlib import Path

NORMALIZATION_PROFILE_ID = "lape-text-normalization-v0.1"
NORMALIZATION_STATUS = "provisional-frozen"
NORMALIZATION_SPEC_PATH = "specs/text-normalization.md"
NORMALIZATION_IMPLEMENTATION_PATH = "python/lape26/normalize.py"

REPO_ROOT = Path(__file__).resolve().parents[3]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_git(args: list[str]) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def git_committed_blob_sha(relative_path: str) -> str:
    """The SHA-1 of the blob HEAD actually points at for this path — NOT
    the same as hashing the current working-tree file, which may be dirty.
    """
    return _run_git(["rev-parse", f"HEAD:{relative_path}"])


def git_head_commit() -> str:
    return _run_git(["rev-parse", "HEAD"])


def git_is_dirty(paths: list[str] | None = None) -> bool:
    """Whether git reports uncommitted changes. When `paths` is given,
    scopes the check to only those paths — critical for provenance
    purposes, since a generated artifact's own output files are
    necessarily untracked at the moment it's generated (they can't be
    committed before they exist), which would make an unscoped check
    spuriously report "dirty" for every single generation run, forever.
    Scoping to just the source/input paths that actually determine the
    artifact's content avoids that false positive.
    """
    args = ["status", "--porcelain"]
    if paths:
        args.append("--")
        args.extend(paths)
    return bool(_run_git(args))


def source_tree_digest(paths: list[str]) -> str:
    hasher = hashlib.sha256()
    for relative_path in sorted(paths):
        hasher.update(relative_path.encode("utf-8"))
        hasher.update(sha256_file(REPO_ROOT / relative_path).encode("utf-8"))
    return hasher.hexdigest()


def input_data_digest(paths: list[str]) -> str:
    hasher = hashlib.sha256()
    for relative_path in sorted(paths):
        hasher.update(relative_path.encode("utf-8"))
        full_path = REPO_ROOT / relative_path
        if full_path.exists():
            hasher.update(sha256_file(full_path).encode("utf-8"))
        else:
            hasher.update(b"MISSING")
    return hasher.hexdigest()


def build_provenance_block(
    *,
    pipeline_source_paths: list[str],
    input_data_paths: list[str],
) -> dict[str, object]:
    normalize_path = REPO_ROOT / NORMALIZATION_IMPLEMENTATION_PATH
    return {
        "normalization_profile_id": NORMALIZATION_PROFILE_ID,
        "normalization_status": NORMALIZATION_STATUS,
        "normalization_spec_path": NORMALIZATION_SPEC_PATH,
        "normalization_implementation": NORMALIZATION_IMPLEMENTATION_PATH,
        "normalization_live_sha256": sha256_file(normalize_path),
        "normalization_committed_blob_sha": git_committed_blob_sha(NORMALIZATION_IMPLEMENTATION_PATH),
        "pipeline_source_commit": git_head_commit(),
        "working_tree_dirty": git_is_dirty(pipeline_source_paths + input_data_paths),
        "source_tree_digest": source_tree_digest(pipeline_source_paths),
        "input_data_sha256": input_data_digest(input_data_paths),
    }


_VOLATILE_PROVENANCE_FIELDS = ("pipeline_source_commit", "working_tree_dirty")


def semantic_content_bytes(payload: dict[str, object]) -> bytes:
    """Canonical JSON bytes for `payload` with volatile provenance fields
    (pipeline_source_commit, working_tree_dirty) stripped, so semantically
    identical artifacts hash the same even when generated at different
    times or from a dirty tree. Used for artifact-index checksums and
    corpus-check's drift comparison.
    """
    normalized = copy.deepcopy(payload)
    provenance = normalized.get("provenance")
    if isinstance(provenance, dict):
        for field in _VOLATILE_PROVENANCE_FIELDS:
            provenance.pop(field, None)
    return json.dumps(normalized, sort_keys=True, ensure_ascii=False).encode("utf-8")


def semantic_content_sha256(payload: dict[str, object]) -> str:
    return hashlib.sha256(semantic_content_bytes(payload)).hexdigest()
