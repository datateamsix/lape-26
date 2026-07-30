from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

APPROVED_PACKAGES = ("gutenberg", "words", "wordnet", "opinion_lexicon", "vader_lexicon")


@dataclass(frozen=True)
class LockEntry:
    package_id: str
    source_version: str
    resource_path: str
    archive_sha256: str | None
    installed_tree_sha256: str
    retrieved_at: str


def download_approved_packages(download_dir: Path) -> None:
    """Download exactly the 5 approved NLTK packages into download_dir.
    `nltk` is imported here, not at module scope, so this module (and the
    rest of the corpus pipeline) stays importable and testable without
    nltk installed or any network access unless this function is called.
    """
    import nltk

    download_dir.mkdir(parents=True, exist_ok=True)
    for package_id in APPROVED_PACKAGES:
        nltk.download(package_id, download_dir=str(download_dir), quiet=True)


def _hash_directory(path: Path) -> str:
    hasher = hashlib.sha256()
    for file_path in sorted(path.rglob("*")):
        if file_path.is_file():
            hasher.update(str(file_path.relative_to(path)).encode("utf-8"))
            hasher.update(file_path.read_bytes())
    return hasher.hexdigest()


def _hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _find_package_resource(download_dir: Path, package_id: str) -> Path:
    """Find either an extracted `<package_id>/` directory or an
    unextracted `<package_id>.zip` file anywhere under download_dir.
    NLTK resources don't all live under the same subdirectory category
    (e.g. vader_lexicon lives under sentiment/, not corpora/), and some
    resources may remain as a zip rather than being auto-extracted, so
    this searches broadly rather than assuming a fixed layout.
    """
    directories = [candidate for candidate in download_dir.rglob(package_id) if candidate.is_dir()]
    if directories:
        return directories[0]
    zip_files = list(download_dir.rglob(f"{package_id}.zip"))
    if zip_files:
        return zip_files[0]
    raise FileNotFoundError(f"Downloaded package resource not found for {package_id!r}")


def build_lock_entries(download_dir: Path, retrieved_at: str | None = None) -> list[LockEntry]:
    retrieved_at = retrieved_at or datetime.now(timezone.utc).date().isoformat()
    entries = []
    for package_id in APPROVED_PACKAGES:
        resource = _find_package_resource(download_dir, package_id)
        if resource.is_dir():
            tree_hash = _hash_directory(resource)
            archive_hash = None  # the original archive (if any) was extracted and discarded
        else:
            tree_hash = _hash_file(resource)
            archive_hash = tree_hash  # the zip itself IS what's installed and used
        entries.append(
            LockEntry(
                package_id=package_id,
                source_version="documented-or-unknown",
                resource_path=str(resource.relative_to(download_dir)),
                archive_sha256=archive_hash,
                installed_tree_sha256=tree_hash,
                retrieved_at=retrieved_at,
            )
        )
    return entries


def write_lock_file(entries: list[LockEntry], lock_path: Path) -> None:
    payload = {"packages": [asdict(entry) for entry in entries]}
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_lock_file(lock_path: Path) -> list[LockEntry]:
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    return [LockEntry(**entry) for entry in payload["packages"]]


def verify_lock(download_dir: Path, lock_path: Path) -> tuple[bool, str]:
    if not lock_path.exists():
        return False, f"Lock file not found at {lock_path}"

    locked_entries = {entry.package_id: entry for entry in read_lock_file(lock_path)}
    for package_id in APPROVED_PACKAGES:
        if package_id not in locked_entries:
            return False, f"Lock file missing entry for {package_id!r}"
        try:
            resource = _find_package_resource(download_dir, package_id)
        except FileNotFoundError as error:
            return False, str(error)
        current_hash = _hash_directory(resource) if resource.is_dir() else _hash_file(resource)
        if current_hash != locked_entries[package_id].installed_tree_sha256:
            return False, (
                f"Local cache for {package_id!r} does not match corpus-lock.json "
                f"(expected {locked_entries[package_id].installed_tree_sha256}, "
                f"got {current_hash}). Run `make corpus-relock` if this is intentional."
            )
    return True, "Corpus lock verified"


def ensure_lock(download_dir: Path, lock_path: Path) -> tuple[bool, str]:
    if not lock_path.exists():
        entries = build_lock_entries(download_dir)
        write_lock_file(entries, lock_path)
        return True, f"Created new lock at {lock_path}"
    return verify_lock(download_dir, lock_path)


def relock(download_dir: Path, lock_path: Path) -> str:
    previous_entries = (
        {e.package_id: e for e in read_lock_file(lock_path)} if lock_path.exists() else {}
    )
    new_entries = build_lock_entries(download_dir)
    write_lock_file(new_entries, lock_path)

    changes: list[str] = []
    for entry in new_entries:
        old = previous_entries.get(entry.package_id)
        if old is None:
            changes.append(f"{entry.package_id}: new entry")
        elif old.installed_tree_sha256 != entry.installed_tree_sha256:
            changes.append(
                f"{entry.package_id}: changed "
                f"({old.installed_tree_sha256[:12]}... -> {entry.installed_tree_sha256[:12]}...)"
            )
    return "No changes." if not changes else "\n".join(changes)
