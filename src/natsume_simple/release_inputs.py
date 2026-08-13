"""Locked source and frozen-subset values for production corpus releases."""

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ReleaseInputError(ValueError):
    """One bounded release-input invariant failed."""


@dataclass(frozen=True)
class LockedFile:
    source_lock_corpus_id: str
    name: str
    local_name: str
    url: str
    size: int
    sha256: str


@dataclass(frozen=True)
class ReleaseSources:
    jnlp_archive: LockedFile
    wikipedia_shard: LockedFile
    wikipedia_identity_sha256: str


@dataclass(frozen=True)
class WikipediaSubset:
    source_lock_corpus_id: str
    article_ids: tuple[str, ...]


def load_release_sources(source_lock: Path) -> ReleaseSources:
    """Load the two locked files selected for the production release."""
    payload = _read_json(source_lock, "source_lock_invalid")
    try:
        entries = {
            entry["corpusId"]: entry
            for entry in payload["sources"]
            if entry["status"] == "ready"
        }
        jnlp = entries["jnlp"]["candidate"]
        wikipedia = entries["wikipedia-ja-20231101"]["candidate"]
    except (KeyError, TypeError) as error:
        raise ReleaseInputError("source_lock_entry_missing") from error

    try:
        verified_shard = wikipedia["verification"]["verifiedShard"]
        shard = next(
            item for item in wikipedia["files"] if item["name"] == verified_shard
        )
        jnlp_release = str(jnlp["observedRelease"])
        return ReleaseSources(
            jnlp_archive=LockedFile(
                source_lock_corpus_id="jnlp",
                name="NLP_LATEX_CORPUS.zip",
                local_name=f"NLP_LATEX_CORPUS-{jnlp_release}.zip",
                url=str(jnlp["url"]),
                size=int(jnlp["size"]),
                sha256=str(jnlp["sha256"]),
            ),
            wikipedia_shard=LockedFile(
                source_lock_corpus_id="wikipedia-ja-20231101",
                name=str(shard["name"]),
                local_name=str(shard["name"]),
                url=str(wikipedia["urlTemplate"]).format(name=shard["name"]),
                size=int(shard["size"]),
                sha256=str(shard["sha256"]),
            ),
            wikipedia_identity_sha256=str(
                wikipedia["verification"]["orderedIdentityListSha256"]
            ),
        )
    except (KeyError, StopIteration, TypeError, ValueError) as error:
        raise ReleaseInputError("source_lock_invalid") from error


def canonical_article_ids_sha256(article_ids: Sequence[str]) -> str:
    """Hash ordered article IDs using the documented compact JSON encoding."""
    serialized = json.dumps(
        article_ids, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def load_wikipedia_subset(path: Path, *, sources: ReleaseSources) -> WikipediaSubset:
    """Load and validate the frozen 971-article Wikipedia selection."""
    payload = _read_json(path, "subset_invalid")
    try:
        source_lock_corpus_id = payload["sourceLockCorpusId"]
        article_ids = payload["articleIds"]
    except (KeyError, TypeError) as error:
        raise ReleaseInputError("subset_invalid") from error

    if source_lock_corpus_id != sources.wikipedia_shard.source_lock_corpus_id:
        raise ReleaseInputError("subset_source_mismatch")
    if not isinstance(article_ids, list) or len(article_ids) != 971:
        raise ReleaseInputError("subset_count_invalid")
    if any(
        not isinstance(article_id, str) or not article_id for article_id in article_ids
    ):
        raise ReleaseInputError("subset_identity_invalid")
    if len(set(article_ids)) != len(article_ids):
        raise ReleaseInputError("subset_identity_duplicate")
    if canonical_article_ids_sha256(article_ids) != sources.wikipedia_identity_sha256:
        raise ReleaseInputError("subset_checksum_mismatch")
    return WikipediaSubset(source_lock_corpus_id, tuple(article_ids))


def validate_wikipedia_paths(paths: Sequence[Path], *, sources: ReleaseSources) -> Path:
    """Return the one locked Wikipedia shard path or reject the selection."""
    if len(paths) != 1 or paths[0].name != sources.wikipedia_shard.name:
        raise ReleaseInputError("wikipedia_shard_invalid")
    return paths[0]


def verify_file(path: Path, locked: LockedFile) -> None:
    """Verify one local file against its locked byte count and SHA-256."""
    try:
        stat = path.stat()
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        raise ReleaseInputError("source_unreadable") from error
    if stat.st_size != locked.size:
        raise ReleaseInputError("source_size_mismatch")
    if digest.hexdigest() != locked.sha256:
        raise ReleaseInputError("source_checksum_mismatch")


def _read_json(path: Path, reason: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ReleaseInputError(reason) from error
    if not isinstance(payload, dict):
        raise ReleaseInputError(reason)
    return payload
