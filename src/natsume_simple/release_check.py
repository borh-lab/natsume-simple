"""Structural policy checks for the first production corpus release."""

from pathlib import Path
from typing import Any

import duckdb

from natsume_simple.artifact_builder import source_manifest_sha256
from natsume_simple.artifact_validation import (
    ArtifactValidationError,
    validate_artifact,
)
from natsume_simple.release_inputs import (
    load_release_sources,
    load_wikipedia_subset,
)


class ReleaseCheckError(ValueError):
    """One bounded production-release invariant failed."""


def check_release_artifact(
    artifact_dir: Path,
    *,
    source_lock: Path,
    wikipedia_subset: Path,
) -> dict[str, object]:
    """Check cheap structural facts specific to the production release."""
    try:
        database_path, manifest = validate_artifact(artifact_dir)
    except ArtifactValidationError as error:
        raise ReleaseCheckError(f"artifact_invalid:{error.reason}") from error

    for notice_name in ("LICENSE-CONTENT.txt", "ATTRIBUTION.md"):
        try:
            if not (artifact_dir / notice_name).read_text(encoding="utf-8").strip():
                raise ReleaseCheckError("notice_missing")
        except OSError as error:
            raise ReleaseCheckError("notice_missing") from error

    sources = load_release_sources(source_lock)
    subset = load_wikipedia_subset(wikipedia_subset, sources=sources)
    with duckdb.connect(str(database_path), read_only=True) as connection:
        corpus_ids = [
            row[0]
            for row in connection.execute(
                "SELECT id FROM corpus ORDER BY id"
            ).fetchall()
        ]
        database_sources = [
            {
                "contentSha256": row[2],
                "corpusId": row[0],
                "externalId": row[1],
            }
            for row in connection.execute(
                """
                SELECT corpus_id, external_id, content_sha256
                FROM source
                ORDER BY corpus_id, external_id
                """
            ).fetchall()
        ]
        stored_source_checksum = connection.execute(
            "SELECT source_manifest_sha256 FROM build_metadata"
        ).fetchone()[0]

    if corpus_ids != ["jnlp", "wiki"]:
        raise ReleaseCheckError("release_corpus_mismatch")
    wikipedia_ids = {
        source["externalId"]
        for source in database_sources
        if source["corpusId"] == "wiki"
    }
    if wikipedia_ids != set(subset.article_ids):
        raise ReleaseCheckError("wikipedia_identity_mismatch")

    try:
        manifest_sources = manifest["identityInputs"]["sources"]
    except (KeyError, TypeError) as error:
        raise ReleaseCheckError("manifest_source_mismatch") from error
    if manifest_sources != database_sources:
        raise ReleaseCheckError("manifest_source_mismatch")
    if source_manifest_sha256(manifest_sources) != stored_source_checksum:
        raise ReleaseCheckError("source_manifest_checksum_mismatch")

    _check_rejection_limits(manifest)
    return {
        "artifactInstanceId": manifest["artifactInstanceId"],
        "corpusIds": corpus_ids,
        "sourceCount": len(database_sources),
        "wikipediaSourceCount": len(wikipedia_ids),
    }


def _check_rejection_limits(manifest: dict[str, Any]) -> None:
    try:
        counts = manifest["rejectionCounts"]
        totals = manifest["rejectionTotals"]
        max_count = manifest["rejectionLimits"]["maxCount"]
        max_fraction = manifest["rejectionLimits"]["maxFraction"]
        if not isinstance(counts, dict) or not isinstance(totals, dict):
            raise TypeError
        if not isinstance(max_count, int) or not isinstance(max_fraction, (int, float)):
            raise TypeError
        for stage, reasons in counts.items():
            total = totals[stage]
            if (
                not isinstance(reasons, dict)
                or not isinstance(total, int)
                or total <= 0
            ):
                raise TypeError
            for count in reasons.values():
                if not isinstance(count, int):
                    raise TypeError
                if count > max_count or count / total > max_fraction:
                    raise ReleaseCheckError("rejection_limit_exceeded")
    except (KeyError, TypeError, ZeroDivisionError) as error:
        raise ReleaseCheckError("rejection_policy_invalid") from error
