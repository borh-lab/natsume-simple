"""Validation shared by offline publication and API startup."""

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import duckdb


class ArtifactValidationError(Exception):
    def __init__(self, reason: str):
        self.reason = reason


def validate_artifact(artifact_dir: Path) -> tuple[Path, dict[str, Any]]:
    """Return a compatible database and manifest, or one bounded reason."""
    manifest_path = artifact_dir / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text())
    except FileNotFoundError as error:
        raise ArtifactValidationError("manifest_missing") from error
    except (OSError, json.JSONDecodeError, TypeError) as error:
        raise ArtifactValidationError("manifest_invalid") from error

    try:
        if manifest["schemaVersion"] != 1:
            raise ArtifactValidationError("unsupported_schema")
        expected_checksum = manifest["databaseSha256"]
        artifact_instance_id = manifest["artifactInstanceId"]
    except (KeyError, TypeError) as error:
        raise ArtifactValidationError("manifest_invalid") from error

    database_path = artifact_dir / "corpus.duckdb"
    try:
        with database_path.open("rb") as source:
            checksum = hashlib.file_digest(source, "sha256").hexdigest()
    except OSError as error:
        raise ArtifactValidationError("database_unreadable") from error
    if checksum != expected_checksum:
        raise ArtifactValidationError("checksum_mismatch")

    try:
        with duckdb.connect(str(database_path), read_only=True) as connection:
            metadata = connection.execute(
                "SELECT schema_version, artifact_instance_id FROM build_metadata"
            ).fetchone()
            corpus_count = connection.execute("SELECT count(*) FROM corpus").fetchone()[
                0
            ]
            corpus_ids = [
                row[0]
                for row in connection.execute(
                    "SELECT id FROM corpus ORDER BY id"
                ).fetchall()
            ]
    except duckdb.Error as error:
        raise ArtifactValidationError("database_unreadable") from error

    if metadata != (1, artifact_instance_id):
        raise ArtifactValidationError("identity_mismatch")
    if not 1 <= corpus_count <= 3:
        raise ArtifactValidationError("invalid_corpus_count")
    if any(
        not re.fullmatch(r"[a-z][a-z0-9-]{0,11}", corpus_id) for corpus_id in corpus_ids
    ):
        raise ArtifactValidationError("invalid_corpus_id")
    return database_path, manifest
