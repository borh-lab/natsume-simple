import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pytest

from natsume_simple.artifact_builder import (
    ArtifactRecords,
    BuildMetadata,
    CollocationOccurrence,
    CorpusRecord,
    SentenceRecord,
    SourceDocument,
    build_artifact,
)
from natsume_simple.release_check import (
    ReleaseCheckError,
    check_release_artifact,
)
from tests.test_release_inputs import write_source_lock, write_subset


def release_records(*, wikipedia_corpus_id: str = "wiki") -> ArtifactRecords:
    article_ids = [str(index) for index in range(971)]
    sources = [
        SourceDocument(
            "jnlp",
            "paper.tex",
            "論文",
            2026,
            None,
            "自然言語処理",
            None,
            ("情報を集める。",),
            "a" * 64,
        ),
        *[
            SourceDocument(
                wikipedia_corpus_id,
                article_id,
                f"記事 {article_id}",
                2023,
                "Wikipedia Contributors",
                "Wikimedia Foundation",
                None,
                ("情報を集める。",),
                hashlib.sha256(article_id.encode()).hexdigest(),
            )
            for article_id in article_ids
        ],
    ]
    sentences = (
        SentenceRecord(("jnlp", "paper.tex"), 0, "情報を集める。"),
        SentenceRecord((wikipedia_corpus_id, "0"), 0, "情報を集める。"),
    )
    occurrences = tuple(
        CollocationOccurrence(
            sentence.source_identity,
            0,
            "情報",
            "を",
            "集める",
            (0, 2),
            (2, 3),
            (3, 6),
            "fixture-extractor",
        )
        for sentence in sentences
    )
    return ArtifactRecords(
        corpora=(
            CorpusRecord("jnlp", "自然言語処理"),
            CorpusRecord(wikipedia_corpus_id, "Wikipedia"),
        ),
        sources=tuple(sources),
        sentences=sentences,
        occurrences=occurrences,
    )


def build_release_fixture(
    directory: Path, *, wikipedia_corpus_id: str = "wiki"
) -> Path:
    return build_artifact(
        directory,
        records=release_records(wikipedia_corpus_id=wikipedia_corpus_id),
        metadata=BuildMetadata(
            artifact_instance_id=directory.name,
            identity_inputs={"builderRevision": "fixture"},
            built_at=datetime(2026, 8, 13, tzinfo=UTC),
            content_license="CC BY-SA 4.0",
            attribution="Fixture attribution",
            rejection_counts={
                "jnlp": {"missing_source_path": 1},
                "wiki": {},
                "extraction": {},
            },
            rejection_limits={"maxCount": 2, "maxFraction": 0.5},
            rejection_totals={"jnlp": 3, "wiki": 971, "extraction": 2},
        ),
    )


def release_evidence(tmp_path: Path) -> tuple[Path, Path]:
    article_ids = [str(index) for index in range(971)]
    source_lock = write_source_lock(tmp_path / "sources.json", article_ids)
    subset = write_subset(tmp_path / "subset.json", article_ids)
    return source_lock, subset


def update_database_checksum(artifact: Path) -> None:
    database = artifact / "corpus.duckdb"
    manifest_path = artifact / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["databaseSha256"] = hashlib.sha256(database.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))


def check(artifact: Path, evidence: tuple[Path, Path]):
    source_lock, subset = evidence
    return check_release_artifact(
        artifact, source_lock=source_lock, wikipedia_subset=subset
    )


def test_release_check_accepts_the_exact_two_corpus_artifact(tmp_path: Path):
    artifact = build_release_fixture(tmp_path / "release")

    summary = check(artifact, release_evidence(tmp_path))

    assert summary == {
        "artifactInstanceId": "release",
        "corpusIds": ["jnlp", "wiki"],
        "sourceCount": 972,
        "wikipediaSourceCount": 971,
    }


def test_release_check_requires_nonempty_notices(tmp_path: Path):
    artifact = build_release_fixture(tmp_path / "release")
    (artifact / "ATTRIBUTION.md").write_text("")

    with pytest.raises(ReleaseCheckError, match="notice_missing"):
        check(artifact, release_evidence(tmp_path))


def test_release_check_requires_exact_corpus_ids(tmp_path: Path):
    artifact = build_release_fixture(tmp_path / "release", wikipedia_corpus_id="other")

    with pytest.raises(ReleaseCheckError, match="release_corpus_mismatch"):
        check(artifact, release_evidence(tmp_path))


def test_release_check_requires_all_frozen_wikipedia_ids(tmp_path: Path):
    artifact = build_release_fixture(tmp_path / "release")
    database = artifact / "corpus.duckdb"
    with duckdb.connect(str(database)) as connection:
        connection.execute(
            "DELETE FROM source WHERE corpus_id = 'wiki' AND external_id = '970'"
        )
    update_database_checksum(artifact)

    with pytest.raises(ReleaseCheckError, match="wikipedia_identity_mismatch"):
        check(artifact, release_evidence(tmp_path))


def test_release_check_reconciles_manifest_sources_with_database(tmp_path: Path):
    artifact = build_release_fixture(tmp_path / "release")
    manifest_path = artifact / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["identityInputs"]["sources"][0]["contentSha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ReleaseCheckError, match="manifest_source_mismatch"):
        check(artifact, release_evidence(tmp_path))


def test_release_check_reconciles_database_source_manifest_hash(tmp_path: Path):
    artifact = build_release_fixture(tmp_path / "release")
    database = artifact / "corpus.duckdb"
    with duckdb.connect(str(database)) as connection:
        connection.execute(
            "UPDATE build_metadata SET source_manifest_sha256 = ?", ["0" * 64]
        )
    update_database_checksum(artifact)

    with pytest.raises(ReleaseCheckError, match="source_manifest_checksum_mismatch"):
        check(artifact, release_evidence(tmp_path))


def test_release_check_enforces_recorded_rejection_limits(tmp_path: Path):
    artifact = build_release_fixture(tmp_path / "release")
    manifest_path = artifact / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["rejectionCounts"]["jnlp"]["missing_source_path"] = 2
    manifest["rejectionLimits"]["maxFraction"] = 0.5
    manifest["rejectionTotals"]["jnlp"] = 3
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ReleaseCheckError, match="rejection_limit_exceeded"):
        check(artifact, release_evidence(tmp_path))
