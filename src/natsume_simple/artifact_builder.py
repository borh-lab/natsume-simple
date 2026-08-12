from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import duckdb


SCHEMA_VERSION = 1
PARTICLES = frozenset({"が", "を", "に", "で", "から", "より", "と", "へ"})

SCHEMA_SQL = """
CREATE TABLE build_metadata (
    schema_version INTEGER NOT NULL,
    artifact_instance_id TEXT NOT NULL UNIQUE,
    builder_version TEXT NOT NULL,
    extractor_id TEXT NOT NULL,
    execution_profile_json TEXT NOT NULL,
    source_manifest_sha256 TEXT NOT NULL,
    built_at_utc TIMESTAMP NOT NULL
);
CREATE TABLE corpus (id TEXT PRIMARY KEY, label TEXT NOT NULL);
CREATE TABLE source (
    id INTEGER PRIMARY KEY,
    corpus_id TEXT NOT NULL REFERENCES corpus(id),
    external_id TEXT NOT NULL,
    title TEXT NOT NULL,
    year INTEGER,
    author TEXT,
    publisher TEXT,
    url TEXT,
    content_sha256 TEXT NOT NULL,
    UNIQUE (corpus_id, external_id)
);
CREATE TABLE sentence (
    id INTEGER PRIMARY KEY,
    source_id INTEGER NOT NULL REFERENCES source(id),
    ordinal INTEGER NOT NULL,
    text TEXT NOT NULL,
    UNIQUE (source_id, ordinal)
);
CREATE TABLE collocation_occurrence (
    sentence_id INTEGER NOT NULL REFERENCES sentence(id),
    noun TEXT NOT NULL,
    particle TEXT NOT NULL,
    verb TEXT NOT NULL,
    n_begin INTEGER NOT NULL,
    n_end INTEGER NOT NULL,
    p_begin INTEGER NOT NULL,
    p_end INTEGER NOT NULL,
    v_begin INTEGER NOT NULL,
    v_end INTEGER NOT NULL,
    extractor_id TEXT NOT NULL,
    UNIQUE (
        sentence_id, noun, particle, verb,
        n_begin, n_end, p_begin, p_end, v_begin, v_end, extractor_id
    )
);
CREATE TABLE corpus_stats (
    corpus_id TEXT PRIMARY KEY REFERENCES corpus(id),
    source_count INTEGER NOT NULL,
    sentence_count INTEGER NOT NULL,
    collocation_count INTEGER NOT NULL
);
CREATE TABLE lemma_frequency (
    part_of_speech TEXT NOT NULL,
    lemma TEXT NOT NULL,
    occurrence_count INTEGER NOT NULL,
    UNIQUE (part_of_speech, lemma)
);
CREATE VIEW collocation_frequency AS
    SELECT src.corpus_id, o.noun, o.particle, o.verb,
           count(*)::INTEGER AS raw_frequency
    FROM collocation_occurrence o
    JOIN sentence s ON s.id = o.sentence_id
    JOIN source src ON src.id = s.source_id
    GROUP BY src.corpus_id, o.noun, o.particle, o.verb;
"""


class ArtifactBuildError(ValueError):
    """A canonical input cannot produce a publishable serving artifact."""


@dataclass(frozen=True)
class CorpusRecord:
    id: str
    label: str


@dataclass(frozen=True)
class SourceDocument:
    corpus_id: str
    external_id: str
    title: str
    year: int | None
    author: str | None
    publisher: str | None
    url: str | None
    text_units: tuple[str, ...]
    content_sha256: str


@dataclass(frozen=True)
class SentenceRecord:
    source_identity: tuple[str, str]
    ordinal: int
    text: str


@dataclass(frozen=True)
class CollocationOccurrence:
    source_identity: tuple[str, str]
    sentence_ordinal: int
    noun: str
    particle: str
    verb: str
    noun_span: tuple[int, int]
    particle_span: tuple[int, int]
    verb_span: tuple[int, int]
    extractor_id: str


def create_schema_v1(connection: duckdb.DuckDBPyConnection) -> None:
    """Create the complete serving schema in an empty DuckDB database."""
    connection.execute(SCHEMA_SQL)


def build_artifact(
    output_directory: Path,
    *,
    artifact_instance_id: str,
    corpora: list[CorpusRecord],
    sources: list[SourceDocument],
    sentences: list[SentenceRecord],
    occurrences: list[CollocationOccurrence],
    identity_inputs: dict[str, Any],
    built_at: datetime,
    content_license: str,
    attribution: str,
) -> Path:
    """Persist canonical records into one validated, immutable artifact."""
    staging = output_directory.with_name(f"{output_directory.name}.staging")
    if output_directory.exists() or staging.exists():
        raise ArtifactBuildError("artifact_path_exists")

    ordered_corpora = sorted(corpora, key=lambda corpus: corpus.id)
    ordered_sources = sorted(
        sources, key=lambda source: (source.corpus_id, source.external_id)
    )
    ordered_sentences = sorted(
        sentences,
        key=lambda sentence: (*sentence.source_identity, sentence.ordinal),
    )
    ordered_occurrences = sorted(
        occurrences,
        key=lambda occurrence: (
            *occurrence.source_identity,
            occurrence.sentence_ordinal,
            occurrence.noun,
            occurrence.particle,
            occurrence.verb,
            occurrence.noun_span,
            occurrence.particle_span,
            occurrence.verb_span,
            occurrence.extractor_id,
        ),
    )
    _validate_records(
        artifact_instance_id,
        ordered_corpora,
        ordered_sources,
        ordered_sentences,
        ordered_occurrences,
    )

    staging.mkdir()
    database_path = staging / "corpus.duckdb"
    source_ids = {
        (source.corpus_id, source.external_id): source_id
        for source_id, source in enumerate(ordered_sources, start=1)
    }
    sentence_ids = {
        (sentence.source_identity, sentence.ordinal): sentence_id
        for sentence_id, sentence in enumerate(ordered_sentences, start=1)
    }
    extractor_id = ordered_occurrences[0].extractor_id
    source_identity_inputs = _source_identity_inputs(ordered_sources)
    source_manifest_sha256 = _source_manifest_sha256(source_identity_inputs)
    applied_identity_inputs = {**identity_inputs, "sources": source_identity_inputs}

    with duckdb.connect(str(database_path)) as connection:
        create_schema_v1(connection)
        connection.execute(
            "INSERT INTO build_metadata VALUES (?, ?, ?, ?, ?, ?, ?)",
            [
                SCHEMA_VERSION,
                artifact_instance_id,
                str(identity_inputs.get("builderRevision", "unknown")),
                extractor_id,
                json.dumps(
                    identity_inputs.get("executionProfile", {}),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                source_manifest_sha256,
                built_at,
            ],
        )
        connection.executemany(
            "INSERT INTO corpus VALUES (?, ?)",
            [(corpus.id, corpus.label) for corpus in ordered_corpora],
        )
        connection.executemany(
            "INSERT INTO source VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    source_ids[(source.corpus_id, source.external_id)],
                    source.corpus_id,
                    source.external_id,
                    source.title,
                    source.year,
                    source.author,
                    source.publisher,
                    source.url,
                    source.content_sha256,
                )
                for source in ordered_sources
            ],
        )
        connection.executemany(
            "INSERT INTO sentence VALUES (?, ?, ?, ?)",
            [
                (
                    sentence_ids[(sentence.source_identity, sentence.ordinal)],
                    source_ids[sentence.source_identity],
                    sentence.ordinal,
                    sentence.text,
                )
                for sentence in ordered_sentences
            ],
        )
        connection.executemany(
            "INSERT INTO collocation_occurrence VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    sentence_ids[
                        (occurrence.source_identity, occurrence.sentence_ordinal)
                    ],
                    occurrence.noun,
                    occurrence.particle,
                    occurrence.verb,
                    *occurrence.noun_span,
                    *occurrence.particle_span,
                    *occurrence.verb_span,
                    occurrence.extractor_id,
                )
                for occurrence in ordered_occurrences
            ],
        )
        connection.execute(
            """
            INSERT INTO corpus_stats
            SELECT c.id,
                   count(DISTINCT src.id)::INTEGER,
                   count(DISTINCT s.id)::INTEGER,
                   count(o.sentence_id)::INTEGER
            FROM corpus c
            LEFT JOIN source src ON src.corpus_id = c.id
            LEFT JOIN sentence s ON s.source_id = src.id
            LEFT JOIN collocation_occurrence o ON o.sentence_id = s.id
            GROUP BY c.id
            ORDER BY c.id
            """
        )
        connection.execute(
            """
            INSERT INTO lemma_frequency
            SELECT part_of_speech, lemma, count(*)::INTEGER
            FROM (
                SELECT 'noun' AS part_of_speech, noun AS lemma
                FROM collocation_occurrence
                UNION ALL
                SELECT 'verb' AS part_of_speech, verb AS lemma
                FROM collocation_occurrence
            ) lemmas
            GROUP BY part_of_speech, lemma
            ORDER BY part_of_speech, lemma
            """
        )
        _validate_persisted_facts(connection)

    relation_counts = {
        "corpus": len(ordered_corpora),
        "source": len(ordered_sources),
        "sentence": len(ordered_sentences),
        "collocationOccurrence": len(ordered_occurrences),
    }
    database_sha256 = hashlib.sha256(database_path.read_bytes()).hexdigest()
    manifest = {
        "artifactInstanceId": artifact_instance_id,
        "schemaVersion": SCHEMA_VERSION,
        "databaseSha256": database_sha256,
        "identityInputs": applied_identity_inputs,
        "relationCounts": relation_counts,
    }
    (staging / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    (staging / "validation.json").write_text(
        json.dumps(
            {"status": "passed", "relationCounts": relation_counts},
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (staging / "LICENSE-CONTENT.txt").write_text(
        content_license.rstrip() + "\n", encoding="utf-8"
    )
    (staging / "ATTRIBUTION.md").write_text(
        attribution.rstrip() + "\n", encoding="utf-8"
    )
    staging.rename(output_directory)
    return output_directory


def _validate_records(
    artifact_instance_id: str,
    corpora: list[CorpusRecord],
    sources: list[SourceDocument],
    sentences: list[SentenceRecord],
    occurrences: list[CollocationOccurrence],
) -> None:
    if not artifact_instance_id:
        raise ArtifactBuildError("artifact_instance_id_invalid")
    if not 1 <= len(corpora) <= 3:
        raise ArtifactBuildError("corpus_count_invalid")
    corpus_ids = [corpus.id for corpus in corpora]
    if len(set(corpus_ids)) != len(corpus_ids):
        raise ArtifactBuildError("corpus_identity_duplicate")
    if any(
        not re.fullmatch(r"[a-z][a-z0-9-]{0,11}", corpus.id)
        or not 1 <= len(corpus.label) <= 64
        for corpus in corpora
    ):
        raise ArtifactBuildError("corpus_invalid")

    source_identities = [(source.corpus_id, source.external_id) for source in sources]
    if len(set(source_identities)) != len(source_identities):
        raise ArtifactBuildError("source_identity_duplicate")
    if any(
        source.corpus_id not in corpus_ids
        or not source.external_id
        or not 1 <= len(source.title) <= 512
        or not re.fullmatch(r"[0-9a-f]{64}", source.content_sha256)
        for source in sources
    ):
        raise ArtifactBuildError("source_invalid")

    sentence_by_identity = {
        (sentence.source_identity, sentence.ordinal): sentence for sentence in sentences
    }
    if len(sentence_by_identity) != len(sentences):
        raise ArtifactBuildError("sentence_identity_duplicate")
    if any(
        sentence.source_identity not in source_identities
        or sentence.ordinal < 0
        or not 1 <= len(sentence.text) <= 4096
        for sentence in sentences
    ):
        raise ArtifactBuildError("sentence_invalid")

    extractor_ids = {occurrence.extractor_id for occurrence in occurrences}
    if len(extractor_ids) != 1 or "" in extractor_ids:
        raise ArtifactBuildError("extractor_identity_invalid")
    for occurrence in occurrences:
        sentence = sentence_by_identity.get(
            (occurrence.source_identity, occurrence.sentence_ordinal)
        )
        if sentence is None:
            raise ArtifactBuildError("occurrence_sentence_missing")
        if (
            not 1 <= len(occurrence.noun) <= 64
            or not 1 <= len(occurrence.verb) <= 64
            or occurrence.particle not in PARTICLES
        ):
            raise ArtifactBuildError("occurrence_lemma_invalid")
        if any(
            begin < 0 or begin >= end or end > len(sentence.text)
            for begin, end in (
                occurrence.noun_span,
                occurrence.particle_span,
                occurrence.verb_span,
            )
        ):
            raise ArtifactBuildError("occurrence_span_invalid")

    for corpus_id in corpus_ids:
        corpus_sources = {
            identity for identity in source_identities if identity[0] == corpus_id
        }
        corpus_sentences = {
            (sentence.source_identity, sentence.ordinal)
            for sentence in sentences
            if sentence.source_identity in corpus_sources
        }
        if (
            not corpus_sources
            or not corpus_sentences
            or not any(
                (occurrence.source_identity, occurrence.sentence_ordinal)
                in corpus_sentences
                for occurrence in occurrences
            )
        ):
            raise ArtifactBuildError("corpus_empty")


def _source_identity_inputs(sources: list[SourceDocument]) -> list[dict[str, str]]:
    return [
        {
            "contentSha256": source.content_sha256,
            "corpusId": source.corpus_id,
            "externalId": source.external_id,
        }
        for source in sources
    ]


def _source_manifest_sha256(source_identities: list[dict[str, str]]) -> str:
    serialized = json.dumps(
        source_identities, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(serialized).hexdigest()


def _validate_persisted_facts(connection: duckdb.DuckDBPyConnection) -> None:
    corpus_stats_match = connection.execute(
        """
        SELECT NOT EXISTS (
            (SELECT * FROM corpus_stats EXCEPT
             SELECT c.id,
                    count(DISTINCT src.id)::INTEGER,
                    count(DISTINCT s.id)::INTEGER,
                    count(o.sentence_id)::INTEGER
             FROM corpus c
             LEFT JOIN source src ON src.corpus_id = c.id
             LEFT JOIN sentence s ON s.source_id = src.id
             LEFT JOIN collocation_occurrence o ON o.sentence_id = s.id
             GROUP BY c.id)
            UNION ALL
            (SELECT c.id,
                    count(DISTINCT src.id)::INTEGER,
                    count(DISTINCT s.id)::INTEGER,
                    count(o.sentence_id)::INTEGER
             FROM corpus c
             LEFT JOIN source src ON src.corpus_id = c.id
             LEFT JOIN sentence s ON s.source_id = src.id
             LEFT JOIN collocation_occurrence o ON o.sentence_id = s.id
             GROUP BY c.id
             EXCEPT SELECT * FROM corpus_stats)
        )
        """
    ).fetchone()[0]
    if not corpus_stats_match:
        raise ArtifactBuildError("corpus_stats_mismatch")
