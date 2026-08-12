import hashlib
import json
from pathlib import Path

import duckdb

from natsume_simple.database import init_database


def build_search_database(path: Path) -> Path:
    """Build the smallest database that exercises every legacy query direction."""
    conn = init_database(path)
    conn.execute(
        """
        INSERT INTO source (id, year, title, corpus) VALUES
            (1, 2024, 'Alpha one', 'alpha'),
            (2, 2024, 'Alpha two', 'alpha'),
            (3, 2025, 'Beta one', 'beta');

        INSERT INTO sentence (id, text, source_id) VALUES
            (1, '情報を集める。', 1),
            (2, '情報を安全に集める。', 2),
            (3, '<img src=x onerror=alert(1)>情報を集める。', 3),
            (4, '研究が進む。', 1),
            (5, '情報が進む。', 3);

        INSERT INTO lemma (id, string, pos) VALUES
            (1, '情報', 'NOUN'),
            (2, 'を', 'ADP'),
            (3, '集める', 'VERB'),
            (4, '研究', 'NOUN'),
            (5, 'が', 'ADP'),
            (6, '進む', 'VERB');

        INSERT INTO word (id, string, pron, inf, dep, lemma_id) VALUES
            (1, '情報', 'ジョウホウ', NULL, 'obj', 1),
            (2, 'を', 'ヲ', NULL, 'case', 2),
            (3, '集める', 'アツメル', NULL, 'ROOT', 3),
            (4, '研究', 'ケンキュウ', NULL, 'nsubj', 4),
            (5, 'が', 'ガ', NULL, 'case', 5),
            (6, '進む', 'ススム', NULL, 'ROOT', 6);

        INSERT INTO sentence_word (id, sentence_id, word_id, begin, "end") VALUES
            (1, 1, 1, 0, 2), (2, 1, 2, 2, 3), (3, 1, 3, 3, 6),
            (4, 2, 1, 0, 2), (5, 2, 2, 2, 3), (6, 2, 3, 6, 9),
            (7, 3, 1, 28, 30), (8, 3, 2, 30, 31), (9, 3, 3, 31, 34),
            (10, 4, 4, 0, 2), (11, 4, 5, 2, 3), (12, 4, 6, 3, 5),
            (13, 5, 1, 0, 2), (14, 5, 5, 2, 3), (15, 5, 6, 3, 5);

        INSERT INTO collocation VALUES
            (1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12), (13, 14, 15);
        """
    )
    conn.close()
    return path


def build_search_artifact(directory: Path) -> Path:
    """Build a schema-v1 artifact with two corpora and selection-sensitive data."""
    directory.mkdir()
    database_path = directory / "corpus.duckdb"
    conn = duckdb.connect(str(database_path))
    conn.execute(
        """
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
            extractor_id TEXT NOT NULL
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

        INSERT INTO build_metadata VALUES
            (1, 'fixture-build-001', 'fixture', 'fixture-extractor', '{}',
             'fixture-sources', '2026-08-12T00:00:00Z');
        INSERT INTO corpus VALUES ('alpha', 'Alpha'), ('beta', 'Beta');
        INSERT INTO source VALUES
            (1, 'alpha', 'a1', 'Alpha one', 'sha-a1'),
            (2, 'alpha', 'a2', 'Alpha two', 'sha-a2'),
            (3, 'beta', 'b1', 'Beta one', 'sha-b1');
        INSERT INTO sentence VALUES
            (1, 1, 1, '情報を集める。'),
            (2, 2, 1, '情報を安全に集める。'),
            (3, 3, 1, '<img src=x onerror=alert(1)>情報を集める。'),
            (4, 1, 2, '研究が進める。'),
            (5, 3, 2, '情報が進める。'),
            (6, 1, 3, '情報を分析する。'),
            (7, 1, 4, '情報を分析する。'),
            (8, 1, 5, '情報を分析する。'),
            (9, 3, 3, '情報を調べる。'),
            (10, 3, 4, '情報を調べる。');
        INSERT INTO collocation_occurrence VALUES
            (1, '情報', 'を', '集める', 0, 2, 2, 3, 3, 6, 'fixture-extractor'),
            (2, '情報', 'を', '集める', 0, 2, 2, 3, 6, 9, 'fixture-extractor'),
            (3, '情報', 'を', '集める', 28, 30, 30, 31, 31, 34, 'fixture-extractor'),
            (4, '研究', 'が', '進める', 0, 2, 2, 3, 3, 6, 'fixture-extractor'),
            (5, '情報', 'が', '進める', 0, 2, 2, 3, 3, 6, 'fixture-extractor'),
            (6, '情報', 'を', '分析する', 0, 2, 2, 3, 3, 7, 'fixture-extractor'),
            (7, '情報', 'を', '分析する', 0, 2, 2, 3, 3, 7, 'fixture-extractor'),
            (8, '情報', 'を', '分析する', 0, 2, 2, 3, 3, 7, 'fixture-extractor'),
            (9, '情報', 'を', '調べる', 0, 2, 2, 3, 3, 6, 'fixture-extractor'),
            (10, '情報', 'を', '調べる', 0, 2, 2, 3, 3, 6, 'fixture-extractor');
        INSERT INTO corpus_stats VALUES
            ('alpha', 2, 6, 6), ('beta', 1, 4, 4);
        INSERT INTO lemma_frequency VALUES
            ('noun', '情報', 9), ('noun', '研究', 1),
            ('verb', '分析する', 3), ('verb', '集める', 3),
            ('verb', '調べる', 2), ('verb', '進める', 2);
        """
    )
    conn.close()

    checksum = hashlib.sha256(database_path.read_bytes()).hexdigest()
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "artifactInstanceId": "fixture-build-001",
                "schemaVersion": 1,
                "databaseSha256": checksum,
            }
        ),
        encoding="utf-8",
    )
    return directory
