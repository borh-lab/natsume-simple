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


def build_maximum_response_artifact(directory: Path) -> Path:
    """Build valid maximum-cardinality rows for public response sizing."""
    directory = build_search_artifact(directory)
    database_path = directory / "corpus.duckdb"
    with duckdb.connect(str(database_path)) as conn:
        conn.execute(
            """
            DELETE FROM collocation_occurrence;
            DELETE FROM lemma_frequency;
            DELETE FROM corpus_stats;
            DELETE FROM sentence;
            DELETE FROM source;
            DELETE FROM corpus;

            INSERT INTO corpus VALUES
                ('max-a', 'Maximum A'),
                ('max-b', 'Maximum B'),
                ('max-c', 'Maximum C');
            INSERT INTO source VALUES
                (20, 'max-a', 'max-a', 'Maximum A', 'sha-max-a'),
                (21, 'max-b', 'max-b', 'Maximum B', 'sha-max-b'),
                (22, 'max-c', 'max-c', 'Maximum C', 'sha-max-c');

            CREATE TEMP TABLE maximum_rows AS
            SELECT corpus_id,
                   source_id,
                   sentence_base + particle_order * 200 + item + 1 AS sentence_id,
                   particle_order * 200 + item + 1 AS ordinal,
                   repeat('名', 64) AS noun,
                   particle,
                   repeat('動', 61) || lpad(item::VARCHAR, 3, '0') AS verb
            FROM (VALUES
                    ('max-a', 20, 0),
                    ('max-b', 21, 2000),
                    ('max-c', 22, 4000)
                 ) corpora(corpus_id, source_id, sentence_base)
            CROSS JOIN (VALUES
                    (0, 'が'), (1, 'を'), (2, 'に'), (3, 'で'),
                    (4, 'から'), (5, 'より'), (6, 'と'), (7, 'へ')
                 ) particles(particle_order, particle)
            CROSS JOIN range(200) items(item);

            INSERT INTO sentence
            SELECT sentence_id, source_id, ordinal, noun || particle || verb
            FROM maximum_rows;

            INSERT INTO collocation_occurrence
            SELECT sentence_id,
                   noun,
                   particle,
                   verb,
                   0, 64,
                   64, 64 + length(particle),
                   64 + length(particle), 64 + length(particle) + 64,
                   'fixture-extractor'
            FROM maximum_rows;

            INSERT INTO corpus_stats VALUES
                ('max-a', 21, 1620, 1620),
                ('max-b', 1, 1600, 1600),
                ('max-c', 1, 1600, 1600);
            INSERT INTO lemma_frequency VALUES ('noun', repeat('名', 64), 4800);

            INSERT INTO source
            SELECT 100 + item,
                   'max-a',
                   'example-' || item,
                   repeat('題', 512),
                   'sha-example-' || item
            FROM range(20) items(item);
            INSERT INTO sentence
            SELECT 10000 + item, 100 + item, 1, '例を示す' || repeat('文', 4092)
            FROM range(20) items(item);
            INSERT INTO collocation_occurrence
            SELECT 10000 + item, '例', 'を', '示す', 0, 1, 1, 2, 2, 4,
                   'fixture-extractor'
            FROM range(20) items(item);

            INSERT INTO lemma_frequency VALUES
                ('noun', '例', 20),
                ('verb', '示す', 20);
            INSERT INTO lemma_frequency
            SELECT 'verb', repeat('動', 61) || lpad(item::VARCHAR, 3, '0'), 24
            FROM range(200) items(item);
            """
        )

    checksum = hashlib.sha256(database_path.read_bytes()).hexdigest()
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["databaseSha256"] = checksum
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return directory
