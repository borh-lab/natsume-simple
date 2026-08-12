import hashlib
import json
from pathlib import Path

import duckdb

from natsume_simple.artifact_builder import create_schema_v1


def build_search_artifact(directory: Path) -> Path:
    """Build a schema-v1 artifact with two corpora and selection-sensitive data."""
    directory.mkdir()
    database_path = directory / "corpus.duckdb"
    conn = duckdb.connect(str(database_path))
    create_schema_v1(conn)
    conn.execute(
        """
        INSERT INTO build_metadata VALUES
            (1, 'fixture-build-001', 'fixture', 'fixture-extractor', '{}',
             'fixture-sources', '2026-08-12T00:00:00Z');
        INSERT INTO corpus VALUES ('alpha', 'Alpha'), ('beta', 'Beta');
        INSERT INTO source (id, corpus_id, external_id, title, content_sha256) VALUES
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
            INSERT INTO source (id, corpus_id, external_id, title, content_sha256) VALUES
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

            INSERT INTO source (id, corpus_id, external_id, title, content_sha256)
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
