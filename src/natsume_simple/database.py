from pathlib import Path
from typing import List, Optional, Tuple

import duckdb
import polars as pl

from natsume_simple.log import setup_logger

logger = setup_logger(__name__)


def init_database(db_path: Path) -> duckdb.DuckDBPyConnection:
    """Initialize the database with the schema."""
    conn = duckdb.connect(str(db_path))

    # Create schema
    conn.execute("""
        CREATE SEQUENCE IF NOT EXISTS id_seq_source START 1;
        CREATE TABLE IF NOT EXISTS source (
            id INTEGER PRIMARY KEY DEFAULT NEXTVAL('id_seq_source'),
            year INTEGER NOT NULL,
            title TEXT NOT NULL,
            author TEXT,
            publisher TEXT,
            corpus TEXT
        );

        CREATE SEQUENCE IF NOT EXISTS id_seq_sentence START 1;
        CREATE TABLE IF NOT EXISTS sentence (
            id INTEGER PRIMARY KEY DEFAULT NEXTVAL('id_seq_sentence'),
            text TEXT NOT NULL,
            source_id INTEGER NOT NULL REFERENCES source(id)
        );

        CREATE SEQUENCE IF NOT EXISTS id_seq_lemma START 1;
        CREATE TABLE IF NOT EXISTS lemma (
            id INTEGER PRIMARY KEY DEFAULT NEXTVAL('id_seq_lemma'),
            string TEXT NOT NULL,
            pos TEXT NOT NULL,
            UNIQUE(string, pos)
        );

        CREATE SEQUENCE IF NOT EXISTS id_seq_word START 1;
        CREATE TABLE IF NOT EXISTS word (
            id INTEGER PRIMARY KEY DEFAULT NEXTVAL('id_seq_word'),
            string TEXT NOT NULL,
            pron TEXT NOT NULL,      -- Pronunciation ("Reading" field from GiNZA)
            inf TEXT,                -- Inflection type (can be NULL)
            dep TEXT NOT NULL,       -- UD dependency relation
            lemma_id INTEGER NOT NULL REFERENCES lemma(id)
        );

        CREATE SEQUENCE IF NOT EXISTS id_seq_sentence_word START 1;
        CREATE TABLE IF NOT EXISTS sentence_word (
            id INTEGER PRIMARY KEY DEFAULT NEXTVAL('id_seq_sentence_word'),
            sentence_id INTEGER NOT NULL REFERENCES sentence(id),
            word_id INTEGER NOT NULL REFERENCES word(id),
            begin INTEGER NOT NULL,
            "end" INTEGER NOT NULL,
            UNIQUE(sentence_id, begin, "end")
        );
        CREATE INDEX IF NOT EXISTS idx_sentence_word_pos ON sentence_word(begin, "end");

        CREATE TABLE IF NOT EXISTS collocation (
            word_1_sw_id INTEGER NOT NULL REFERENCES sentence_word(id),
            particle_sw_id INTEGER NOT NULL REFERENCES sentence_word(id),
            word_2_sw_id INTEGER NOT NULL REFERENCES sentence_word(id),
            UNIQUE(word_1_sw_id, particle_sw_id, word_2_sw_id) -- Not strictly necessary, just data validation
        );
        
        CREATE INDEX IF NOT EXISTS idx_colloc_words ON collocation(word_1_sw_id, word_2_sw_id);
    """)

    return conn


def insert_source(
    conn: duckdb.DuckDBPyConnection,
    corpus: str,
    title: str,
    year: Optional[int] = None,
    author: Optional[str] = None,
    publisher: Optional[str] = None,
) -> int:
    """Insert a source and return its ID."""
    conn.execute(
        """
        INSERT INTO source (corpus, year, title, author, publisher)
        VALUES (?, ?, ?, ?, ?)
        RETURNING id
    """,
        [corpus, year, title, author, publisher],
    )
    return conn.fetchone()[0]


def insert_sentences(
    conn: duckdb.DuckDBPyConnection,
    sentences: List[str],
    source_id: int,
) -> int:
    """Insert sentences and return their IDs.

    Args:
        conn: Database connection
        sentences: List of sentence texts
        source_id: ID of the source document

    Returns:
        ID of the first inserted sentence
    """
    conn.execute(
        """
        INSERT INTO sentence (text, source_id) 
        SELECT unnest(?::VARCHAR[]), ? 
        RETURNING id
        """,
        [sentences, source_id],
    )
    return conn.fetchone()[0]


def get_or_create_lemma(conn: duckdb.DuckDBPyConnection, string: str, pos: str) -> int:
    """Get lemma ID or create if not exists."""
    # Try to insert, ignore if already exists
    conn.execute(
        """
        INSERT OR IGNORE INTO lemma (string, pos) 
        VALUES (?, ?)
        """,
        [string, pos],
    )

    # Get the ID (whether from existing or newly inserted row)
    result = conn.execute(
        "SELECT id FROM lemma WHERE string = ? AND pos = ?", [string, pos]
    ).fetchone()
    return result[0]


def get_or_create_word(
    conn: duckdb.DuckDBPyConnection,
    string: str,
    pron: str,
    inf: Optional[str],
    dep: str,
    lemma_id: int,
) -> int:
    """Get word ID or create if not exists."""
    conn.execute(
        """
        INSERT INTO word (string, pron, inf, dep, lemma_id) 
        VALUES (?, ?, ?, ?, ?)
        """,
        [string, pron, inf, dep, lemma_id],
    )
    result = conn.execute(
        "SELECT id FROM word WHERE string = ? AND lemma_id = ?", [string, lemma_id]
    ).fetchone()
    return result[0]


def get_or_create_sentence_word(
    conn: duckdb.DuckDBPyConnection,
    sentence_id: int,
    word_id: int,
    begin: int,
    end: int,
) -> int:
    """Get sentence_word ID or create if not exists."""
    conn.execute(
        """
        INSERT INTO sentence_word (sentence_id, word_id, begin, "end")
        VALUES (?, ?, ?, ?)
        """,
        [sentence_id, word_id, begin, end],
    )
    result = conn.execute(
        """
        SELECT id FROM sentence_word 
        WHERE sentence_id = ? AND word_id = ? AND begin = ? AND "end" = ?
        """,
        [sentence_id, word_id, begin, end],
    ).fetchone()
    return result[0]


def save_corpus_to_db(
    conn: duckdb.DuckDBPyConnection,
    corpus_name: str,
    sentences: List[str],
    year: int,
    title: str,
    author: Optional[str] = None,
    publisher: Optional[str] = None,
) -> int:
    """Save a corpus to the database and return sentence IDs."""
    source_id = insert_source(
        conn,
        corpus=corpus_name,
        title=title,
        year=year,
        author=author,
        publisher=publisher,
    )
    return insert_sentences(conn, sentences, source_id)


def get_sentence_word_id(
    conn: duckdb.DuckDBPyConnection, sentence_id: int, begin: int, end: int
) -> Optional[int]:
    """Get sentence_word ID for a given span."""
    result = conn.execute(
        """
        SELECT id FROM sentence_word 
        WHERE sentence_id = ? 
        AND begin = ? 
        AND "end" = ?
        """,
        [sentence_id, begin, end],
    ).fetchone()
    return result[0] if result else None


def save_collocations(
    conn: duckdb.DuckDBPyConnection,
    collocations: List[Tuple[int, int, int]],
) -> None:
    """Save collocations to database in bulk.

    Args:
        conn: Database connection
        collocations: List of (word_1_sw_id, particle_sw_id, word_2_sw_id) tuples
    """
    # Convert list of tuples to list of dicts for easier processing
    data = [
        {
            "word_1_sw_id": c[0],
            "particle_sw_id": c[1],
            "word_2_sw_id": c[2],
        }
        for c in collocations
    ]

    # Use DuckDB's from_arrow for efficient bulk insert
    conn.execute(
        """
        INSERT INTO collocation 
        (word_1_sw_id, particle_sw_id, word_2_sw_id)
        SELECT 
            word_1_sw_id, 
            particle_sw_id, 
            word_2_sw_id
        FROM __data
        WHERE NOT EXISTS (
            SELECT 1 FROM collocation 
            WHERE word_1_sw_id = __data.word_1_sw_id
            AND particle_sw_id = __data.particle_sw_id
            AND word_2_sw_id = __data.word_2_sw_id
        )
    """,
        {"__data": pl.DataFrame(data)},
    )
