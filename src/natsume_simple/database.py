from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
            pos TEXT NOT NULL
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
            "end" INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS collocation (
            word_1_sw_id INTEGER NOT NULL REFERENCES sentence_word(id),
            particle_sw_id INTEGER NOT NULL REFERENCES sentence_word(id),
            word_2_sw_id INTEGER NOT NULL REFERENCES sentence_word(id)
        );
    """)

    return conn


def insert_sources_batch(
    conn: duckdb.DuckDBPyConnection, sources: List[Dict[str, Any]]
) -> Dict[Tuple[str, str], int]:
    """Insert multiple sources in batch and return mapping of (corpus, title) to IDs.

    Args:
        conn: Database connection
        sources: List of source dictionaries with corpus, title, year, author, publisher

    Returns:
        Dictionary mapping (corpus, title) tuples to source IDs
    """
    if not sources:
        logger.warning("No sources to insert")
        return {}

    # Convert to DataFrame
    df = pl.DataFrame(sources)
    conn.register("__temp_sources", df)

    # Bulk insert and get IDs
    conn.execute("""
        INSERT INTO source (corpus, year, title, author, publisher)
        SELECT corpus, year, title, author, publisher
        FROM __temp_sources
        RETURNING id, corpus, title
    """)
    results = conn.fetchall()

    conn.unregister("__temp_sources")

    # Create mapping of (corpus, title) to ID
    return {(r[1], r[2]): r[0] for r in results}


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


def bulk_insert_sentences(
    conn: duckdb.DuckDBPyConnection,
    sentences_df: pl.DataFrame,
) -> None:
    """Bulk insert sentences from a DataFrame.

    Args:
        conn: Database connection
        sentences_df: DataFrame with 'text' and 'source_id' columns
    """
    logger.info(f"Inserting {len(sentences_df)} sentences...")
    conn.register("__temp_sentences", sentences_df)
    conn.execute("""
        INSERT INTO sentence (text, source_id)
        SELECT text, source_id
        FROM __temp_sentences
    """)
    conn.unregister("__temp_sentences")
    conn.commit()


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
    source_id_map: Optional[Dict[Tuple[str, str], int]] = None,
) -> int:
    """Save a corpus to the database and return sentence IDs.

    Args:
        conn: Database connection
        corpus_name: Name of the corpus
        sentences: List of sentences
        year: Year of publication
        title: Title of the source
        author: Optional author name
        publisher: Optional publisher name
        source_id_map: Optional mapping of (corpus, title) to source IDs
    """
    logger.info(
        f"Saving corpus {corpus_name} and {len(sentences)} sentences to database"
    )

    # Get source ID from map or insert new
    if source_id_map is not None:
        source_id = source_id_map.get((corpus_name, title))
        if source_id is None:
            logger.warning(
                f"Source {title} not found in ID map, inserting individually"
            )
            source_id = insert_source(
                conn,
                corpus=corpus_name,
                title=title,
                year=year,
                author=author,
                publisher=publisher,
            )
    else:
        source_id = insert_source(
            conn,
            corpus=corpus_name,
            title=title,
            year=year,
            author=author,
            publisher=publisher,
        )

    # Convert sentences to DataFrame format
    sentences_df = pl.DataFrame(
        [{"text": text, "source_id": source_id} for text in sentences]
    )
    bulk_insert_sentences(conn, sentences_df)
    return len(sentences)


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


def create_indices(conn: duckdb.DuckDBPyConnection) -> None:
    """Create all database indices.

    Args:
        conn: Database connection
    """
    logger.info("Creating database indices...")
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_lemma_string ON lemma(string);
        CREATE INDEX IF NOT EXISTS idx_word_string ON word(string);
        CREATE INDEX IF NOT EXISTS idx_sentence_word_pos ON sentence_word(begin, "end");
        CREATE INDEX IF NOT EXISTS idx_colloc_word_1 ON collocation(word_1_sw_id);
        CREATE INDEX IF NOT EXISTS idx_colloc_word_2 ON collocation(word_2_sw_id);
    """)


def bulk_insert_lemmas(
    conn: duckdb.DuckDBPyConnection,
    lemmas: List[Dict[str, Any]],
) -> Dict[Tuple[str, str], int]:
    """Bulk insert lemmas and return mapping of (string, pos) to IDs."""
    if not lemmas:
        return {}

    df = pl.DataFrame(lemmas)
    conn.register("__temp_lemmas", df)

    # Insert new lemmas and get all IDs (both new and existing)
    conn.execute("""
        INSERT OR IGNORE INTO lemma (string, pos)
        SELECT DISTINCT string, pos FROM __temp_lemmas;
        
        SELECT id, string, pos FROM lemma
        WHERE (string, pos) IN (
            SELECT string, pos FROM __temp_lemmas
        );
    """)
    results = conn.fetchall()

    conn.unregister("__temp_lemmas")
    return {(r[1], r[2]): r[0] for r in results}


def bulk_insert_words(
    conn: duckdb.DuckDBPyConnection,
    words: List[Dict[str, Any]],
) -> Dict[Tuple[str, int], int]:
    """Bulk insert words and return mapping of (string, lemma_id) to IDs."""
    if not words:
        return {}

    df = pl.DataFrame(words)
    conn.register("__temp_words", df)

    conn.execute("""
        INSERT INTO word (string, pron, inf, dep, lemma_id)
        SELECT string, pron, inf, dep, lemma_id
        FROM __temp_words
        RETURNING id, string, lemma_id;
    """)
    results = conn.fetchall()

    conn.unregister("__temp_words")
    return {(r[1], r[2]): r[0] for r in results}


def bulk_insert_sentence_words(
    conn: duckdb.DuckDBPyConnection,
    sentence_words: List[Dict[str, Any]],
) -> Dict[Tuple[int, int, int, int], int]:
    """Bulk insert sentence_words and return mapping of (sentence_id, word_id, begin, end) to IDs."""
    if not sentence_words:
        return {}

    df = pl.DataFrame(sentence_words)
    conn.register("__temp_sentence_words", df)

    conn.execute("""
        INSERT INTO sentence_word (sentence_id, word_id, begin, "end")
        SELECT sentence_id, word_id, begin, "end"
        FROM __temp_sentence_words
        RETURNING id, sentence_id, word_id, begin, "end";
    """)
    results = conn.fetchall()

    conn.unregister("__temp_sentence_words")
    return {(r[1], r[2], r[3], r[4]): r[0] for r in results}


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

    # Create DataFrame and register temporary view
    df = pl.DataFrame(data)
    conn.register("__temp_collocations", df)

    # Use the registered view for insert
    conn.execute(
        """
        INSERT INTO collocation 
        (word_1_sw_id, particle_sw_id, word_2_sw_id)
        SELECT 
            word_1_sw_id, 
            particle_sw_id, 
            word_2_sw_id
        FROM __temp_collocations
        WHERE NOT EXISTS (
            SELECT 1 FROM collocation 
            WHERE word_1_sw_id = __temp_collocations.word_1_sw_id
            AND particle_sw_id = __temp_collocations.particle_sw_id
            AND word_2_sw_id = __temp_collocations.word_2_sw_id
        )
        """
    )

    # Unregister the temporary view
    conn.unregister("__temp_collocations")
