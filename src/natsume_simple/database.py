import duckdb
from pathlib import Path
from typing import List, Optional, Tuple

from natsume_simple.log import setup_logger

logger = setup_logger(__name__)


def init_database(db_path: Path) -> duckdb.DuckDBPyConnection:
    """Initialize the database with the schema."""
    conn = duckdb.connect(str(db_path))

    # Create schema
    conn.execute("""
        CREATE TABLE IF NOT EXISTS source (
            id INTEGER PRIMARY KEY,
            year INTEGER NOT NULL,
            title TEXT NOT NULL,
            author TEXT,
            publisher TEXT,
            corpus TEXT
        );

        CREATE TABLE IF NOT EXISTS sentence (
            id INTEGER PRIMARY KEY,
            text TEXT NOT NULL,
            source_id INTEGER NOT NULL REFERENCES source(id)
        );

        CREATE TABLE IF NOT EXISTS lemma (
            id INTEGER PRIMARY KEY,
            string TEXT NOT NULL,
            pos TEXT NOT NULL,
            UNIQUE(string, pos)
        );

        CREATE TABLE IF NOT EXISTS word (
            id INTEGER PRIMARY KEY,
            string TEXT NOT NULL,
            pron TEXT NOT NULL,
            inf TEXT,
            dep TEXT NOT NULL,
            lemma_id INTEGER NOT NULL REFERENCES lemma(id)
        );

        CREATE TABLE IF NOT EXISTS sentence_word (
            id INTEGER PRIMARY KEY,
            sentence_id INTEGER NOT NULL REFERENCES sentence(id),
            word_id INTEGER NOT NULL REFERENCES word(id),
            begin INTEGER NOT NULL,
            end INTEGER NOT NULL
        );

        CREATE TABLE IF NOT EXISTS collocation (
            word_1_sw_id INTEGER NOT NULL REFERENCES sentence_word(id),
            particle_sw_id INTEGER NOT NULL REFERENCES sentence_word(id),
            word_2_sw_id INTEGER NOT NULL REFERENCES sentence_word(id)
        );
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
        INSERT INTO source (year, title, author, publisher, corpus)
        VALUES (?, ?, ?, ?, ?)
        RETURNING id
    """,
        [year, title, author, publisher, corpus],
    )
    return conn.fetchone()[0]


def insert_sentences(
    conn: duckdb.DuckDBPyConnection, sentences: List[str], source_id: int
) -> None:
    """Insert sentences for a given source."""
    conn.execute(
        """
        INSERT INTO sentence (text, source_id)
        SELECT text, ? FROM (SELECT UNNEST(?) as text)
    """,
        [source_id, sentences],
    )


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
        INSERT INTO sentence_word (sentence_id, word_id, begin, end)
        VALUES (?, ?, ?, ?)
        """,
        [sentence_id, word_id, begin, end],
    )
    result = conn.execute(
        """
        SELECT id FROM sentence_word 
        WHERE sentence_id = ? AND word_id = ? AND begin = ? AND end = ?
        """,
        [sentence_id, word_id, begin, end],
    ).fetchone()
    return result[0]


def save_corpus_to_db(
    conn: duckdb.DuckDBPyConnection,
    corpus_name: str,
    sentences: List[str],
    year: Optional[int] = None,
) -> None:
    """Save a corpus to the database."""
    source_id = insert_source(
        conn, corpus=corpus_name, title=f"{corpus_name} corpus", year=year
    )
    insert_sentences(conn, sentences, source_id)
    logger.info(f"Saved {len(sentences)} sentences from {corpus_name} corpus")


def save_collocations_to_db(
    db_path: Path, collocations: List[Tuple[str, str, str]], corpus_name: str
) -> None:
    """
    Save collocations to database, linking to existing sentences.

    Args:
        db_path: Path to database file
        collocations: List of (noun, particle, verb) tuples
        corpus_name: Name of the corpus
    """
    conn = duckdb.connect(str(db_path))

    try:
        # Get source ID for this corpus
        source_id = conn.execute(
            "SELECT id FROM source WHERE corpus = ?", [corpus_name]
        ).fetchone()[0]

        # Get all sentences for this source
        sentences = conn.execute(
            "SELECT id, text FROM sentence WHERE source_id = ?", [source_id]
        ).fetchdf()

        for noun, particle, verb in collocations:
            # Find sentences containing this collocation pattern
            pattern = f"{noun}{particle}.*{verb}"
            matching_sentences = sentences[sentences["text"].str.contains(pattern)]

            if matching_sentences.empty:
                logger.debug(f"No matching sentences found for pattern: {pattern}")
                continue

            # Get or create lemmas
            noun_lemma_id = get_or_create_lemma(conn, noun, "NOUN")
            particle_lemma_id = get_or_create_lemma(conn, particle, "ADP")
            verb_lemma_id = get_or_create_lemma(conn, verb, "VERB")

            # Get or create words
            noun_word_id = get_or_create_word(
                conn, noun, noun, None, "nsubj", noun_lemma_id
            )
            particle_word_id = get_or_create_word(
                conn, particle, particle, None, "case", particle_lemma_id
            )
            verb_word_id = get_or_create_word(
                conn, verb, verb, None, "ROOT", verb_lemma_id
            )

            for _, sentence in matching_sentences.iterrows():
                # Find positions of words in the sentence
                text = sentence["text"]
                noun_pos = text.find(noun)
                if noun_pos == -1:
                    continue

                particle_pos = text.find(particle, noun_pos + len(noun))
                if particle_pos == -1:
                    continue

                verb_pos = text.find(verb, particle_pos + len(particle))
                if verb_pos == -1:
                    continue

                # Create sentence_word entries
                noun_sw_id = get_or_create_sentence_word(
                    conn, sentence["id"], noun_word_id, noun_pos, noun_pos + len(noun)
                )
                particle_sw_id = get_or_create_sentence_word(
                    conn,
                    sentence["id"],
                    particle_word_id,
                    particle_pos,
                    particle_pos + len(particle),
                )
                verb_sw_id = get_or_create_sentence_word(
                    conn, sentence["id"], verb_word_id, verb_pos, verb_pos + len(verb)
                )

                # Create collocation entry
                conn.execute(
                    """
                    INSERT INTO collocation (word_1_sw_id, particle_sw_id, word_2_sw_id)
                    VALUES (?, ?, ?)
                    ON CONFLICT DO NOTHING
                    """,
                    [noun_sw_id, particle_sw_id, verb_sw_id],
                )

        conn.commit()
    finally:
        conn.close()
