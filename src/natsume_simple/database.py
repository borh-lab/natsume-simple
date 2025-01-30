import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, TypeAlias

import duckdb
import polars as pl

from natsume_simple.log import setup_logger

WordKey: TypeAlias = Tuple[str, str, Optional[str], str, str, str, int]

logger = setup_logger(__name__)


def _make_word_key(
    string: str, pron: str, inf: Optional[str], dep: str, lemma_id: int
) -> WordKey:
    """Create a consistent tuple key for word mappings."""
    # Include all fields that make a word unique
    return (string, pron, inf, dep, str(lemma_id), str(lemma_id), lemma_id)


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

        CREATE TABLE IF NOT EXISTS lemma (
            id INTEGER PRIMARY KEY,
            string TEXT NOT NULL,
            pos TEXT NOT NULL
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
        RETURNING id, corpus, title;
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
    if len(sentences_df) == 0:
        return

    logger.info(f"Inserting {len(sentences_df)} sentences...")
    conn.register("__temp_sentences", sentences_df)
    conn.execute("""
        INSERT INTO sentence (text, source_id)
        SELECT text, source_id
        FROM __temp_sentences
    """)
    conn.unregister("__temp_sentences")


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


def clean_pattern_data(conn: duckdb.DuckDBPyConnection) -> None:
    """Clean all pattern-related data from database while preserving corpus data.

    This removes all lemmas, words, sentence_words, and collocations but keeps
    sources and sentences intact.
    """
    logger.info("Cleaning pattern-related data from database...")

    # Delete in correct order to respect foreign key constraints
    conn.execute("""
        -- First drop all sequences since we'll recreate them
        DROP SEQUENCE IF EXISTS id_seq_lemma;
        DROP SEQUENCE IF EXISTS id_seq_word;
        DROP SEQUENCE IF EXISTS id_seq_sentence_word;
        
        -- Delete tables in reverse order of dependencies
        DELETE FROM collocation;
        DELETE FROM sentence_word;
        DELETE FROM word;
        DELETE FROM lemma;
        
        -- Recreate sequences starting from 1
        CREATE SEQUENCE id_seq_lemma START 1;
        CREATE SEQUENCE id_seq_word START 1;
        CREATE SEQUENCE id_seq_sentence_word START 1;
    """)
    logger.info("Pattern data cleaned successfully")


def get_table_counts(conn: duckdb.DuckDBPyConnection) -> Dict[str, int]:
    """Get row counts for all tables in database."""
    tables = ["source", "sentence", "lemma", "word", "sentence_word", "collocation"]
    counts = {}
    for table in tables:
        result = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()
        counts[table] = result[0]
    return counts


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Database management operations")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing the database (default: ./data)",
    )
    parser.add_argument(
        "--action",
        choices=["clean-patterns", "show-counts"],
        required=True,
        help="Action to perform",
    )

    args = parser.parse_args()

    db_path = args.data_dir / "corpus.db"
    if not db_path.exists():
        logger.error(f"Database not found at {db_path}")
        exit(1)

    conn = duckdb.connect(str(db_path))
    try:
        if args.action == "clean-patterns":
            clean_pattern_data(conn)
            logger.info("Pattern data cleaned successfully")

        elif args.action == "show-counts":
            counts = get_table_counts(conn)
            for table, count in counts.items():
                print(f"{table}: {count:,} rows")

    finally:
        conn.close()


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


class BulkIngestCollector:
    """Collector for bulk database ingestion."""

    def __init__(self, conn: duckdb.DuckDBPyConnection):
        self.conn = conn
        # Use tuples of all relevant columns as keys
        self.lemmas: Dict[Tuple[str, str], Dict[str, Any]] = {}
        self.words: Dict[WordKey, Dict[str, Any]] = {}
        self.sentence_words: List[Dict[str, Any]] = []
        self.collocations: List[Dict[str, int]] = []

        # Map complete tuples to IDs
        self._lemma_id_map: Dict[Tuple[str, str], int] = {}  # (string, pos) -> id
        self._word_id_map: Dict[
            WordKey, int
        ] = {}  # (string, pron, inf, dep, str_lemma_id, str_lemma_id, lemma_id) -> id
        self._sentence_word_id_map: Dict[
            Tuple[int, int, int, int], int
        ] = {}  # (sentence_id, word_id, begin, end) -> id

        # Initialize next IDs from database
        self._init_next_ids()

    def _init_next_ids(self) -> None:
        """Initialize next IDs and mappings from current database state."""
        # Get max IDs from each table
        result = self.conn.execute("""
            SELECT 
                COALESCE(MAX(id), 0) + 1 as next_lemma_id,
                COALESCE((SELECT MAX(id) FROM word), 0) + 1 as next_word_id,
                COALESCE((SELECT MAX(id) FROM sentence_word), 0) + 1 as next_sentence_word_id
            FROM lemma
        """).fetchone()

        self._next_lemma_id = result[0]
        self._next_word_id = result[1]
        self._next_sentence_word_id = result[2]

        # Load existing lemma mappings
        existing_lemmas = self.conn.execute("""
            SELECT id, string, pos FROM lemma
        """).fetchall()
        for lemma_id, string, pos in existing_lemmas:
            self._lemma_id_map[(string, pos)] = lemma_id

        # Load existing word mappings
        existing_words = self.conn.execute("""
            SELECT id, string, pron, inf, dep, lemma_id
            FROM word
        """).fetchall()
        for word_id, string, pron, inf, dep, lemma_id in existing_words:
            word_key = _make_word_key(string, pron, inf, dep, lemma_id)
            self._word_id_map[word_key] = word_id

    def add_lemma(self, string: str, pos: str) -> int:
        """Add lemma and return its ID."""
        key = (string, pos)
        if key not in self._lemma_id_map:
            self._lemma_id_map[key] = self._next_lemma_id
            self.lemmas[key] = {"string": string, "pos": pos}
            self._next_lemma_id += 1
        return self._lemma_id_map[key]

    def add_word(
        self, string: str, pron: str, inf: Optional[str], dep: str, lemma_id: int
    ) -> int:
        """Add word and return its ID."""
        key = _make_word_key(string, pron, inf, dep, lemma_id)

        if key not in self._word_id_map:
            self._word_id_map[key] = self._next_word_id
            self.words[key] = {
                "string": string,
                "pron": pron,
                "inf": inf,
                "dep": dep,
                "lemma_id": lemma_id,
            }
            self._next_word_id += 1
        return self._word_id_map[key]

    def add_sentence_word(
        self, sentence_id: int, word_id: int, begin: int, end: int
    ) -> int:
        """Add sentence word and return its ID."""
        key = (sentence_id, word_id, begin, end)
        if key not in self._sentence_word_id_map:
            self._sentence_word_id_map[key] = self._next_sentence_word_id
            self.sentence_words.append(
                {
                    "sentence_id": sentence_id,
                    "word_id": word_id,
                    "begin": begin,
                    "end": end,
                }
            )
            self._next_sentence_word_id += 1
        return self._sentence_word_id_map[key]

    def add_collocation(
        self, word_1_sw_id: int, particle_sw_id: int, word_2_sw_id: int
    ) -> None:
        """Add collocation."""
        self.collocations.append(
            {
                "word_1_sw_id": word_1_sw_id,
                "particle_sw_id": particle_sw_id,
                "word_2_sw_id": word_2_sw_id,
            }
        )

    def bulk_insert_all(self, conn: duckdb.DuckDBPyConnection) -> None:
        """Insert all collected data into database using explicit IDs."""
        logger.info("Beginning bulk insert of all data...")

        # Create timestamp for file names
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        debug_dir = Path("data/debug")
        debug_dir.mkdir(parents=True, exist_ok=True)

        # Insert lemmas with explicit IDs
        if self.lemmas:
            logger.info(f"Inserting {len(self.lemmas)} lemmas...")
            lemma_data = [
                {"id": self._lemma_id_map[key], **lemma}
                for key, lemma in self.lemmas.items()
            ]
            df = pl.DataFrame(lemma_data)
            df.write_csv(debug_dir / f"insert_lemmas-{timestamp}.csv")

            conn.register("__temp_lemmas", df)
            conn.execute("""
                INSERT INTO lemma (id, string, pos)
                SELECT id, string, pos 
                FROM __temp_lemmas
            """)
            conn.unregister("__temp_lemmas")

        # Insert words with explicit IDs
        if self.words:
            logger.info(f"Inserting {len(self.words)} words...")
            word_data = [
                {"id": self._word_id_map[key], **word}
                for key, word in self.words.items()
            ]
            df = pl.DataFrame(word_data)
            df.write_csv(debug_dir / f"insert_words-{timestamp}.csv")

            conn.register("__temp_words", df)
            conn.execute("""
                INSERT INTO word (id, string, pron, inf, dep, lemma_id)
                SELECT id, string, pron, inf, dep, lemma_id
                FROM __temp_words
            """)
            conn.unregister("__temp_words")

        # Insert sentence words with explicit IDs
        if self.sentence_words:
            logger.info(f"Inserting {len(self.sentence_words)} sentence words...")
            sentence_word_data = [
                {
                    "id": self._sentence_word_id_map[
                        (sw["sentence_id"], sw["word_id"], sw["begin"], sw["end"])
                    ],
                    **sw,
                }
                for sw in self.sentence_words
            ]
            df = pl.DataFrame(sentence_word_data)
            df.write_csv(debug_dir / f"insert_sentence_words-{timestamp}.csv")

            conn.register("__temp_sentence_words", df)
            conn.execute("""
                INSERT INTO sentence_word (id, sentence_id, word_id, begin, "end")
                SELECT id, sentence_id, word_id, begin, "end"
                FROM __temp_sentence_words
            """)
            conn.unregister("__temp_sentence_words")

        # Insert collocations (no IDs needed)
        if self.collocations:
            logger.info(f"Inserting {len(self.collocations)} collocations...")
            df = pl.DataFrame(self.collocations)
            df.write_csv(debug_dir / f"insert_collocations-{timestamp}.csv")

            conn.register("__temp_collocations", df)
            conn.execute("""
                INSERT INTO collocation (word_1_sw_id, particle_sw_id, word_2_sw_id)
                SELECT word_1_sw_id, particle_sw_id, word_2_sw_id
                FROM __temp_collocations
            """)
            conn.unregister("__temp_collocations")

        logger.info("Bulk insert completed successfully")
