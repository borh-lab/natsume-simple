# /// script
# dependencies = [
#   "fastapi",
#   "polars",
#   "duckdb",
# ]
# ///

from pathlib import Path
from typing import Any, Dict, List

import duckdb
from fastapi import FastAPI  # type: ignore
from fastapi.middleware.cors import CORSMiddleware  # type: ignore
from fastapi.staticfiles import StaticFiles  # type: ignore

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_sentences_db() -> duckdb.DuckDBPyConnection:
    db_path = Path("data/corpus.db")
    return duckdb.connect(str(db_path), read_only=True)


def calculate_corpus_norm(conn: duckdb.DuckDBPyConnection) -> Dict[str, float]:
    """Calculate normalization factors for different corpora.

    Args:
        conn: Database connection

    Returns:
        Dictionary mapping corpus names to their normalization factors
    """
    corpus_freqs = conn.execute("""
        SELECT src.corpus, COUNT(*) as frequency
        FROM collocation c
        JOIN sentence_word sw ON c.word_1_sw_id = sw.id
        JOIN sentence s ON sw.sentence_id = s.id
        JOIN source src ON s.source_id = src.id
        GROUP BY src.corpus
    """).pl()

    min_count = corpus_freqs["frequency"].min()
    return {
        corpus: min_count / frequency
        for corpus, frequency in zip(corpus_freqs["corpus"], corpus_freqs["frequency"])
    }


db = load_sentences_db()
corpus_norm = calculate_corpus_norm(db)


@app.get("/corpus/norm")
def get_corpus_norm() -> Dict[str, float]:
    return corpus_norm


@app.get("/npv/noun/{noun}")
def read_npv_noun(noun: str) -> List[Dict[str, Any]]:
    matches = (
        db.execute(
            """
        WITH pattern_counts AS (
            SELECT 
                l1.string as n,
                l2.string as p,
                l3.string as v,
                src.corpus,
                COUNT(*) as frequency
            FROM collocation c
            JOIN sentence_word sw1 ON c.word_1_sw_id = sw1.id
            JOIN word w1 ON sw1.word_id = w1.id
            JOIN lemma l1 ON w1.lemma_id = l1.id
            JOIN sentence_word sw2 ON c.particle_sw_id = sw2.id
            JOIN word w2 ON sw2.word_id = w2.id
            JOIN lemma l2 ON w2.lemma_id = l2.id
            JOIN sentence_word sw3 ON c.word_2_sw_id = sw3.id
            JOIN word w3 ON sw3.word_id = w3.id
            JOIN lemma l3 ON w3.lemma_id = l3.id
            JOIN sentence s ON sw1.sentence_id = s.id
            JOIN source src ON s.source_id = src.id
            WHERE l1.string = ?
            GROUP BY l1.string, l2.string, l3.string, src.corpus
        )
        SELECT 
            n, p, v,
            SUM(frequency) as frequency,
            ARRAY_AGG(STRUCT_PACK(corpus := corpus, frequency := frequency)) as contributions
        FROM pattern_counts
        GROUP BY n, p, v
        ORDER BY frequency DESC
    """,
            [noun],
        )
        .pl()
        .to_dicts()
    )
    return matches


@app.get("/npv/verb/{verb}")
def read_npv_verb(verb: str) -> List[Dict[str, Any]]:
    matches = (
        db.execute(
            """
        WITH pattern_counts AS (
            SELECT 
                l1.string as n,
                l2.string as p,
                l3.string as v,
                src.corpus,
                COUNT(*) as frequency
            FROM collocation c
            JOIN sentence_word sw1 ON c.word_1_sw_id = sw1.id
            JOIN word w1 ON sw1.word_id = w1.id
            JOIN lemma l1 ON w1.lemma_id = l1.id
            JOIN sentence_word sw2 ON c.particle_sw_id = sw2.id
            JOIN word w2 ON sw2.word_id = w2.id
            JOIN lemma l2 ON w2.lemma_id = l2.id
            JOIN sentence_word sw3 ON c.word_2_sw_id = sw3.id
            JOIN word w3 ON sw3.word_id = w3.id
            JOIN lemma l3 ON w3.lemma_id = l3.id
            JOIN sentence s ON sw1.sentence_id = s.id
            JOIN source src ON s.source_id = src.id
            WHERE l3.string = ?
            GROUP BY l1.string, l2.string, l3.string, src.corpus
        )
        SELECT 
            n, p, v,
            SUM(frequency) as frequency,
            ARRAY_AGG(STRUCT_PACK(corpus := corpus, frequency := frequency)) as contributions
        FROM pattern_counts
        GROUP BY n, p, v
        ORDER BY frequency DESC
    """,
            [verb],
        )
        .pl()
        .to_dicts()
    )
    return matches


@app.get("/sentences/{n}/{p}/{v}")
def read_sentences(n: str, p: str, v: str) -> List[dict[str, str]]:
    conn = load_sentences_db()
    matches = (
        conn.execute(
            """
        WITH colloc AS (
            SELECT sw1.sentence_id
            FROM collocation c
            JOIN sentence_word sw1 ON c.word_1_sw_id = sw1.id
            JOIN word w1 ON sw1.word_id = w1.id
            JOIN lemma l1 ON w1.lemma_id = l1.id
            JOIN sentence_word sw2 ON c.particle_sw_id = sw2.id
            JOIN word w2 ON sw2.word_id = w2.id
            JOIN lemma l2 ON w2.lemma_id = l2.id
            JOIN sentence_word sw3 ON c.word_2_sw_id = sw3.id
            JOIN word w3 ON sw3.word_id = w3.id
            JOIN lemma l3 ON w3.lemma_id = l3.id
            WHERE l1.string = ?
            AND l2.string = ?
            AND l3.string = ?
        )
        SELECT s.text, src.corpus 
        FROM sentence s
        JOIN source src ON s.source_id = src.id
        JOIN colloc ON s.id = colloc.sentence_id
    """,
            [n, p, v],
        )
        .pl()
        .to_dicts()
    )
    return matches


@app.get("/search/{query}")
def read_query(query: str) -> List[tuple[str, str]]:
    matches = db.execute(
        """
        WITH lemma_matches AS (
            SELECT DISTINCT l.string, 'n' as type, COUNT(*) as frequency
            FROM lemma l
            JOIN word w ON l.id = w.lemma_id
            JOIN sentence_word sw ON w.id = sw.word_id
            WHERE l.string LIKE ?
            AND l.pos IN ('NOUN', 'PROPN')
            GROUP BY l.string
            UNION ALL
            SELECT DISTINCT l.string, 'v' as type, COUNT(*) as frequency
            FROM lemma l
            JOIN word w ON l.id = w.lemma_id
            JOIN sentence_word sw ON w.id = sw.word_id
            WHERE l.string LIKE ?
            AND l.pos = 'VERB'
            GROUP BY l.string
        )
        SELECT string, type
        FROM lemma_matches
        ORDER BY frequency DESC
    """,
        ["%" + query + "%", "%" + query + "%"],
    ).fetchall()

    return [(str(m[0]), str(m[1])) for m in matches]


app.mount("/", StaticFiles(directory="natsume-frontend/build", html=True), name="app")
