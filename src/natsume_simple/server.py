# /// script
# dependencies = [
#   "fastapi",
#   "polars",
#   "duckdb",
# ]
# ///

from pathlib import Path
from typing import Any, Dict, List, TypedDict

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


def load_db() -> duckdb.DuckDBPyConnection:
    db_path = Path("data/corpus.db")
    return duckdb.connect(str(db_path), read_only=True)


def calculate_normalized_frequencies(
    frequency: int, corpus: str, corpus_norm: dict
) -> tuple[float, int]:
    """Calculate both normalized and raw frequencies."""
    norm_factor = corpus_norm.get(corpus, 1)
    return (frequency * norm_factor, frequency)


def calculate_corpus_stats(
    conn: duckdb.DuckDBPyConnection,
) -> Dict[str, Dict[str, float]]:
    """Calculate corpus statistics including collocation counts and normalization factors."""
    corpus_freqs = conn.execute("""
        SELECT src.corpus, COUNT(*) as frequency
        FROM collocation c
        JOIN sentence_word sw ON c.word_1_sw_id = sw.id
        JOIN sentence s ON sw.sentence_id = s.id
        JOIN source src ON s.source_id = src.id
        GROUP BY src.corpus
    """).pl()

    min_count = corpus_freqs["frequency"].min()
    stats = {}

    for corpus, frequency in zip(corpus_freqs["corpus"], corpus_freqs["frequency"]):
        stats[corpus] = {
            "collocationCount": int(frequency),
            "normalizationFactor": min_count / frequency,
        }

    return stats


conn = load_db()
corpus_stats = calculate_corpus_stats(conn)


@app.get("/corpus/stats")
def get_corpus_stats() -> Dict[str, Dict[str, float]]:
    return corpus_stats


@app.get("/corpus/norm")
def get_corpus_norm() -> Dict[str, Dict[str, float]]:
    """Return both normalization factors and collocation counts for each corpus."""
    return {
        corpus: {
            "normalizationFactor": stats["normalizationFactor"],
            "collocationCount": stats["collocationCount"],
        }
        for corpus, stats in corpus_stats.items()
    }


# Add type definitions
class Contribution(TypedDict):
    corpus: str
    normalizedFrequency: float
    rawFrequency: int


class Collocate(TypedDict):
    n: str
    p: str
    v: str
    totalNormalizedFrequency: float
    totalRawFrequency: int
    contributions: List[Contribution]


class Distribution(TypedDict):
    normalized: float
    raw: int
    normalizedWidth: float
    normalizedOffset: float
    rawWidth: float
    rawOffset: float


class ParticleGroup(TypedDict):
    collocates: List[Collocate]
    maxFrequency: Dict[str, float]
    distribution: Dict[str, Distribution]


def process_query_results(
    raw_matches: Any, particles: list[str], corpus_norm: Dict[str, float]
) -> Dict[str, ParticleGroup]:
    """Process raw query results into particle groups with normalized and raw frequencies."""
    particle_groups: Dict[str, List[Collocate]] = {p: [] for p in particles}

    for row in raw_matches.to_dicts():
        particle = row["p"]
        if particle in particles:
            # Calculate both normalized and raw frequencies for each contribution
            contributions: List[Contribution] = []
            total_norm_freq: float = 0.0
            total_raw_freq: int = 0

            for contrib in row["contributions"]:
                norm_freq, raw_freq = calculate_normalized_frequencies(
                    contrib["frequency"], contrib["corpus"], corpus_norm
                )
                contributions.append(
                    {
                        "corpus": contrib["corpus"],
                        "normalizedFrequency": float(norm_freq),
                        "rawFrequency": int(raw_freq),
                    }
                )
                total_norm_freq += float(norm_freq)
                total_raw_freq += int(raw_freq)

            particle_groups[particle].append(
                {
                    "n": row["n"],
                    "p": row["p"],
                    "v": row["v"],
                    "totalNormalizedFrequency": total_norm_freq,
                    "totalRawFrequency": total_raw_freq,
                    "contributions": contributions,
                }
            )

    # Process each particle group
    result: Dict[str, ParticleGroup] = {}
    for particle, collocates in particle_groups.items():
        if collocates:
            # Sort by both normalized and raw frequencies
            collocates.sort(
                key=lambda x: (x["totalNormalizedFrequency"], x["totalRawFrequency"]),
                reverse=True,
            )

            # Calculate max frequencies for both normalized and raw values
            max_norm_freq = max(c["totalNormalizedFrequency"] for c in collocates)
            max_raw_freq = max(c["totalRawFrequency"] for c in collocates)

            # Calculate distribution with both normalized and raw values
            distribution: Dict[str, Distribution] = {}
            total_norm: float = 0.0
            total_raw: int = 0

            # First pass: calculate totals
            for collocate in collocates:
                for contrib in collocate["contributions"]:
                    corpus = contrib["corpus"]
                    if corpus not in distribution:
                        distribution[corpus] = {
                            "normalized": 0.0,
                            "raw": 0,
                            "normalizedWidth": 0.0,
                            "normalizedOffset": 0.0,
                            "rawWidth": 0.0,
                            "rawOffset": 0.0,
                        }
                    distribution[corpus]["normalized"] += float(
                        contrib["normalizedFrequency"]
                    )
                    distribution[corpus]["raw"] += int(contrib["rawFrequency"])
                    total_norm += float(contrib["normalizedFrequency"])
                    total_raw += int(contrib["rawFrequency"])

            # Second pass: calculate percentages and positions
            norm_offset: float = 0.0
            raw_offset: float = 0.0
            for corpus, freqs in distribution.items():
                norm_width = (
                    (freqs["normalized"] / total_norm * 100) if total_norm > 0 else 0.0
                )
                raw_width = (freqs["raw"] / total_raw * 100) if total_raw > 0 else 0.0

                distribution[corpus].update(
                    {
                        "normalizedWidth": norm_width,
                        "normalizedOffset": norm_offset,
                        "rawWidth": raw_width,
                        "rawOffset": raw_offset,
                    }
                )

                norm_offset += norm_width
                raw_offset += raw_width

            result[particle] = {
                "collocates": collocates,
                "maxFrequency": {
                    "normalized": max_norm_freq,
                    "raw": float(max_raw_freq),
                },
                "distribution": distribution,
            }

    return result


def get_npv_query(search_type: str, term: str) -> tuple[str, list]:
    """Get the appropriate SQL query based on search type."""
    base_query = """
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
            WHERE {where_clause}
            GROUP BY l1.string, l2.string, l3.string, src.corpus
        )
        SELECT 
            n, p, v,
            CAST(SUM(frequency) AS INTEGER) as total_frequency,
            ARRAY_AGG(STRUCT_PACK(corpus := corpus, frequency := frequency)) as contributions
        FROM pattern_counts
        GROUP BY n, p, v
        ORDER BY total_frequency DESC
    """

    where_clause = "l1.string = ?" if search_type == "noun" else "l3.string = ?"
    return base_query.format(where_clause=where_clause), [term]


@app.get("/npv/{search_type}/{term}")
def read_npv(search_type: str, term: str) -> Dict[str, Any]:
    if search_type not in ["noun", "verb"]:
        raise ValueError("search_type must be either 'noun' or 'verb'")

    query, params = get_npv_query(search_type, term)
    raw_matches = conn.execute(query, params).pl()

    particles = ["が", "を", "に", "で", "から", "より", "と", "へ"]
    particle_groups = process_query_results(
        raw_matches,
        particles,
        {
            corpus: stats["normalizationFactor"]
            for corpus, stats in corpus_stats.items()
        },
    )

    return {
        "particleGroups": particle_groups,
        "corpusNorm": {
            corpus: {
                "normalizationFactor": stats["normalizationFactor"],
                "collocationCount": stats["collocationCount"],
            }
            for corpus, stats in corpus_stats.items()
        },
        "totalResults": len(raw_matches),
    }


@app.get("/sentences/{n}/{p}/{v}/{limit}")
def read_sentences(
    n: str, p: str, v: str, limit: int = 5
) -> List[dict[str, str | int]]:
    matches = (
        conn.execute(
            """
        WITH colloc AS (
            SELECT 
                sw1.sentence_id,
                sw1.begin as n_begin,
                sw1."end" as n_end,
                sw2.begin as p_begin,
                sw2."end" as p_end,
                sw3.begin as v_begin,
                sw3."end" as v_end
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
        SELECT 
            s.text, 
            src.corpus,
            colloc.n_begin,
            colloc.n_end,
            colloc.p_begin,
            colloc.p_end,
            colloc.v_begin,
            colloc.v_end
        FROM sentence s
        JOIN source src ON s.source_id = src.id
        JOIN colloc ON s.id = colloc.sentence_id
        LIMIT ?
    """,
            [n, p, v, limit],
        )
        .pl()
        .to_dicts()
    )
    return matches


@app.get("/search/{query}")
def read_query(query: str) -> List[tuple[str, str]]:
    matches = conn.execute(
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
