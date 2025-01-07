# /// script
# dependencies = [
#   "fastapi",
#   "polars",
# ]
# ///

from collections import Counter
from typing import Any, Dict, List

import polars as pl  # type: ignore
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


def load_database(model_name: str) -> pl.DataFrame:
    db = pl.read_csv(f"data/ted_npvs_{model_name}.csv")
    db = db.vstack(pl.read_csv(f"data/jnlp_npvs_{model_name}.csv"))
    db = db.vstack(pl.read_csv(f"data/wiki_npvs_{model_name}.csv"))
    return db.group_by(db.columns).agg(pl.len().alias("frequency"))


def load_sentences() -> list[str]:
    with open("data/ted_corpus.txt", "r") as f:
        return f.readlines()


def calculate_corpus_norm(db: pl.DataFrame) -> Dict[str, float]:
    """Calculate normalization factors for different corpora.

    Args:
        db: DataFrame containing corpus frequencies

    Returns:
        Dictionary mapping corpus names to their normalization factors

    Examples:
        >>> import polars as pl
        >>> df = pl.DataFrame({
        ...     "corpus": ["ted", "ted", "jnlp", "jnlp", "jnlp"],
        ...     "frequency": [1, 2, 3, 2, 1]
        ... })
        >>> norms = calculate_corpus_norm(df)
        >>> norms["ted"] == 1.0  # ted has lower total frequency
        True
        >>> norms["jnlp"] == 0.5  # jnlp has double the frequency
        True
    """
    corpus_freqs = {
        corpus: db.filter(pl.col("corpus") == corpus)["frequency"].sum()
        for corpus in db["corpus"].unique()
    }
    min_count = min(corpus_freqs.values())
    return {corpus: min_count / frequency for corpus, frequency in corpus_freqs.items()}


model_name = "ja_ginza"
db = load_database(model_name)
corpus_norm = calculate_corpus_norm(db)
sentences = load_sentences()


@app.get("/corpus/norm")
def get_corpus_norm() -> Dict[str, float]:
    return corpus_norm


@app.get("/npv/noun/{noun}")
def read_npv_noun(noun: str) -> List[Dict[str, Any]]:
    matches = db.filter(pl.col("n") == noun).drop("n").to_dicts()
    return matches


@app.get("/npv/verb/{verb}")
def read_npv_verb(verb: str) -> List[Dict[str, Any]]:
    matches = db.filter(pl.col("v") == verb).drop("v").to_dicts()
    return matches


@app.get("/sentences/{n}/{p}/{v}")
def read_sentences(n: str, p: str, v: str) -> List[str]:
    matches = [s for s in sentences if n + p in s]
    print(f"matches: {matches}")
    return matches


# @app.get("/search/{query}")
# def read_query(query:str) -> List[tuple[str, str]]:
#     matches = db.select(
#         pl.col('n').str.starts_with(query),
#         pl.col('frequency')
#     ).sort('frequency', descending=True).itertuples()

#     return matches

#     # str.startswith
#     # [("aa", "v"), ("ab", "n"), (...)]


@app.get("/search/{query}")
def read_query(query: str) -> List[tuple[str, str]]:
    # Filter rows containing the query
    matches = (
        db.filter(pl.col("n").str.contains(query) | pl.col("v").str.contains(query))
        .select(["n", "v"])
        .to_dicts()
    )

    result = []

    # Iterate through the filtered results
    for row in matches:
        # If the query exists in the `n` column, add it to the result list
        if query in row["n"]:
            result.append((row["n"], "n"))
        # If the query exists in the `v` column, add it to the result list
        if query in row["v"]:
            result.append((row["v"], "v"))

    tuple_counts = Counter(result)

    sorted_unique_result = sorted(
        tuple_counts.keys(), key=lambda x: tuple_counts[x], reverse=True
    )

    return sorted_unique_result


app.mount("/", StaticFiles(directory="natsume-frontend/build", html=True), name="app")
