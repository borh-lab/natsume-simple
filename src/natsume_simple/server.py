# /// script
# dependencies = [
#   "fastapi",
#   "polars",
#   "duckdb",
# ]
# ///

from contextlib import asynccontextmanager
import hashlib
import json
from pathlib import Path
import re
from typing import Annotated, Any, Dict, List, Literal, TypedDict
import uuid

import duckdb
from fastapi import FastAPI, Query, Request  # type: ignore
from fastapi.middleware.cors import CORSMiddleware  # type: ignore
from fastapi.exceptions import RequestValidationError  # type: ignore
from fastapi.responses import JSONResponse  # type: ignore
from fastapi.staticfiles import StaticFiles  # type: ignore
from pydantic import BaseModel

DATABASE_PATH = Path("data/corpus.db")


def load_db(db_path: Path) -> duckdb.DuckDBPyConnection:
    return duckdb.connect(str(db_path), read_only=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    conn = load_db(DATABASE_PATH)
    app.state.conn = conn
    app.state.corpus_stats = calculate_corpus_stats(conn)
    try:
        yield
    finally:
        conn.close()


app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change this to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health/live")
def health_live() -> dict[str, str]:
    return {"status": "ok"}


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


@app.get("/corpus/norm")
def get_corpus_norm(request: Request) -> Dict[str, Dict[str, float]]:
    """Return both normalization factors and collocation counts for each corpus."""
    return {
        corpus: {
            "normalizationFactor": stats["normalizationFactor"],
            "collocationCount": stats["collocationCount"],
        }
        for corpus, stats in request.app.state.corpus_stats.items()
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
    total_normalized: float
    total_raw: int


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
                            "total_normalized": 0.0,
                            "total_raw": 0,
                        }
                    distribution[corpus]["normalized"] += float(
                        contrib["normalizedFrequency"]
                    )
                    distribution[corpus]["raw"] += int(contrib["rawFrequency"])
                    distribution[corpus]["total_normalized"] = total_norm
                    distribution[corpus]["total_raw"] = total_raw
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
def read_npv(request: Request, search_type: str, term: str) -> Dict[str, Any]:
    if search_type not in ["noun", "verb"]:
        raise ValueError("search_type must be either 'noun' or 'verb'")

    query, params = get_npv_query(search_type, term)
    raw_matches = request.app.state.conn.execute(query, params).pl()
    corpus_stats = request.app.state.corpus_stats

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
    request: Request, n: str, p: str, v: str, limit: int = 5
) -> List[dict[str, str | int]]:
    matches = (
        request.app.state.conn.execute(
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
def read_query(request: Request, query: str) -> List[tuple[str, str]]:
    matches = request.app.state.conn.execute(
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


app.mount(
    "/",
    StaticFiles(directory="natsume-frontend/build", html=True, check_dir=False),
    name="app",
)


class HealthResponse(BaseModel):
    status: str


class ReadyResponse(HealthResponse):
    databaseBuildId: str
    schemaVersion: int


class CorpusResponse(BaseModel):
    id: str
    label: str
    collocationCount: int
    sentenceCount: int


class CorporaResponse(BaseModel):
    corpora: list[CorpusResponse]
    databaseBuildId: str


class SuggestionResponse(BaseModel):
    lemma: str
    pos: Literal["noun", "verb"]
    occurrenceCount: int


class SuggestionsResponse(BaseModel):
    suggestions: list[SuggestionResponse]


class CorpusContributionResponse(BaseModel):
    corpusId: str
    rawFrequency: int
    frequencyPerMillion: float


class CollocationItemResponse(BaseModel):
    noun: str
    particle: str
    verb: str
    totalRawFrequency: int
    meanFrequencyPerMillion: float
    contributions: list[CorpusContributionResponse]


class ParticleGroupResponse(BaseModel):
    particle: str
    totalMatchingCollocations: int
    returnedCount: int
    items: list[CollocationItemResponse]
    corpusDistribution: list[CorpusContributionResponse]


class CollocationsResponse(BaseModel):
    particleGroups: list[ParticleGroupResponse]
    selectedCorpusIds: list[str]
    rankBy: Literal["raw", "meanPerMillion"]
    databaseBuildId: str


class TextSpanResponse(BaseModel):
    start: int
    end: int


class ExampleResponse(BaseModel):
    corpusId: str
    sourceId: int
    sourceTitle: str
    sentenceId: int
    text: str
    nounSpan: TextSpanResponse
    particleSpan: TextSpanResponse
    verbSpan: TextSpanResponse


class ExamplesResponse(BaseModel):
    examples: list[ExampleResponse]
    selectedCorpusIds: list[str]
    databaseBuildId: str


def create_app(artifact_dir: Path) -> FastAPI:
    @asynccontextmanager
    async def artifact_lifespan(api: FastAPI):
        manifest = json.loads((artifact_dir / "manifest.json").read_text())
        database_path = artifact_dir / "corpus.duckdb"
        checksum = hashlib.sha256(database_path.read_bytes()).hexdigest()
        if manifest["schemaVersion"] != 1 or checksum != manifest["databaseSha256"]:
            raise RuntimeError("incompatible database artifact")

        connection = duckdb.connect(str(database_path), read_only=True)
        metadata = connection.execute(
            "SELECT schema_version, artifact_instance_id FROM build_metadata"
        ).fetchone()
        if metadata != (manifest["schemaVersion"], manifest["artifactInstanceId"]):
            connection.close()
            raise RuntimeError("database identity does not match manifest")

        api.state.database_path = database_path
        api.state.database_build_id = manifest["artifactInstanceId"]
        api.state.schema_version = manifest["schemaVersion"]
        connection.close()
        yield

    api = FastAPI(lifespan=artifact_lifespan)

    @api.middleware("http")
    async def request_identity(request: Request, call_next):
        supplied = request.headers.get("X-Request-ID", "")
        request.state.request_id = (
            supplied
            if len(supplied) <= 128 and re.fullmatch(r"[\x21-\x7e]+", supplied)
            else uuid.uuid4().hex
        )
        return await call_next(request)

    @api.exception_handler(RequestValidationError)
    async def validation_error(request: Request, _error: RequestValidationError):
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "invalid_parameter",
                    "message": "Request parameters are invalid",
                    "requestId": request.state.request_id,
                }
            },
        )

    @api.get("/api/health/live", response_model=HealthResponse)
    def live() -> HealthResponse:
        return HealthResponse(status="ok")

    @api.get("/api/health/ready", response_model=ReadyResponse)
    def ready(request: Request) -> ReadyResponse:
        with duckdb.connect(
            str(request.app.state.database_path), read_only=True
        ) as conn:
            conn.execute("SELECT 1").fetchone()
        return ReadyResponse(
            status="ok",
            databaseBuildId=request.app.state.database_build_id,
            schemaVersion=request.app.state.schema_version,
        )

    @api.get("/api/corpora", response_model=CorporaResponse)
    def corpora(request: Request) -> CorporaResponse:
        with duckdb.connect(
            str(request.app.state.database_path), read_only=True
        ) as conn:
            rows = conn.execute(
                """
                SELECT c.id, c.label, cs.collocation_count, cs.sentence_count
                FROM corpus c
                JOIN corpus_stats cs ON cs.corpus_id = c.id
                ORDER BY c.id
                """
            ).fetchall()
        return CorporaResponse(
            corpora=[
                CorpusResponse(
                    id=row[0],
                    label=row[1],
                    collocationCount=row[2],
                    sentenceCount=row[3],
                )
                for row in rows
            ],
            databaseBuildId=request.app.state.database_build_id,
        )

    @api.get("/api/suggestions", response_model=SuggestionsResponse)
    def suggestions(
        request: Request,
        q: Annotated[str, Query(min_length=1, max_length=64)],
        pos: Literal["noun", "verb"],
        limit: Annotated[int, Query(ge=1, le=20)] = 10,
    ) -> SuggestionsResponse:
        with duckdb.connect(
            str(request.app.state.database_path), read_only=True
        ) as conn:
            rows = conn.execute(
                """
                SELECT lemma, part_of_speech, occurrence_count
                FROM lemma_frequency
                WHERE part_of_speech = ? AND contains(lemma, ?)
                ORDER BY occurrence_count DESC, lemma ASC
                LIMIT ?
                """,
                [pos, q, limit],
            ).fetchall()
        return SuggestionsResponse(
            suggestions=[
                SuggestionResponse(lemma=row[0], pos=row[1], occurrenceCount=row[2])
                for row in rows
            ]
        )

    @api.get("/api/collocations", response_model=CollocationsResponse)
    def collocations(
        request: Request,
        term: Annotated[str, Query(min_length=1, max_length=64)],
        pos: Literal["noun", "verb"],
        rankBy: Literal["raw", "meanPerMillion"],
        corpusId: Annotated[list[str] | None, Query()] = None,
        limitPerParticle: Annotated[int, Query(ge=1, le=200)] = 100,
    ) -> CollocationsResponse:
        with duckdb.connect(
            str(request.app.state.database_path), read_only=True
        ) as conn:
            corpus_rows = conn.execute(
                "SELECT corpus_id, collocation_count FROM corpus_stats ORDER BY corpus_id"
            ).fetchall()
            corpus_counts = {row[0]: row[1] for row in corpus_rows}
            selected = sorted(set(corpusId or corpus_counts))

            term_column = "noun" if pos == "noun" else "verb"
            placeholders = ", ".join("?" for _ in selected)
            rows = conn.execute(
                f"""
                SELECT corpus_id, noun, particle, verb, raw_frequency
                FROM collocation_frequency
                WHERE {term_column} = ? AND corpus_id IN ({placeholders})
                """,
                [term, *selected],
            ).fetchall()

        by_triple: dict[tuple[str, str, str], dict[str, int]] = {}
        for corpus_id, noun, particle, verb, raw_frequency in rows:
            by_triple.setdefault((noun, particle, verb), {})[corpus_id] = raw_frequency

        by_particle: dict[str, list[CollocationItemResponse]] = {}
        for (noun, particle, verb), raw_by_corpus in by_triple.items():
            contributions = [
                CorpusContributionResponse(
                    corpusId=corpus_id,
                    rawFrequency=raw_by_corpus[corpus_id],
                    frequencyPerMillion=(
                        raw_by_corpus[corpus_id] / corpus_counts[corpus_id] * 1_000_000
                    ),
                )
                for corpus_id in selected
                if corpus_id in raw_by_corpus
            ]
            item = CollocationItemResponse(
                noun=noun,
                particle=particle,
                verb=verb,
                totalRawFrequency=sum(raw_by_corpus.values()),
                meanFrequencyPerMillion=(
                    sum(item.frequencyPerMillion for item in contributions)
                    / len(selected)
                ),
                contributions=contributions,
            )
            by_particle.setdefault(particle, []).append(item)

        particle_groups = []
        for particle in ["が", "を", "に", "で", "から", "より", "と", "へ"]:
            items = by_particle.get(particle, [])
            if not items:
                continue
            if rankBy == "raw":
                items.sort(
                    key=lambda item: (
                        -item.totalRawFrequency,
                        -item.meanFrequencyPerMillion,
                        item.noun,
                        item.particle,
                        item.verb,
                    )
                )
            else:
                items.sort(
                    key=lambda item: (
                        -item.meanFrequencyPerMillion,
                        -item.totalRawFrequency,
                        item.noun,
                        item.particle,
                        item.verb,
                    )
                )
            distribution = [
                CorpusContributionResponse(
                    corpusId=corpus_id,
                    rawFrequency=sum(
                        contribution.rawFrequency
                        for item in items
                        for contribution in item.contributions
                        if contribution.corpusId == corpus_id
                    ),
                    frequencyPerMillion=sum(
                        contribution.frequencyPerMillion
                        for item in items
                        for contribution in item.contributions
                        if contribution.corpusId == corpus_id
                    ),
                )
                for corpus_id in selected
            ]
            returned_items = items[:limitPerParticle]
            particle_groups.append(
                ParticleGroupResponse(
                    particle=particle,
                    totalMatchingCollocations=len(items),
                    returnedCount=len(returned_items),
                    items=returned_items,
                    corpusDistribution=distribution,
                )
            )

        return CollocationsResponse(
            particleGroups=particle_groups,
            selectedCorpusIds=selected,
            rankBy=rankBy,
            databaseBuildId=request.app.state.database_build_id,
        )

    @api.get("/api/examples", response_model=ExamplesResponse)
    def examples(
        request: Request,
        noun: Annotated[str, Query(min_length=1, max_length=64)],
        particle: Annotated[str, Query(min_length=1, max_length=64)],
        verb: Annotated[str, Query(min_length=1, max_length=64)],
        corpusId: Annotated[list[str] | None, Query()] = None,
        limit: Annotated[int, Query(ge=1, le=20)] = 5,
    ) -> ExamplesResponse:
        with duckdb.connect(
            str(request.app.state.database_path), read_only=True
        ) as conn:
            all_corpora = [
                row[0]
                for row in conn.execute("SELECT id FROM corpus ORDER BY id").fetchall()
            ]
            selected = sorted(set(corpusId or all_corpora))
            placeholders = ", ".join("?" for _ in selected)
            rows = conn.execute(
                f"""
                SELECT src.corpus_id, src.id, src.title, s.id, s.text,
                       o.n_begin, o.n_end, o.p_begin, o.p_end, o.v_begin, o.v_end
                FROM collocation_occurrence o
                JOIN sentence s ON s.id = o.sentence_id
                JOIN source src ON src.id = s.source_id
                WHERE o.noun = ? AND o.particle = ? AND o.verb = ?
                  AND src.corpus_id IN ({placeholders})
                ORDER BY src.corpus_id, src.id, s.id
                LIMIT ?
                """,
                [noun, particle, verb, *selected, limit],
            ).fetchall()
        return ExamplesResponse(
            examples=[
                ExampleResponse(
                    corpusId=row[0],
                    sourceId=row[1],
                    sourceTitle=row[2],
                    sentenceId=row[3],
                    text=row[4],
                    nounSpan=TextSpanResponse(start=row[5], end=row[6]),
                    particleSpan=TextSpanResponse(start=row[7], end=row[8]),
                    verbSpan=TextSpanResponse(start=row[9], end=row[10]),
                )
                for row in rows
            ],
            selectedCorpusIds=selected,
            databaseBuildId=request.app.state.database_build_id,
        )

    return api
