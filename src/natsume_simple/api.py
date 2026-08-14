import asyncio
import json
import logging
import os
import re
import time
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal

import anyio
import duckdb
from anyio import CapacityLimiter, WouldBlock
from fastapi import Depends, FastAPI, Query, Request  # type: ignore
from fastapi.exceptions import RequestValidationError  # type: ignore
from fastapi.responses import JSONResponse  # type: ignore
from fastapi.staticfiles import StaticFiles  # type: ignore
from pydantic import BaseModel

from natsume_simple.artifact_validation import (
    ArtifactValidationError,
    validate_artifact,
)

QUERY_CAPACITY = 16
QUERY_TIMEOUT_SECONDS = 2.0
MAX_COLLOCATIONS_PER_PARTICLE = 200
COLLOCATION_ITEM_CORPUS_BUDGET = 450
MAX_DATABASE_OFFSET = (1 << 63) - 1
Particle = Literal["が", "を", "に", "で", "から", "より", "と", "へ"]
PARTICLES: tuple[Particle, ...] = ("が", "を", "に", "で", "から", "より", "と", "へ")
request_logger = logging.getLogger("natsume_simple.api.requests")
startup_logger = logging.getLogger("natsume_simple.api.startup")


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


class CorpusDistributionResponse(BaseModel):
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
    corpusDistribution: list[CorpusDistributionResponse]


class CollocationsResponse(BaseModel):
    particleGroups: list[ParticleGroupResponse]
    selectedCorpusIds: list[str]
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
    hasMore: bool
    selectedCorpusIds: list[str]
    databaseBuildId: str


class PublicApiError(Exception):
    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        headers: dict[str, str] | None = None,
    ):
        self.status_code = status_code
        self.code = code
        self.message = message
        self.headers = headers or {}


def query_length_bucket(query: str) -> str:
    length = len(query)
    if length == 1:
        return "1"
    if length <= 4:
        return "2–4"
    if length <= 8:
        return "5–8"
    if length <= 16:
        return "9–16"
    if length <= 32:
        return "17–32"
    return "33–64"


def database_unavailable() -> PublicApiError:
    return PublicApiError(
        503,
        "database_unavailable",
        "The database artifact is unavailable",
    )


def log_artifact_rejection(reason: str) -> None:
    startup_logger.error(
        json.dumps(
            {
                "timestamp": datetime.now(UTC).isoformat(),
                "level": "ERROR",
                "event": "artifact_rejected",
                "reason": reason,
            },
            separators=(",", ":"),
        )
    )


async def run_database_operation[Result](operation: Callable[[], Result]) -> Result:
    try:
        return await anyio.to_thread.run_sync(operation)
    except duckdb.OperationalError as error:
        raise database_unavailable() from error


async def run_bounded_query[Result](
    connection,
    limiter: CapacityLimiter,
    operation: Callable[[], Result],
    *,
    timeout: float,
) -> Result:
    try:
        limiter.acquire_nowait()
    except WouldBlock as error:
        raise PublicApiError(
            429,
            "capacity_exceeded",
            "Query capacity is exhausted",
            {"Retry-After": "1"},
        ) from error

    timed_out = False

    def interrupt():
        nonlocal timed_out
        timed_out = True
        connection.interrupt()

    timer = asyncio.get_running_loop().call_later(timeout, interrupt)
    try:
        return await run_database_operation(operation)
    except duckdb.InterruptException as error:
        if timed_out:
            raise PublicApiError(
                504, "query_timeout", "The database query timed out"
            ) from error
        raise
    finally:
        timer.cancel()
        limiter.release()


def select_corpora(requested: list[str] | None, known: set[str]) -> list[str]:
    if requested is None:
        return sorted(known)

    selected = sorted(set(requested))
    if any(not corpus_id for corpus_id in selected) or not set(selected) <= known:
        raise PublicApiError(
            400, "invalid_parameter", "corpusId must name a known corpus"
        )
    return selected


async def database_connection(
    request: Request,
) -> AsyncIterator[duckdb.DuckDBPyConnection]:
    database_path = request.app.state.database_path
    if database_path is None:
        raise database_unavailable()

    try:
        connection = await anyio.to_thread.run_sync(
            lambda: duckdb.connect(str(database_path), read_only=True)
        )
    except (duckdb.OperationalError, OSError) as error:
        raise database_unavailable() from error
    try:
        yield connection
    finally:
        await anyio.to_thread.run_sync(connection.close)


DatabaseConnection = Annotated[duckdb.DuckDBPyConnection, Depends(database_connection)]


def create_app(artifact_dir: Path, *, frontend_dir: Path | None = None) -> FastAPI:
    @asynccontextmanager
    async def artifact_lifespan(api: FastAPI):
        api.state.database_path = None
        api.state.database_build_id = None
        api.state.schema_version = None
        api.state.query_limiter = CapacityLimiter(QUERY_CAPACITY)

        try:
            selected_artifact = artifact_dir.resolve(strict=True)
            database_path, manifest = validate_artifact(selected_artifact)
        except OSError:
            log_artifact_rejection("artifact_path_unresolvable")
        except ArtifactValidationError as error:
            log_artifact_rejection(error.reason)
        else:
            api.state.database_path = database_path
            api.state.database_build_id = manifest["artifactInstanceId"]
            api.state.schema_version = manifest["schemaVersion"]
        yield

    api = FastAPI(lifespan=artifact_lifespan)

    @api.middleware("http")
    async def request_identity(request: Request, call_next):
        started = time.perf_counter()
        supplied = request.headers.get("X-Request-ID", "")
        request.state.request_id = (
            supplied
            if len(supplied) <= 128 and re.fullmatch(r"[\x21-\x7e]+", supplied)
            else uuid.uuid4().hex
        )
        request.state.result_count = None
        request.state.corpus_ids = []
        request.state.query_length_bucket = None
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            route = request.scope.get("route")
            request_logger.info(
                json.dumps(
                    {
                        "timestamp": datetime.now(UTC).isoformat(),
                        "level": "INFO",
                        "requestId": request.state.request_id,
                        "route": getattr(route, "path", request.url.path),
                        "status": status,
                        "durationMs": round((time.perf_counter() - started) * 1_000, 3),
                        "databaseBuildId": request.app.state.database_build_id,
                        "resultCount": request.state.result_count,
                        "corpusIds": request.state.corpus_ids,
                        "queryLengthBucket": request.state.query_length_bucket,
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )

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

    @api.exception_handler(PublicApiError)
    async def public_api_error(request: Request, error: PublicApiError):
        return JSONResponse(
            status_code=error.status_code,
            headers=error.headers,
            content={
                "error": {
                    "code": error.code,
                    "message": error.message,
                    "requestId": request.state.request_id,
                }
            },
        )

    @api.exception_handler(Exception)
    async def unexpected_error(request: Request, _error: Exception):
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "internal_error",
                    "message": "An unexpected error occurred",
                    "requestId": request.state.request_id,
                }
            },
        )

    @api.get("/api/health/live", response_model=HealthResponse)
    def live() -> HealthResponse:
        return HealthResponse(status="ok")

    @api.get("/api/health/ready", response_model=ReadyResponse)
    async def ready(request: Request, connection: DatabaseConnection) -> ReadyResponse:
        await run_database_operation(lambda: connection.execute("SELECT 1").fetchone())
        return ReadyResponse(
            status="ok",
            databaseBuildId=request.app.state.database_build_id,
            schemaVersion=request.app.state.schema_version,
        )

    @api.get("/api/corpora", response_model=CorporaResponse)
    async def corpora(
        request: Request, connection: DatabaseConnection
    ) -> CorporaResponse:
        rows = await run_database_operation(
            lambda: connection.execute(
                """
                SELECT c.id, c.label, cs.collocation_count, cs.sentence_count
                FROM corpus c
                JOIN corpus_stats cs ON cs.corpus_id = c.id
                ORDER BY c.id
                """
            ).fetchall()
        )
        request.state.result_count = len(rows)
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
    async def suggestions(
        request: Request,
        connection: DatabaseConnection,
        q: Annotated[str, Query(min_length=1, max_length=64)],
        pos: Literal["noun", "verb"],
        limit: Annotated[int, Query(ge=1, le=20)] = 10,
    ) -> SuggestionsResponse:
        request.state.query_length_bucket = query_length_bucket(q)
        rows = await run_bounded_query(
            connection,
            request.app.state.query_limiter,
            lambda: connection.execute(
                """
                SELECT lemma, part_of_speech, occurrence_count
                FROM lemma_frequency
                WHERE part_of_speech = ? AND contains(lemma, ?)
                ORDER BY occurrence_count DESC, lemma ASC
                LIMIT ?
                """,
                [pos, q, limit],
            ).fetchall(),
            timeout=QUERY_TIMEOUT_SECONDS,
        )
        request.state.result_count = len(rows)
        return SuggestionsResponse(
            suggestions=[
                SuggestionResponse(lemma=row[0], pos=row[1], occurrenceCount=row[2])
                for row in rows
            ]
        )

    @api.get("/api/collocations", response_model=CollocationsResponse)
    async def collocations(
        request: Request,
        connection: DatabaseConnection,
        term: Annotated[str, Query(min_length=1, max_length=64)],
        pos: Literal["noun", "verb"],
        corpusId: Annotated[list[str] | None, Query()] = None,
        particle: Annotated[Particle | None, Query()] = None,
        offsetPerParticle: Annotated[int, Query(ge=0)] = 0,
        limitPerParticle: Annotated[
            int, Query(ge=1, le=MAX_COLLOCATIONS_PER_PARTICLE)
        ] = 100,
    ) -> CollocationsResponse:
        request.state.query_length_bucket = query_length_bucket(term)
        if offsetPerParticle and particle is None:
            raise PublicApiError(
                400,
                "invalid_parameter",
                "offsetPerParticle requires particle",
            )

        def load_rows():
            corpus_rows = connection.execute(
                "SELECT corpus_id, collocation_count FROM corpus_stats ORDER BY corpus_id"
            ).fetchall()
            corpus_counts = {row[0]: row[1] for row in corpus_rows}
            selected = select_corpora(corpusId, set(corpus_counts))
            if limitPerParticle * len(selected) > COLLOCATION_ITEM_CORPUS_BUDGET:
                raise PublicApiError(
                    400,
                    "invalid_parameter",
                    "limitPerParticle is too large for the selected corpus count",
                )

            term_column = "noun" if pos == "noun" else "verb"
            placeholders = ", ".join("?" for _ in selected)
            particle_clause = " AND particle = ?" if particle is not None else ""
            parameters = [term, *selected]
            if particle is not None:
                parameters.append(particle)
            rows = connection.execute(
                f"""
                SELECT corpus_id, noun, particle, verb, raw_frequency
                FROM collocation_frequency
                WHERE {term_column} = ? AND corpus_id IN ({placeholders})
                  {particle_clause}
                """,
                parameters,
            ).fetchall()
            return corpus_counts, selected, rows

        corpus_counts, selected, rows = await run_bounded_query(
            connection,
            request.app.state.query_limiter,
            load_rows,
            timeout=QUERY_TIMEOUT_SECONDS,
        )

        by_triple: dict[tuple[str, str, str], dict[str, int]] = {}
        for corpus_id, noun, row_particle, verb, raw_frequency in rows:
            by_triple.setdefault((noun, row_particle, verb), {})[corpus_id] = (
                raw_frequency
            )

        by_particle: dict[str, list[CollocationItemResponse]] = {}
        for (noun, item_particle, verb), raw_by_corpus in by_triple.items():
            rates_by_corpus = {
                corpus_id: raw_by_corpus[corpus_id]
                / corpus_counts[corpus_id]
                * 1_000_000
                for corpus_id in selected
                if corpus_id in raw_by_corpus
            }
            contributions = [
                CorpusContributionResponse(
                    corpusId=corpus_id,
                    rawFrequency=raw_by_corpus[corpus_id],
                )
                for corpus_id in selected
                if corpus_id in raw_by_corpus
            ]
            item = CollocationItemResponse(
                noun=noun,
                particle=item_particle,
                verb=verb,
                totalRawFrequency=sum(raw_by_corpus.values()),
                meanFrequencyPerMillion=(sum(rates_by_corpus.values()) / len(selected)),
                contributions=contributions,
            )
            by_particle.setdefault(item_particle, []).append(item)

        particle_groups = []
        target_particles = PARTICLES if particle is None else (particle,)
        for current_particle in target_particles:
            items = by_particle.get(current_particle, [])
            if not items:
                continue
            items.sort(
                key=lambda item: (
                    -item.meanFrequencyPerMillion,
                    -item.totalRawFrequency,
                    item.noun,
                    item.particle,
                    item.verb,
                )
            )
            distribution = []
            for corpus_id in selected:
                raw_frequency = sum(
                    contribution.rawFrequency
                    for item in items
                    for contribution in item.contributions
                    if contribution.corpusId == corpus_id
                )
                distribution.append(
                    CorpusDistributionResponse(
                        corpusId=corpus_id,
                        rawFrequency=raw_frequency,
                        frequencyPerMillion=(
                            raw_frequency / corpus_counts[corpus_id] * 1_000_000
                        ),
                    )
                )
            returned_items = items[
                offsetPerParticle : offsetPerParticle + limitPerParticle
            ]
            particle_groups.append(
                ParticleGroupResponse(
                    particle=current_particle,
                    totalMatchingCollocations=len(items),
                    returnedCount=len(returned_items),
                    items=returned_items,
                    corpusDistribution=distribution,
                )
            )

        request.state.result_count = sum(
            group.returnedCount for group in particle_groups
        )
        request.state.corpus_ids = selected
        return CollocationsResponse(
            particleGroups=particle_groups,
            selectedCorpusIds=selected,
            databaseBuildId=request.app.state.database_build_id,
        )

    @api.get("/api/examples", response_model=ExamplesResponse)
    async def examples(
        request: Request,
        connection: DatabaseConnection,
        noun: Annotated[str, Query(min_length=1, max_length=64)],
        particle: Annotated[str, Query(min_length=1, max_length=64)],
        verb: Annotated[str, Query(min_length=1, max_length=64)],
        corpusId: Annotated[list[str] | None, Query()] = None,
        limit: Annotated[int, Query(ge=1, le=20)] = 5,
        offset: Annotated[int, Query(ge=0, le=MAX_DATABASE_OFFSET)] = 0,
    ) -> ExamplesResponse:
        def load_rows():
            all_corpora = [
                row[0]
                for row in connection.execute(
                    "SELECT id FROM corpus ORDER BY id"
                ).fetchall()
            ]
            selected = select_corpora(corpusId, set(all_corpora))
            placeholders = ", ".join("?" for _ in selected)
            rows = connection.execute(
                f"""
                SELECT src.corpus_id, src.id, src.title, s.id, s.text,
                       o.n_begin, o.n_end, o.p_begin, o.p_end, o.v_begin, o.v_end
                FROM collocation_occurrence o
                JOIN sentence s ON s.id = o.sentence_id
                JOIN source src ON src.id = s.source_id
                WHERE o.noun = ? AND o.particle = ? AND o.verb = ?
                  AND src.corpus_id IN ({placeholders})
                ORDER BY src.corpus_id, src.id, s.id,
                         o.n_begin, o.n_end,
                         o.p_begin, o.p_end,
                         o.v_begin, o.v_end,
                         o.extractor_id
                LIMIT ? OFFSET ?
                """,
                [noun, particle, verb, *selected, limit + 1, offset],
            ).fetchall()
            return selected, rows

        selected, rows = await run_bounded_query(
            connection,
            request.app.state.query_limiter,
            load_rows,
            timeout=QUERY_TIMEOUT_SECONDS,
        )
        has_more = len(rows) > limit
        rows = rows[:limit]
        request.state.result_count = len(rows)
        request.state.corpus_ids = selected
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
            hasMore=has_more,
            selectedCorpusIds=selected,
            databaseBuildId=request.app.state.database_build_id,
        )

    if frontend_dir is not None:
        api.mount(
            "/",
            StaticFiles(directory=frontend_dir, html=True, check_dir=False),
            name="frontend",
        )
    return api


app = create_app(
    Path(os.environ.get("NATSUME_ARTIFACT_DIR", "deploy/current")),
    frontend_dir=Path(os.environ.get("NATSUME_FRONTEND_DIR", "natsume-frontend/build")),
)
