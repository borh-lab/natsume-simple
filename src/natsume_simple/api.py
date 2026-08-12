import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, UTC
import hashlib
import json
import logging
from pathlib import Path
import re
import time
from typing import Annotated, AsyncIterator, Callable, Literal, TypeVar
import uuid

import anyio
import duckdb
from anyio import CapacityLimiter, WouldBlock
from fastapi import Depends, FastAPI, Query, Request  # type: ignore
from fastapi.exceptions import RequestValidationError  # type: ignore
from fastapi.responses import JSONResponse  # type: ignore
from pydantic import BaseModel

QUERY_CAPACITY = 16
QUERY_TIMEOUT_SECONDS = 2.0
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


class ArtifactValidationError(Exception):
    def __init__(self, reason: str):
        self.reason = reason


QueryResult = TypeVar("QueryResult")


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


def validate_artifact(artifact_dir: Path) -> tuple[Path, dict]:
    manifest_path = artifact_dir / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text())
    except FileNotFoundError as error:
        raise ArtifactValidationError("manifest_missing") from error
    except (OSError, json.JSONDecodeError, TypeError) as error:
        raise ArtifactValidationError("manifest_invalid") from error

    try:
        if manifest["schemaVersion"] != 1:
            raise ArtifactValidationError("unsupported_schema")
        expected_checksum = manifest["databaseSha256"]
        artifact_instance_id = manifest["artifactInstanceId"]
    except (KeyError, TypeError) as error:
        raise ArtifactValidationError("manifest_invalid") from error

    database_path = artifact_dir / "corpus.duckdb"
    try:
        checksum = hashlib.sha256(database_path.read_bytes()).hexdigest()
    except OSError as error:
        raise ArtifactValidationError("database_unreadable") from error
    if checksum != expected_checksum:
        raise ArtifactValidationError("checksum_mismatch")

    try:
        with duckdb.connect(str(database_path), read_only=True) as connection:
            metadata = connection.execute(
                "SELECT schema_version, artifact_instance_id FROM build_metadata"
            ).fetchone()
            corpus_count = connection.execute("SELECT count(*) FROM corpus").fetchone()[
                0
            ]
    except duckdb.Error as error:
        raise ArtifactValidationError("database_unreadable") from error

    if metadata != (1, artifact_instance_id):
        raise ArtifactValidationError("identity_mismatch")
    if not 1 <= corpus_count <= 3:
        raise ArtifactValidationError("invalid_corpus_count")
    return database_path, manifest


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


async def run_database_operation(
    operation: Callable[[], QueryResult],
) -> QueryResult:
    try:
        return await anyio.to_thread.run_sync(operation)
    except duckdb.OperationalError as error:
        raise database_unavailable() from error


async def run_bounded_query(
    connection,
    limiter: CapacityLimiter,
    operation: Callable[[], QueryResult],
    *,
    timeout: float,
) -> QueryResult:
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


def create_app(artifact_dir: Path) -> FastAPI:
    @asynccontextmanager
    async def artifact_lifespan(api: FastAPI):
        api.state.database_path = None
        api.state.database_build_id = None
        api.state.schema_version = None
        api.state.query_limiter = CapacityLimiter(QUERY_CAPACITY)

        try:
            database_path, manifest = validate_artifact(artifact_dir)
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
        request.state.rank_by = None
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
                        "rankBy": request.state.rank_by,
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
        rankBy: Literal["raw", "meanPerMillion"],
        corpusId: Annotated[list[str] | None, Query()] = None,
        limitPerParticle: Annotated[int, Query(ge=1, le=150)] = 100,
    ) -> CollocationsResponse:
        request.state.rank_by = rankBy
        request.state.query_length_bucket = query_length_bucket(term)

        def load_rows():
            corpus_rows = connection.execute(
                "SELECT corpus_id, collocation_count FROM corpus_stats ORDER BY corpus_id"
            ).fetchall()
            corpus_counts = {row[0]: row[1] for row in corpus_rows}
            selected = select_corpora(corpusId, set(corpus_counts))

            term_column = "noun" if pos == "noun" else "verb"
            placeholders = ", ".join("?" for _ in selected)
            rows = connection.execute(
                f"""
                SELECT corpus_id, noun, particle, verb, raw_frequency
                FROM collocation_frequency
                WHERE {term_column} = ? AND corpus_id IN ({placeholders})
                """,
                [term, *selected],
            ).fetchall()
            return corpus_counts, selected, rows

        corpus_counts, selected, rows = await run_bounded_query(
            connection,
            request.app.state.query_limiter,
            load_rows,
            timeout=QUERY_TIMEOUT_SECONDS,
        )

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

        request.state.result_count = sum(
            group.returnedCount for group in particle_groups
        )
        request.state.corpus_ids = selected
        return CollocationsResponse(
            particleGroups=particle_groups,
            selectedCorpusIds=selected,
            rankBy=rankBy,
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
                ORDER BY src.corpus_id, src.id, s.id
                LIMIT ?
                """,
                [noun, particle, verb, *selected, limit],
            ).fetchall()
            return selected, rows

        selected, rows = await run_bounded_query(
            connection,
            request.app.state.query_limiter,
            load_rows,
            timeout=QUERY_TIMEOUT_SECONDS,
        )
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
            selectedCorpusIds=selected,
            databaseBuildId=request.app.state.database_build_id,
        )

    return api
