from contextlib import asynccontextmanager
import hashlib
import json
from pathlib import Path
import re
from typing import Annotated, Literal
import uuid

import duckdb
from fastapi import FastAPI, Query, Request  # type: ignore
from fastapi.exceptions import RequestValidationError  # type: ignore
from fastapi.responses import JSONResponse  # type: ignore
from pydantic import BaseModel


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
    def __init__(self, status_code: int, code: str, message: str):
        self.status_code = status_code
        self.code = code
        self.message = message


def select_corpora(requested: list[str] | None, known: set[str]) -> list[str]:
    if requested is None:
        return sorted(known)

    selected = sorted(set(requested))
    if any(not corpus_id for corpus_id in selected) or not set(selected) <= known:
        raise PublicApiError(
            400, "invalid_parameter", "corpusId must name a known corpus"
        )
    return selected


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

    @api.exception_handler(PublicApiError)
    async def public_api_error(request: Request, error: PublicApiError):
        return JSONResponse(
            status_code=error.status_code,
            content={
                "error": {
                    "code": error.code,
                    "message": error.message,
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
            selected = select_corpora(corpusId, set(corpus_counts))

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
            selected = select_corpora(corpusId, set(all_corpora))
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
