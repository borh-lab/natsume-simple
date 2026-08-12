from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from natsume_simple.api import create_app
from tests.database_fixture import build_search_artifact


def fixture_client(tmp_path: Path) -> TestClient:
    artifact_dir = build_search_artifact(tmp_path / "artifact")
    return TestClient(create_app(artifact_dir))


def test_ready_and_corpora_describe_the_deployed_artifact(tmp_path: Path):
    client = fixture_client(tmp_path)

    with client:
        assert client.get("/api/health/live").json() == {"status": "ok"}
        assert client.get("/api/health/ready").json() == {
            "status": "ok",
            "databaseBuildId": "fixture-build-001",
            "schemaVersion": 1,
        }
        assert client.get("/api/corpora").json() == {
            "corpora": [
                {
                    "id": "alpha",
                    "label": "Alpha",
                    "collocationCount": 6,
                    "sentenceCount": 6,
                },
                {
                    "id": "beta",
                    "label": "Beta",
                    "collocationCount": 4,
                    "sentenceCount": 4,
                },
            ],
            "databaseBuildId": "fixture-build-001",
        }


def test_suggestions_filter_part_of_speech_and_order_by_frequency(tmp_path: Path):
    with fixture_client(tmp_path) as client:
        response = client.get(
            "/api/suggestions", params={"q": "める", "pos": "verb", "limit": 10}
        )

    assert response.status_code == 200
    assert response.json() == {
        "suggestions": [
            {"lemma": "集める", "pos": "verb", "occurrenceCount": 3},
            {"lemma": "進める", "pos": "verb", "occurrenceCount": 2},
        ]
    }


def test_validation_errors_use_the_public_envelope(tmp_path: Path):
    with fixture_client(tmp_path) as client:
        response = client.get(
            "/api/suggestions",
            params={"q": "情報", "pos": "noun", "limit": 0},
            headers={"X-Request-ID": "lesson-request-1"},
        )

    assert response.status_code == 422
    assert response.json() == {
        "error": {
            "code": "invalid_parameter",
            "message": "Request parameters are invalid",
            "requestId": "lesson-request-1",
        }
    }


def test_exhausted_query_capacity_returns_retryable_error(tmp_path: Path):
    with fixture_client(tmp_path) as client:
        limiter = client.app.state.query_limiter
        borrowers = [object() for _ in range(limiter.total_tokens)]
        for borrower in borrowers:
            limiter.acquire_on_behalf_of_nowait(borrower)
        try:
            response = client.get(
                "/api/suggestions",
                params={"q": "情報", "pos": "noun"},
                headers={"X-Request-ID": "busy-request"},
            )
        finally:
            for borrower in borrowers:
                limiter.release_on_behalf_of(borrower)

    assert response.status_code == 429
    assert response.headers["retry-after"] == "1"
    assert response.json() == {
        "error": {
            "code": "capacity_exceeded",
            "message": "Query capacity is exhausted",
            "requestId": "busy-request",
        }
    }


def test_collocations_select_and_rank_before_applying_the_limit(tmp_path: Path):
    with fixture_client(tmp_path) as client:
        response = client.get(
            "/api/collocations",
            params={
                "term": "情報",
                "pos": "noun",
                "corpusId": "beta",
                "rankBy": "raw",
                "limitPerParticle": 1,
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["selectedCorpusIds"] == ["beta"]
    assert body["rankBy"] == "raw"
    assert body["databaseBuildId"] == "fixture-build-001"
    particle = next(
        group for group in body["particleGroups"] if group["particle"] == "を"
    )
    assert particle["totalMatchingCollocations"] == 2
    assert particle["returnedCount"] == 1
    assert particle["items"] == [
        {
            "noun": "情報",
            "particle": "を",
            "verb": "調べる",
            "totalRawFrequency": 2,
            "meanFrequencyPerMillion": 500_000,
            "contributions": [
                {
                    "corpusId": "beta",
                    "rawFrequency": 2,
                    "frequencyPerMillion": 500_000,
                }
            ],
        }
    ]


def test_examples_return_selected_plain_text_and_typed_spans(tmp_path: Path):
    with fixture_client(tmp_path) as client:
        response = client.get(
            "/api/examples",
            params={
                "noun": "情報",
                "particle": "を",
                "verb": "集める",
                "corpusId": "beta",
                "limit": 5,
            },
        )

    assert response.status_code == 200
    assert response.json() == {
        "examples": [
            {
                "corpusId": "beta",
                "sourceId": 3,
                "sourceTitle": "Beta one",
                "sentenceId": 3,
                "text": "<img src=x onerror=alert(1)>情報を集める。",
                "nounSpan": {"start": 28, "end": 30},
                "particleSpan": {"start": 30, "end": 31},
                "verbSpan": {"start": 31, "end": 34},
            }
        ],
        "selectedCorpusIds": ["beta"],
        "databaseBuildId": "fixture-build-001",
    }


@pytest.mark.parametrize("corpus_id", ["unknown", ""])
@pytest.mark.parametrize("route", ["collocations", "examples"])
def test_corpus_selection_rejects_unknown_and_empty_ids(
    tmp_path: Path, route: str, corpus_id: str
):
    params = (
        {"term": "情報", "pos": "noun", "rankBy": "meanPerMillion"}
        if route == "collocations"
        else {"noun": "情報", "particle": "を", "verb": "集める"}
    )
    params["corpusId"] = corpus_id

    with fixture_client(tmp_path) as client:
        response = client.get(
            f"/api/{route}",
            params=params,
            headers={"X-Request-ID": "bad-corpus-selection"},
        )

    assert response.status_code == 400
    assert response.json() == {
        "error": {
            "code": "invalid_parameter",
            "message": "corpusId must name a known corpus",
            "requestId": "bad-corpus-selection",
        }
    }
