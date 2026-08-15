import hashlib
import json
import logging
from pathlib import Path

import duckdb
import pytest
from fastapi.testclient import TestClient

from natsume_simple.api import create_app
from natsume_simple.artifact_registry import publish_artifact
from tests.database_fixture import (
    build_maximum_response_artifact,
    build_search_artifact,
)
from tests.test_artifact_builder import build_fixture


def fixture_client(tmp_path: Path) -> TestClient:
    artifact_dir = build_search_artifact(tmp_path / "artifact")
    return TestClient(create_app(artifact_dir))


def balanced_examples_client(tmp_path: Path) -> TestClient:
    artifact_dir = build_search_artifact(tmp_path / "artifact")
    database_path = artifact_dir / "corpus.duckdb"
    with duckdb.connect(str(database_path)) as connection:
        connection.execute(
            """
            INSERT INTO corpus VALUES ('gamma', 'Gamma');
            INSERT INTO source
                (id, corpus_id, external_id, title, content_sha256)
            VALUES (4, 'gamma', 'g1', 'Gamma one', 'sha-g1');
            INSERT INTO sentence VALUES
                (17, 4, 1, '情報を集める。'),
                (18, 4, 2, '情報を集める。');
            INSERT INTO collocation_occurrence VALUES
                (17, '情報', 'を', '集める', 0, 2, 2, 3, 3, 6,
                 'fixture-extractor'),
                (18, '情報', 'を', '集める', 0, 2, 2, 3, 3, 6,
                 'fixture-extractor');
            INSERT INTO corpus_stats VALUES ('gamma', 1, 2, 2);
            """
        )
    manifest_path = artifact_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["databaseSha256"] = hashlib.sha256(database_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    return TestClient(create_app(artifact_dir))


def assert_database_unavailable(response, request_id: str):
    assert response.status_code == 503
    assert response.json() == {
        "error": {
            "code": "database_unavailable",
            "message": "The database artifact is unavailable",
            "requestId": request_id,
        }
    }


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
                    "collocationCount": 13,
                    "sentenceCount": 12,
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


def test_startup_resolves_artifact_once(tmp_path: Path):
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    first = build_fixture(artifacts / "first", "first-build")
    second = build_fixture(artifacts / "second", "second-build")
    database_path = second / "corpus.duckdb"
    with duckdb.connect(str(database_path)) as connection:
        connection.execute("UPDATE corpus SET label = 'Changed' WHERE id = 'alpha'")
    manifest_path = second / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["databaseSha256"] = hashlib.sha256(database_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))

    deploy = tmp_path / "deploy"
    publish_artifact(first, deploy)
    with TestClient(create_app(deploy / "current")) as client:
        publish_artifact(second, deploy)

        assert (
            client.get("/api/health/ready").json()["databaseBuildId"] == "first-build"
        )
        corpora = client.get("/api/corpora").json()["corpora"]

    assert [corpus["label"] for corpus in corpora] == ["Alpha", "Beta"]


@pytest.mark.parametrize(
    ("damage", "reason"),
    [
        ("missing_manifest", "manifest_missing"),
        ("checksum", "checksum_mismatch"),
        ("schema", "unsupported_schema"),
        ("invalid_database", "database_unreadable"),
        ("corpus_count", "invalid_corpus_count"),
        ("corpus_id", "invalid_corpus_id"),
    ],
)
def test_invalid_artifact_stays_live_but_never_becomes_ready(
    caplog, tmp_path: Path, damage: str, reason: str
):
    caplog.set_level(logging.ERROR, logger="natsume_simple.api.startup")
    artifact_dir = build_search_artifact(tmp_path / "artifact")
    manifest_path = artifact_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())

    if damage == "missing_manifest":
        manifest_path.unlink()
    elif damage == "checksum":
        manifest["databaseSha256"] = "0" * 64
        manifest_path.write_text(json.dumps(manifest))
    elif damage == "schema":
        manifest["schemaVersion"] = 2
        manifest_path.write_text(json.dumps(manifest))
    elif damage == "invalid_database":
        database_path = artifact_dir / "corpus.duckdb"
        database_path.write_bytes(b"not a DuckDB database")
        manifest["databaseSha256"] = hashlib.sha256(
            database_path.read_bytes()
        ).hexdigest()
        manifest_path.write_text(json.dumps(manifest))
    elif damage == "corpus_count":
        database_path = artifact_dir / "corpus.duckdb"
        with duckdb.connect(str(database_path)) as connection:
            connection.execute(
                "INSERT INTO corpus VALUES ('extra', 'Extra'), ('extra2', 'Extra 2')"
            )
        manifest["databaseSha256"] = hashlib.sha256(
            database_path.read_bytes()
        ).hexdigest()
        manifest_path.write_text(json.dumps(manifest))
    else:
        database_path = artifact_dir / "corpus.duckdb"
        with duckdb.connect(str(database_path)) as connection:
            connection.execute(
                "INSERT INTO corpus VALUES ('thirteenchars', 'Invalid ID')"
            )
        manifest["databaseSha256"] = hashlib.sha256(
            database_path.read_bytes()
        ).hexdigest()
        manifest_path.write_text(json.dumps(manifest))

    with TestClient(create_app(artifact_dir), raise_server_exceptions=False) as client:
        assert client.get("/api/health/live").json() == {"status": "ok"}
        response = client.get(
            "/api/health/ready", headers={"X-Request-ID": f"broken-{damage}"}
        )
        search_response = client.get(
            "/api/suggestions",
            params={"q": "情報", "pos": "noun"},
            headers={"X-Request-ID": f"search-{damage}"},
        )

    assert_database_unavailable(response, f"broken-{damage}")
    assert_database_unavailable(search_response, f"search-{damage}")
    startup_record = next(
        record
        for record in caplog.records
        if record.name == "natsume_simple.api.startup"
    )
    logged = json.loads(startup_record.message)
    assert logged == {
        "timestamp": logged["timestamp"],
        "level": "ERROR",
        "event": "artifact_rejected",
        "reason": reason,
    }
    assert str(artifact_dir) not in caplog.text


def test_ready_becomes_unavailable_when_database_disappears(tmp_path: Path):
    artifact_dir = build_search_artifact(tmp_path / "artifact")
    with TestClient(create_app(artifact_dir), raise_server_exceptions=False) as client:
        (artifact_dir / "corpus.duckdb").unlink()
        response = client.get(
            "/api/health/ready", headers={"X-Request-ID": "database-removed"}
        )

    assert_database_unavailable(response, "database-removed")


def test_unexpected_errors_use_a_sanitized_common_envelope(tmp_path: Path):
    artifact_dir = build_search_artifact(tmp_path / "artifact")
    api = create_app(artifact_dir)

    @api.get("/api/test/unexpected")
    def unexpected():
        raise RuntimeError("SELECT secret FROM /private/database")

    with TestClient(api, raise_server_exceptions=False) as client:
        response = client.get(
            "/api/test/unexpected", headers={"X-Request-ID": "unexpected-request"}
        )

    assert response.status_code == 500
    assert response.json() == {
        "error": {
            "code": "internal_error",
            "message": "An unexpected error occurred",
            "requestId": "unexpected-request",
        }
    }
    assert "secret" not in response.text


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


def test_request_log_contains_diagnostics_without_the_query(caplog, tmp_path: Path):
    caplog.set_level(logging.INFO, logger="natsume_simple.api.requests")

    with fixture_client(tmp_path) as client:
        response = client.get(
            "/api/collocations",
            params={
                "term": "情報",
                "pos": "noun",
                "corpusId": "beta",
                "limitPerParticle": 1,
            },
            headers={"X-Request-ID": "logged-request"},
        )

    assert response.status_code == 200
    record = json.loads(caplog.records[-1].message)
    assert record == {
        "timestamp": record["timestamp"],
        "level": "INFO",
        "requestId": "logged-request",
        "route": "/api/collocations",
        "status": 200,
        "durationMs": record["durationMs"],
        "databaseBuildId": "fixture-build-001",
        "resultCount": 2,
        "corpusIds": ["beta"],
        "queryLengthBucket": "2–4",
    }
    assert record["durationMs"] >= 0
    assert "情報" not in caplog.text


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


def test_exhausted_query_capacity_returns_retryable_error(caplog, tmp_path: Path):
    caplog.set_level(logging.INFO, logger="natsume_simple.api.requests")
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
    record = json.loads(caplog.records[-1].message)
    assert record["status"] == 429
    assert record["queryLengthBucket"] == "2–4"
    assert "情報" not in caplog.text


def test_collocations_select_and_normalize_before_applying_the_limit(tmp_path: Path):
    with fixture_client(tmp_path) as client:
        response = client.get(
            "/api/collocations",
            params={
                "term": "情報",
                "pos": "noun",
                "corpusId": "beta",
                "limitPerParticle": 1,
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["selectedCorpusIds"] == ["beta"]
    assert "rankBy" not in body
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
                }
            ],
        }
    ]
    assert particle["corpusDistribution"] == [
        {
            "corpusId": "beta",
            "rawFrequency": 3,
            "frequencyPerMillion": 750_000,
        }
    ]


def test_collocations_rank_by_mean_frequency_per_million(tmp_path: Path):
    with fixture_client(tmp_path) as client:
        response = client.get(
            "/api/collocations",
            params={"term": "情報", "pos": "noun", "limitPerParticle": 1},
        )

    particle = next(
        group
        for group in response.json()["particleGroups"]
        if group["particle"] == "を"
    )
    assert particle["items"][0]["verb"] == "調べる"
    assert particle["items"][0]["totalRawFrequency"] == 2


def test_collocations_page_one_particle_without_overlap(tmp_path: Path):
    with fixture_client(tmp_path) as client:
        first = client.get(
            "/api/collocations",
            params={
                "term": "情報",
                "pos": "noun",
                "corpusId": "beta",
                "particle": "を",
                "limitPerParticle": 1,
                "offsetPerParticle": 0,
            },
        )
        second = client.get(
            "/api/collocations",
            params={
                "term": "情報",
                "pos": "noun",
                "corpusId": "beta",
                "particle": "を",
                "limitPerParticle": 1,
                "offsetPerParticle": 1,
            },
        )

    first_group = first.json()["particleGroups"][0]
    second_group = second.json()["particleGroups"][0]
    assert [group["particle"] for group in first.json()["particleGroups"]] == ["を"]
    assert first_group["totalMatchingCollocations"] == 2
    assert second_group["totalMatchingCollocations"] == 2
    assert first_group["corpusDistribution"] == second_group["corpusDistribution"]
    assert first_group["items"][0]["verb"] == "調べる"
    assert second_group["items"][0]["verb"] == "集める"


def test_collocations_past_end_retains_target_group_metadata(tmp_path: Path):
    with fixture_client(tmp_path) as client:
        response = client.get(
            "/api/collocations",
            params={
                "term": "情報",
                "pos": "noun",
                "corpusId": "beta",
                "particle": "を",
                "limitPerParticle": 1,
                "offsetPerParticle": 2,
            },
        )

    group = response.json()["particleGroups"][0]
    assert group["particle"] == "を"
    assert group["totalMatchingCollocations"] == 2
    assert group["returnedCount"] == 0
    assert group["items"] == []
    assert group["corpusDistribution"] == [
        {
            "corpusId": "beta",
            "rawFrequency": 3,
            "frequencyPerMillion": 750_000,
        }
    ]


def test_collocation_offset_requires_particle(tmp_path: Path):
    with fixture_client(tmp_path) as client:
        response = client.get(
            "/api/collocations",
            params={
                "term": "情報",
                "pos": "noun",
                "offsetPerParticle": 1,
            },
            headers={"X-Request-ID": "missing-particle"},
        )

    assert response.status_code == 400
    assert response.json() == {
        "error": {
            "code": "invalid_parameter",
            "message": "offsetPerParticle requires particle",
            "requestId": "missing-particle",
        }
    }


def test_collocation_particle_rejects_unknown_value(tmp_path: Path):
    with fixture_client(tmp_path) as client:
        response = client.get(
            "/api/collocations",
            params={"term": "情報", "pos": "noun", "particle": "unknown"},
        )

    assert response.status_code == 422


def test_maximum_budgeted_responses_stay_below_one_mebibyte(tmp_path: Path):
    artifact_dir = build_maximum_response_artifact(tmp_path / "artifact")
    with TestClient(create_app(artifact_dir)) as client:
        collocations = client.get(
            "/api/collocations",
            params={
                "term": "名" * 64,
                "pos": "noun",
                "corpusId": ["max-a", "max-b"],
                "limitPerParticle": 200,
            },
        )
        three_corpus_collocations = client.get(
            "/api/collocations",
            params={
                "term": "名" * 64,
                "pos": "noun",
                "corpusId": ["max-a", "max-b", "max-c"],
                "limitPerParticle": 150,
            },
        )
        examples = client.get(
            "/api/examples",
            params={
                "noun": "例",
                "particle": "を",
                "verb": "示す",
                "corpusId": "max-a",
                "limit": 20,
            },
        )

    assert collocations.status_code == 200
    assert len(collocations.content) < 1024 * 1024
    assert len(collocations.json()["particleGroups"]) == 8
    assert all(
        group["returnedCount"] == 200 for group in collocations.json()["particleGroups"]
    )
    assert three_corpus_collocations.status_code == 200
    assert len(three_corpus_collocations.content) < 1024 * 1024
    assert all(
        group["returnedCount"] == 150
        for group in three_corpus_collocations.json()["particleGroups"]
    )
    assert examples.status_code == 200
    assert len(examples.content) < 1024 * 1024
    assert len(examples.json()["examples"]) == 20


def test_collocation_budget_rejects_three_corpora_above_150_items(tmp_path: Path):
    with fixture_client(tmp_path) as client:
        response = client.get(
            "/api/collocations",
            params={
                "term": "情報",
                "pos": "noun",
                "corpusId": ["alpha", "beta"],
                "limitPerParticle": 200,
            },
        )

    assert response.status_code == 200

    artifact_dir = build_maximum_response_artifact(tmp_path / "maximum-artifact")
    with TestClient(create_app(artifact_dir)) as client:
        response = client.get(
            "/api/collocations",
            params={
                "term": "名" * 64,
                "pos": "noun",
                "limitPerParticle": 151,
            },
            headers={"X-Request-ID": "over-budget"},
        )

    assert response.status_code == 400
    assert response.json() == {
        "error": {
            "code": "invalid_parameter",
            "message": "limitPerParticle is too large for the selected corpus count",
            "requestId": "over-budget",
        }
    }


def test_collocation_limit_rejects_values_above_200(tmp_path: Path):
    with fixture_client(tmp_path) as client:
        response = client.get(
            "/api/collocations",
            params={
                "term": "情報",
                "pos": "noun",
                "limitPerParticle": 201,
            },
        )

    assert response.status_code == 422


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
        "hasMore": False,
        "selectedCorpusIds": ["beta"],
        "databaseBuildId": "fixture-build-001",
    }


def test_examples_pages_same_sentence_occurrences_in_total_order(tmp_path: Path):
    with fixture_client(tmp_path) as client:
        pages = [
            client.get(
                "/api/examples",
                params={
                    "noun": "情報",
                    "particle": "を",
                    "verb": "集める",
                    "corpusId": "alpha",
                    "limit": 1,
                    "offset": offset,
                },
            ).json()
            for offset in range(5)
        ]

    assert [page["examples"][0]["nounSpan"] for page in pages[:3]] == [
        {"start": 0, "end": 2},
        {"start": 7, "end": 9},
        {"start": 0, "end": 2},
    ]
    assert [page["hasMore"] for page in pages] == [
        True,
        True,
        False,
        False,
        False,
    ]
    assert pages[3]["examples"] == []
    assert pages[4]["examples"] == []


def test_balanced_examples_round_robin_and_fill_exhausted(tmp_path: Path):
    with balanced_examples_client(tmp_path) as client:
        first = client.get(
            "/api/examples",
            params={
                "noun": "情報",
                "particle": "を",
                "verb": "集める",
                "limit": 5,
            },
        ).json()
        remainder = client.get(
            "/api/examples",
            params={
                "noun": "情報",
                "particle": "を",
                "verb": "集める",
                "limit": 5,
                "offset": 5,
            },
        ).json()

    assert [example["corpusId"] for example in first["examples"]] == [
        "alpha",
        "beta",
        "gamma",
        "alpha",
        "gamma",
    ]
    assert first["hasMore"] is True
    assert [example["corpusId"] for example in remainder["examples"]] == ["alpha"]
    assert remainder["hasMore"] is False


def test_example_pages_compose_and_repeat(tmp_path: Path):
    params = {"noun": "情報", "particle": "を", "verb": "集める"}
    with balanced_examples_client(tmp_path) as client:
        pages = [
            client.get(
                "/api/examples", params={**params, "limit": 2, "offset": offset}
            ).json()
            for offset in (0, 2, 4)
        ]
        complete = client.get("/api/examples", params={**params, "limit": 20}).json()
        repeated = client.get("/api/examples", params={**params, "limit": 20}).json()

    assert [[item["corpusId"] for item in page["examples"]] for page in pages] == [
        ["alpha", "beta"],
        ["gamma", "alpha"],
        ["gamma", "alpha"],
    ]
    combined = [item for page in pages for item in page["examples"]]
    assert combined == complete["examples"]
    assert repeated == complete
    identities = [
        (
            item["corpusId"],
            item["sourceId"],
            item["sentenceId"],
            tuple(item["nounSpan"].items()),
            tuple(item["particleSpan"].items()),
            tuple(item["verbSpan"].items()),
        )
        for item in combined
    ]
    assert len(identities) == len(set(identities)) == 6


def test_example_offset_rejects_negative_values(tmp_path: Path):
    with fixture_client(tmp_path) as client:
        response = client.get(
            "/api/examples",
            params={
                "noun": "情報",
                "particle": "を",
                "verb": "集める",
                "offset": -1,
            },
        )

    assert response.status_code == 422


def test_example_offset_rejects_values_outside_duckdb_range(tmp_path: Path):
    with fixture_client(tmp_path) as client:
        response = client.get(
            "/api/examples",
            params={
                "noun": "情報",
                "particle": "を",
                "verb": "集める",
                "offset": 2**63,
            },
        )

    assert response.status_code == 422


@pytest.mark.parametrize("corpus_id", ["unknown", ""])
@pytest.mark.parametrize("route", ["collocations", "examples"])
def test_corpus_selection_rejects_unknown_and_empty_ids(
    tmp_path: Path, route: str, corpus_id: str
):
    params = (
        {"term": "情報", "pos": "noun"}
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


def test_runtime_app_serves_frontend_after_api_routes(tmp_path: Path):
    artifact_dir = build_search_artifact(tmp_path / "artifact")
    frontend_dir = tmp_path / "frontend"
    frontend_dir.mkdir()
    (frontend_dir / "index.html").write_text("<h1>Natsume fixture</h1>")

    with TestClient(create_app(artifact_dir, frontend_dir=frontend_dir)) as client:
        assert client.get("/api/health/live").json() == {"status": "ok"}
        assert client.get("/").text == "<h1>Natsume fixture</h1>"
