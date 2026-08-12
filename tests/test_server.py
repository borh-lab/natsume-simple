from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests.database_fixture import build_search_database


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from natsume_simple import server

    database_path = build_search_database(tmp_path / "corpus.db")
    monkeypatch.setattr(server, "DATABASE_PATH", database_path)

    with TestClient(server.app, raise_server_exceptions=False) as test_client:
        yield test_client


def test_corpus_normalization_uses_fixture_counts(client: TestClient):
    response = client.get("/corpus/norm")

    assert response.status_code == 200
    assert response.json() == {
        "alpha": {"normalizationFactor": 1 / 3, "collocationCount": 3},
        "beta": {"normalizationFactor": 1, "collocationCount": 1},
    }


@pytest.mark.parametrize(
    ("search_type", "term", "particle", "other_lemma"),
    [
        ("noun", "情報", "を", "集める"),
        ("verb", "集める", "を", "情報"),
    ],
)
def test_npv_search_supports_both_directions(
    client: TestClient,
    search_type: str,
    term: str,
    particle: str,
    other_lemma: str,
):
    response = client.get(f"/npv/{search_type}/{term}")

    assert response.status_code == 200
    body = response.json()
    assert body["totalResults"] == 1
    match = body["particleGroups"][particle]["collocates"][0]
    assert other_lemma in (match["n"], match["v"])
    assert match["totalRawFrequency"] == 3


def test_search_suggestions_and_zero_results(client: TestClient):
    assert client.get("/search/情報").json() == [["情報", "n"]]
    assert client.get("/search/存在しない").json() == []


def test_examples_preserve_text_and_offsets(client: TestClient):
    response = client.get("/sentences/情報/を/集める/10")

    assert response.status_code == 200
    examples = response.json()
    assert len(examples) == 3
    shaped = next(item for item in examples if item["corpus"] == "beta")
    assert shaped["text"] == "<img src=x onerror=alert(1)>情報を集める。"
    assert (shaped["n_begin"], shaped["n_end"]) == (28, 30)


def test_invalid_search_direction_is_rejected(client: TestClient):
    response = client.get("/npv/adjective/情報")

    assert response.status_code == 500
