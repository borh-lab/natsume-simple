from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import pytest

from natsume_simple.benchmark_service import (
    BenchmarkEndpoint,
    BenchmarkError,
    _percentile,
    build_request_family,
    run_benchmark,
)


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        status = 503 if self.path.startswith("/failure") else 200
        if self.path.startswith("/empty-collocations"):
            body = json.dumps({"particleGroups": [{"returnedCount": 0}]}).encode()
        elif self.path.startswith("/empty-examples"):
            body = json.dumps({"examples": []}).encode()
        else:
            body = self.path.encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        pass


@contextmanager
def _http_server() -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        host, port = server.server_address
        yield f"http://{host}:{port}"
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_request_family_is_fixed_and_repeats_corpus_ids() -> None:
    endpoints = build_request_family("http://127.0.0.1:8000/", ("jnlp", "wiki"))

    assert [endpoint.name for endpoint in endpoints] == [
        "suggestions-noun",
        "collocations-noun",
        "collocations-verb",
        "examples",
        "collocations-page-verb",
        "examples-page",
    ]
    queries = {
        endpoint.name: parse_qs(urlparse(endpoint.url).query) for endpoint in endpoints
    }
    assert queries["suggestions-noun"] == {
        "q": ["情報"],
        "pos": ["noun"],
        "limit": ["10"],
    }
    for name, query in queries.items():
        if name != "suggestions-noun":
            assert query["corpusId"] == ["jnlp", "wiki"]
    assert "rankBy" not in queries["collocations-noun"]
    assert queries["collocations-verb"]["pos"] == ["verb"]
    assert queries["collocations-verb"]["term"] == ["行う"]
    assert queries["examples"]["limit"] == ["5"]
    assert queries["collocations-page-verb"] == {
        "term": ["する"],
        "pos": ["verb"],
        "particle": ["を"],
        "offsetPerParticle": ["4500"],
        "limitPerParticle": ["200"],
        "corpusId": ["jnlp", "wiki"],
    }
    assert queries["examples-page"] == {
        "noun": ["必要"],
        "particle": ["が"],
        "verb": ["ある"],
        "offset": ["4000"],
        "limit": ["20"],
        "corpusId": ["jnlp", "wiki"],
    }


@pytest.mark.parametrize(
    ("name", "path"),
    [
        ("collocations-page-verb", "/empty-collocations"),
        ("examples-page", "/empty-examples"),
    ],
)
def test_benchmark_rejects_empty_later_page(name: str, path: str) -> None:
    with _http_server() as base_url:
        endpoints = (BenchmarkEndpoint(name, f"{base_url}{path}"),)

        with pytest.raises(BenchmarkError, match=f"benchmark_empty_result:{name}"):
            run_benchmark(endpoints, request_count=1, concurrency=1, timeout=2.0)


def test_percentile_uses_linear_interpolation() -> None:
    values = [1.0, 2.0, 3.0, 4.0]

    assert _percentile(values, 0.5) == 2.5
    assert _percentile(values, 0.95) == pytest.approx(3.85)


def test_benchmark_round_robins_and_summarizes_responses() -> None:
    with _http_server() as base_url:
        endpoints = (
            BenchmarkEndpoint("short", f"{base_url}/a"),
            BenchmarkEndpoint("long", f"{base_url}/longer"),
        )
        report = run_benchmark(endpoints, request_count=5, concurrency=2, timeout=2.0)

    assert report["requests"] == 5
    assert report["successes"] == 5
    assert report["bodyBytesMin"] == len("/a")
    assert report["bodyBytesMax"] == len("/longer")
    assert report["p50Ms"] >= 0
    assert report["p95Ms"] >= report["p50Ms"]
    assert report["maxMs"] >= report["p95Ms"]
    assert report["wallSeconds"] > 0
    assert report["requestsPerSecond"] > 0
    assert report["endpoints"]["short"]["requests"] == 3
    assert report["endpoints"]["long"]["requests"] == 2
    json.dumps(report)


def test_benchmark_rejects_non_200_response() -> None:
    with _http_server() as base_url:
        endpoints = (BenchmarkEndpoint("failure", f"{base_url}/failure"),)

        with pytest.raises(BenchmarkError, match="benchmark_http_status:failure:503"):
            run_benchmark(endpoints, request_count=1, concurrency=1, timeout=2.0)


@pytest.mark.parametrize(
    ("request_count", "concurrency", "timeout"),
    [(0, 1, 1.0), (1, 0, 1.0), (1, 1, 0.0)],
)
def test_benchmark_rejects_invalid_arguments(
    request_count: int, concurrency: int, timeout: float
) -> None:
    endpoint = BenchmarkEndpoint("ok", "http://127.0.0.1:1/ok")

    with pytest.raises(ValueError):
        run_benchmark(
            (endpoint,),
            request_count=request_count,
            concurrency=concurrency,
            timeout=timeout,
        )
