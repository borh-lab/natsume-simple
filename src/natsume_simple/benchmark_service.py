"""Run the fixed, host-labelled service benchmark used for release diagnostics."""

from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import urlopen

from natsume_simple.api import (
    COLLOCATION_ITEM_CORPUS_BUDGET,
    MAX_COLLOCATIONS_PER_PARTICLE,
)


class BenchmarkError(RuntimeError):
    """The service did not complete the benchmark contract."""


@dataclass(frozen=True)
class BenchmarkEndpoint:
    """One named request in the fixed benchmark family."""

    name: str
    url: str


@dataclass(frozen=True)
class _Observation:
    endpoint: str
    elapsed_ms: float
    status: int
    body_bytes: int


def _url(base_url: str, path: str, parameters: list[tuple[str, str]]) -> str:
    return f"{base_url.rstrip('/')}{path}?{urlencode(parameters, doseq=True)}"


def build_request_family(
    base_url: str, corpus_ids: tuple[str, ...]
) -> tuple[BenchmarkEndpoint, ...]:
    """Build the stable requests used to compare immutable artifacts."""
    if not corpus_ids:
        raise ValueError("at least one corpus ID is required")
    corpora = [("corpusId", corpus_id) for corpus_id in corpus_ids]
    page_size = min(
        MAX_COLLOCATIONS_PER_PARTICLE,
        COLLOCATION_ITEM_CORPUS_BUDGET // len(corpus_ids),
    )

    def collocations(pos: str) -> str:
        term = "情報" if pos == "noun" else "行う"
        return _url(
            base_url,
            "/api/collocations",
            [
                ("term", term),
                ("pos", pos),
                ("limitPerParticle", "100"),
                *corpora,
            ],
        )

    return (
        BenchmarkEndpoint(
            "suggestions-noun",
            _url(
                base_url,
                "/api/suggestions",
                [("q", "情報"), ("pos", "noun"), ("limit", "10")],
            ),
        ),
        BenchmarkEndpoint("collocations-noun", collocations("noun")),
        BenchmarkEndpoint("collocations-verb", collocations("verb")),
        BenchmarkEndpoint(
            "examples",
            _url(
                base_url,
                "/api/examples",
                [
                    ("noun", "情報"),
                    ("particle", "が"),
                    ("verb", "含まれる"),
                    ("limit", "5"),
                    *corpora,
                ],
            ),
        ),
        BenchmarkEndpoint(
            "collocations-page-verb",
            _url(
                base_url,
                "/api/collocations",
                [
                    ("term", "する"),
                    ("pos", "verb"),
                    ("particle", "を"),
                    ("offsetPerParticle", "4500"),
                    ("limitPerParticle", str(page_size)),
                    *corpora,
                ],
            ),
        ),
        BenchmarkEndpoint(
            "examples-page",
            _url(
                base_url,
                "/api/examples",
                [
                    ("noun", "必要"),
                    ("particle", "が"),
                    ("verb", "ある"),
                    ("offset", "4000"),
                    ("limit", "20"),
                    *corpora,
                ],
            ),
        ),
    )


def _later_page_result_count(endpoint: str, body: bytes) -> int | None:
    if endpoint not in {"collocations-page-verb", "examples-page"}:
        return None
    try:
        payload = json.loads(body)
        if endpoint == "examples-page":
            return len(payload["examples"])
        return sum(group["returnedCount"] for group in payload["particleGroups"])
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise BenchmarkError(f"benchmark_response_invalid:{endpoint}") from error


def _fetch(endpoint: BenchmarkEndpoint, timeout: float) -> _Observation:
    started = time.perf_counter()
    try:
        with urlopen(endpoint.url, timeout=timeout) as response:
            status = response.status
            body = response.read()
    except HTTPError as error:
        status = error.code
        body = error.read()
    except (OSError, URLError) as error:
        raise BenchmarkError(f"benchmark_request_failed:{endpoint.name}") from error
    elapsed_ms = (time.perf_counter() - started) * 1_000
    if _later_page_result_count(endpoint.name, body) == 0:
        raise BenchmarkError(f"benchmark_empty_result:{endpoint.name}")
    return _Observation(endpoint.name, elapsed_ms, status, len(body))


def _percentile(values: list[float], fraction: float) -> float:
    """Return a linearly interpolated percentile from non-empty observations."""
    if not values:
        raise ValueError("percentile requires at least one value")
    if not 0 <= fraction <= 1:
        raise ValueError("percentile fraction must be between zero and one")
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def _summary(observations: list[_Observation], wall_seconds: float) -> dict[str, Any]:
    elapsed = [observation.elapsed_ms for observation in observations]
    body_bytes = [observation.body_bytes for observation in observations]
    successes = sum(observation.status == 200 for observation in observations)
    return {
        "requests": len(observations),
        "successes": successes,
        "bodyBytesMin": min(body_bytes),
        "bodyBytesMax": max(body_bytes),
        "p50Ms": _percentile(elapsed, 0.5),
        "p95Ms": _percentile(elapsed, 0.95),
        "maxMs": max(elapsed),
        "wallSeconds": wall_seconds,
        "requestsPerSecond": len(observations) / wall_seconds,
    }


def run_benchmark(
    endpoints: tuple[BenchmarkEndpoint, ...],
    *,
    request_count: int,
    concurrency: int,
    timeout: float,
) -> dict[str, object]:
    """Warm, execute, validate, and summarize a round-robin request family."""
    if not endpoints:
        raise ValueError("at least one endpoint is required")
    if request_count <= 0 or concurrency <= 0 or timeout <= 0:
        raise ValueError("request_count, concurrency, and timeout must be positive")

    for endpoint in endpoints:
        warmup = _fetch(endpoint, timeout)
        if warmup.status != 200:
            raise BenchmarkError(
                f"benchmark_http_status:{warmup.endpoint}:{warmup.status}"
            )

    schedule = [endpoints[index % len(endpoints)] for index in range(request_count)]
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        observations = list(
            executor.map(lambda endpoint: _fetch(endpoint, timeout), schedule)
        )
    wall_seconds = time.perf_counter() - started

    for observation in observations:
        if observation.status != 200:
            raise BenchmarkError(
                f"benchmark_http_status:{observation.endpoint}:{observation.status}"
            )

    report = _summary(observations, wall_seconds)
    report["endpoints"] = {
        endpoint.name: _summary(
            [item for item in observations if item.endpoint == endpoint.name],
            wall_seconds,
        )
        for endpoint in endpoints
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--corpus-id", action="append", required=True)
    parser.add_argument("--requests", type=int, default=500)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    endpoints = build_request_family(arguments.base_url, tuple(arguments.corpus_id))
    report = run_benchmark(
        endpoints,
        request_count=arguments.requests,
        concurrency=arguments.concurrency,
        timeout=arguments.timeout,
    )
    arguments.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
