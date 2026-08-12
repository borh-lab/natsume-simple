# Safe Public Service and Typed API Design

**Status:** Revised draft after written review

**Date:** 2026-08-12
**Boundary:** Anonymous public reads, API protocol, runtime database ownership,
and rendering trust boundary

## Problem

The current server opens a module-global DuckDB connection during import,
shares it across FastAPI worker threads, calculates global corpus state at
import time, accepts unbounded search inputs, returns loosely typed dictionaries,
and raises `ValueError` for invalid public parameters. The frontend and backend
disagree about contribution fields. Corpus sentences are inserted through
`{@html}`, and tooltip markup is assigned with `innerHTML`.

The service is intended to be public, anonymous, read-only, and single-instance.
There are no external API consumers, so backend and frontend may adopt one
honest contract atomically.

## Goals

- Define one validated and documented public protocol under `/api`.
- Treat corpus and search content as untrusted text from persistence through DOM.
- Bound all public work and response cardinality.
- Make database startup, connection concurrency, readiness, and shutdown
  explicit.
- Return statistical facts, not presentation geometry.
- Make OpenAPI the checked source for frontend compile-time API types.

## Non-Goals

- Authentication, accounts, cookies, write routes, application-level distributed
  rate-limit infrastructure, multi-instance coordination, API backward
  compatibility, or cursor pagination. Edge limits and bounded in-process query
  concurrency remain release requirements.
- Production corpus acquisition/building; Spec 3 owns the real artifact.
- Full frontend state decomposition; Spec 5 owns that refactor.

## Runtime Ownership

FastAPI lifespan resolves an explicit artifact-directory configuration, reads
`manifest.json`, verifies supported schema version and database identity, and
opens the database in DuckDB read-only mode before readiness becomes true. It
closes owned resources on shutdown. Importing the module does not open a database.

A single FastAPI dependency opens and yields a request-local read-only DuckDB
connection, then closes it. Connections are never shared concurrently across
Python request threads. There is no connection-manager class, pool, repository,
or factory hierarchy to explain on the backend walkthrough.

This follows DuckDB's Python guidance that parallel threads use their own
connections and FastAPI's lifespan guidance for process-lifetime resources:

- <https://duckdb.org/docs/stable/clients/python/overview>
- <https://fastapi.tiangolo.com/advanced/events/>

## Public Routes

The static Svelte application owns `/`. Application routes use `/api`; no `/v1`
namespace is introduced while frontend and backend deploy atomically.

### `GET /api/health/live`

Returns `200 {"status":"ok"}` without querying DuckDB.

### `GET /api/health/ready`

Returns `200` with `status`, `databaseBuildId`, and `schemaVersion` after a
trivial database read. It returns the common `503 database_unavailable` error if
the artifact is missing, incompatible, or unreadable.
`databaseBuildId` is the deployed artifact instance ID, not the semantic build
ID, so every publish/rollback changes browser-visible artifact identity.

### `GET /api/corpora`

Returns ordered corpus records:

```json
{
  "corpora": [
    {
      "id": "ted",
      "label": "TED",
      "collocationCount": 123456,
      "sentenceCount": 234567
    }
  ],
  "databaseBuildId": "20260812T120000Z-128bit-random"
}
```

Corpus IDs are stable ASCII identifiers. Labels are display strings.

### `GET /api/suggestions`

Parameters:

- `q`: required, 1–64 Unicode code points.
- `pos`: required enum `noun | verb`.
- `limit`: default 10, integer 1–20.

Initial matching is substring-based. Results contain `lemma`, `pos`, and
`occurrenceCount`, ordered by count descending and lemma ascending.

### `GET /api/collocations`

Parameters:

- `term`: required, 1–64 Unicode code points.
- `pos`: required enum `noun | verb`.
- `corpusId`: repeatable stable corpus ID. Absence means all corpora; duplicates
  are normalized; an empty or unknown value is `400`.
- `rankBy`: required enum `raw | meanPerMillion`.
- `limitPerParticle`: default 100, integer 1–200.

At most the configured eight particle groups are returned. Each group contains
`particle`, `totalMatchingCollocations`, `returnedCount`, bounded `items`, and
`corpusDistribution`. `totalMatchingCollocations` is the count of distinct
collocation triples matching the term, particle, and selected corpus set before
the item limit. Each item contains `noun`, `particle`, `verb`,
`totalRawFrequency`, `meanFrequencyPerMillion`, and contributions containing
`corpusId`, `rawFrequency`, and `frequencyPerMillion`.

The server applies corpus selection, computes both aggregate metrics, orders by
the requested metric, and only then applies `limitPerParticle`. Raw ranking uses
total raw frequency descending then mean per-million descending; normalized
ranking uses mean per-million descending then total raw frequency descending.
Both finish with noun/particle/verb lexical order. Client-side filtering or
reranking of a server-truncated page is not part of the contract.
The response echoes canonical `selectedCorpusIds`, `rankBy`, and
`databaseBuildId`, so stale or mismatched data cannot be presented as the result
of a newer selection.

### `GET /api/examples`

Parameters:

- `noun`, `particle`, `verb`: required, each 1–64 Unicode code points.
- `corpusId`: repeatable with the same selection semantics as collocations.
- `limit`: default 5, integer 1–20.

Examples order by corpus ID, source ID, and sentence ID. Each result contains
corpus/source metadata, plain sentence text, and three typed half-open spans
`[start, end)`. The response echoes canonical `selectedCorpusIds` and
`databaseBuildId` and never contains HTML.

## Statistical Contract

For corpus `c`:

```text
frequencyPerMillion(c) =
    rawFrequency(c) / corpusCollocationCount(c) × 1,000,000
```

For selected corpus set `S`, a missing contribution has rate zero:

```text
totalRawFrequency = sum(rawFrequency(c) for c in S)
meanFrequencyPerMillion =
    sum(frequencyPerMillion(c) for c in S) / |S|
```

The arithmetic mean gives each selected corpus equal weight without making the
metric grow merely because another corpus was selected. It is deliberately
distinct from a pooled rate, which weights corpora by their collocation counts.
The server returns selection-specific numeric facts and ranking. The frontend
computes only presentation geometry, colors, and tooltip layout.

The wire contract removes `normalizedWidth`, `normalizedOffset`, `rawWidth`,
`rawOffset`, `total_normalized`, and `total_raw`.

## Error Contract

Every non-success JSON response uses:

```json
{
  "error": {
    "code": "invalid_parameter",
    "message": "limitPerParticle must be between 1 and 200",
    "requestId": "01..."
  }
}
```

- `400` for a validly encoded but invalid semantic combination.
- `404` for an explicitly addressed resource that does not exist.
- `422` for parameter validation, converted to the common envelope.
- `429` for application admission-capacity exhaustion.
- `503` for database absence/incompatibility/unavailability.
- `504` for an interrupted query deadline.
- `500` for unexpected failure without SQL, filesystem paths, tracebacks, or
  corpus content.

No matches is `200` with empty collections. Pydantic response models validate,
document, and filter all response fields as recommended by FastAPI:
<https://fastapi.tiangolo.com/tutorial/response-model/>.

## Trust and Rendering Requirements

- Corpus content and all API strings are untrusted.
- Sentence highlighting splits plain text into validated text segments and
  renders ordinary Svelte nodes. It never constructs an HTML string.
- Tooltips render Svelte markup or assign `textContent`; they never use
  `innerHTML`.
- Invalid, overlapping, or out-of-range spans produce a safe plain-text fallback
  and a bounded diagnostic event without sentence content.
- SQL values use parameters. Dynamic SQL is limited to closed, code-owned
  fragments selected by enums.
- Production serving is same-origin and has no CORS middleware. Development
  origins are an explicit finite configuration and never combine `*` with
  credentials.
- There are no cookies, credentials, authentication state, or public writes.

## Operational Bounds

- Accept a syntactically valid `X-Request-ID` of at most 128 safe ASCII
  characters or generate a request ID.
- Structured logs include timestamp, level, request ID, route template, status,
  duration, database build ID, result count, corpus IDs, and ranking mode.
- Logs exclude raw query terms, corpus text, sentence text, SQL, and local paths.
- Search diagnostics may include only a code-point length bucket: `1`, `2–4`,
  `5–8`, `9–16`, `17–32`, or `33–64`. No query fingerprint or key-management
  mechanism is introduced without a demonstrated diagnostic need.
- Each DuckDB query has a hard 2-second execution deadline. The blocking query
  uses its request-local connection in one worker. The async handler schedules
  `loop.call_later(2.0, connection.interrupt)`, awaits the worker, cancels the
  timer handle in `finally`, and closes rather than reuses the connection on
  every outcome. The timer runs on the existing event loop; it adds no watchdog
  thread or join lifecycle. Timeout maps to the stable `query_timeout` service
  error.
- Every valid response must remain below 1 MiB when serialized.
- The initial capacity gate runs a curated search set at 10 concurrent clients;
  p95 end-to-end API latency must remain below 1 second on the documented
  production host class. The benchmark records host CPU, memory, DuckDB settings,
  dataset build ID, and query set so the number is reproducible.
- DuckDB memory and thread settings are explicit production configuration rather
  than machine-dependent defaults.
- One application-owned AnyIO capacity limiter admits at most 16 concurrent
DuckDB search/example workers. The handler uses non-blocking acquisition;
`WouldBlock` becomes `429 capacity_exceeded` with `Retry-After: 1`, and `finally`
releases the token. Rejecting rather than queueing prevents clients across many
source IPs from building an unbounded in-process wait queue whose requests have
already missed the latency budget. This is the global bound; Spec 6's edge
limits are per client.

## OpenAPI and TypeScript

Backend Pydantic models and FastAPI routes produce OpenAPI. A pinned
development-time generator writes committed TypeScript declarations to
`natsume-frontend/src/lib/api/schema.d.ts`. CI regenerates into a temporary path
and fails on a diff. The frontend uses a small handwritten fetch wrapper; no
runtime generated-client framework is added.

## Test Strategy

- Router integration tests use a real fixture DuckDB database.
- Contract tests validate every success and error shape.
- Boundary tests cover empty, maximum, over-maximum, Unicode, slash-containing,
  wildcard-containing, and invalid enum inputs.
- Ranking tests construct a collocation that is outside the global top N but
  inside a selected corpus's top N, and prove selection and `rankBy` happen
  before truncation. Aggregate mean tests include absent contributions as zero.
- A concurrency smoke test issues parallel requests and proves connections are
  not shared unsafely.
- Response-cardinality and serialized-size tests exercise maximum valid limits.
- Malicious corpus fixtures include tags, entities, quotes, event-handler-like
  text, and malformed spans; browser assertions prove no executable nodes appear.
- Readiness tests cover missing manifest, checksum mismatch, unsupported schema,
  unreadable database, and successful trivial query.

## Acceptance Criteria

- All routes and models above appear in OpenAPI and generated frontend types.
- Invalid `pos`/limits return the common 4xx envelope, never an uncaught
  `ValueError`/500.
- The server starts only with a compatible artifact and becomes unready if its
  startup validation fails.
- Parallel read tests pass with request-local connections.
- No corpus/API content reaches an HTML interpreter or structured logs.
- Every maximum valid request remains below the response-size cap and respects
  cardinality bounds.
- Corpus selection and ranking changes return the true selection-specific top N,
  totals, and examples rather than a refinement of a global page.
- Timeout tests interrupt a deliberately long query, close its connection, and
  prove the next request succeeds on a new connection. They also exercise
  completion just before the timer fires so cancellation cannot interrupt a
  later request.
- The request path in `server.py` remains readable top-to-bottom for the backend
  walkthrough: FastAPI dependency, validation, bounded query helper, response.
  No repository, pool, watchdog, or generic manager layer is introduced.
- The curated concurrency benchmark meets the documented p95 target or the
  service is not released with those resource settings.

## Rollout and Rollback

Backend and frontend deploy atomically. The new fixture schema allows protocol
development before the production builder exists. Production cutover waits for
Spec 3 to emit a compatible artifact. Rollback restores the previous complete
application plus its matching database artifact; mixed old/new contracts are
not supported.

## Decision Log

| Decision                                            | Status   | Reason                                                                                            | Revisit trigger                                              |
| --------------------------------------------------- | -------- | ------------------------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| No API version namespace                            | Accepted | No external consumers; atomic deployment                                                          | First external consumer                                      |
| Per-corpus rate plus equal-weight selected mean     | Accepted | Keeps the per-million denominator honest and avoids aggregate magnitude scaling with corpus count | Domain analysis prefers pooled corpus-size weighting         |
| Selection and ranking happen before limiting        | Accepted | A globally truncated response cannot produce correct selection-specific top N client-side         | Cursor pagination or unbounded result transfer is introduced |
| Limit to 200 items per particle                     | Accepted | Bounds public work without unused pagination                                                      | Present consumer needs deeper results                        |
| Request-local read-only connections                 | Accepted | Matches DuckDB Python concurrency guidance                                                        | Measured connection overhead becomes material                |
| No production CORS                                  | Accepted | Frontend and API are same-origin                                                                  | Separate trusted frontend origin is deployed                 |
| OpenAPI-generated compile-time types only           | Accepted | Prevents drift without runtime client machinery                                                   | Multiple clients need richer generation                      |
| Interrupt and discard timed-out request connections | Accepted | DuckDB exposes connection interruption but no declarative per-query timeout                       | Selected DuckDB release provides a safer native deadline     |
| Use the event-loop timer, not a watchdog thread      | Accepted | The async handler already owns scheduling; a second thread and join lifecycle add no guarantee    | Runtime evidence shows event-loop starvation delays interrupts |
| Fail fast at the global query bound                  | Accepted | Per-client edge limits do not bound aggregate clients; queueing would violate latency bounds       | Capacity measurements justify a queue or a different limit     |
| Log only query-length buckets                        | Accepted | No present diagnostic consumer justifies stable query correlation or keyed hashing                 | A concrete incident cannot be diagnosed from existing fields  |
