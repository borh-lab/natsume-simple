# Post-release hardening design

## Purpose

Turn the first three-corpus release review into measured performance evidence and
small behavior-preserving simplifications. The published corpus artifact remains
unchanged: this work neither rebuilds it nor adds corpus data to Git.

## Decisions

### Performance evidence

Add one repository-owned benchmark command that reports per-endpoint latency
distributions rather than one aggregate distribution. It accepts a base URL,
selected corpus IDs, concurrency, request count, and JSON output path. The fixed
request family is the six endpoints used by the two release records: suggestions,
four collocation ranking/direction combinations, and examples.

Run the harness against these cases on the named release host:

1. 2026-08-13 artifact with JNLP and Wikipedia;
2. 2026-08-14 artifact with JNLP and Wikipedia;
3. 2026-08-14 artifact with JNLP, TED, and Wikipedia;
4. each applicable case at concurrency 1 and 10.

The harness warms every URL, requires every measured response to be HTTP 200,
and emits request count, success count, response-size range, p50, p95, maximum,
wall time, and throughput for each endpoint and overall. Measurements are
diagnostic, not a Nix check: host scheduling makes a hard CI latency gate
misleading. The 2026-08-14 release record will state that its mixed aggregate
cannot establish super-linear growth, record the matched results, and set the
next investigation trigger at p95 at or above 600 ms for a fixed matched case,
any maximum above two seconds, or the next corpus addition.

Rejected alternatives:

- Optimizing the observed sequential scans now: no endpoint profile has named
  the dominant bound.
- Another one-off benchmark: it would preserve the current reproducibility gap.
- A permanent performance service: one public single-instance release has no
  present consumer for that machinery.

### Frontend simplification

`SearchControls` keeps `requestGeneration` as the owner of out-of-order response
invalidation. `open` becomes derived from reactive `focusedWithin`,
`suggestions`, `dismissedQuery`, and the current trimmed term. Event handlers
update causes only; they no longer synchronize a second mutable visibility
state. Existing delayed-response, focus, selection, submission, and Escape
browser tests are the characterization boundary.

`CollocationItem` owns the complete disclosure shell: `<details>`, summary,
frequency bar, and label. `SentenceExamples` owns only lazy example fetching and
the disclosure body. This removes the hollow wrapper and the `segments`,
`colors`, and `label` props from a component whose name promises examples.
Example rows use an unkeyed each block because each response is assigned once and
never reordered or spliced.

The spreadsheet browser helper asserts overflow and keyboard scrolling at both
375 px and 1280 px. The production layout intentionally overflows at both widths.

### Input hardening

Malformed TED document structure is an identity/format failure and aborts the
adapter. Tests cover nested starts, stray ends, text before a document,
unterminated documents, and metadata outside a document. Metadata outside a
document raises `ted_archive_structure_invalid`; it is not a bounded rejection.

A ready TED source-lock entry whose `servingCorpusId` is not `ted` raises the
specific `source_lock_serving_corpus_mismatch` reason. Builder type narrowing
uses explicit control flow or typing casts rather than runtime `assert`
statements that resemble validation.

### Explicitly deferred

- Per-corpus sentence-filter counts begin with the next corpus build. They cannot
  repair the existing artifact's aggregate observation without rebuilding it.
- Notice markers or `terms.json` are not added. A self-asserted marker would not
  prove the visible prose agrees with it; the release checker continues to test
  the user-visible statements directly.
- Query indexes, caches, and materialization changes wait for the matched
  per-endpoint benchmark and a profile that names the bound.

## Verification and lifecycle

Every behavior change lands test-first. The full `nix flake check` and frontend,
server, and CPU builder package builds must pass. The temporary spec and plan are
retired after their decisions are captured in the benchmark command, release
record, tests, and commit history.
