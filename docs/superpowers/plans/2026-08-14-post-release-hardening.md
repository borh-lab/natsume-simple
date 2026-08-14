# Post-release Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add reproducible matched latency evidence, simplify the protected frontend state/component seams, and harden malformed TED/source-lock inputs without rebuilding the corpus artifact.

**Architecture:** A standard-library benchmark module owns the fixed release request family and JSON summaries; release records consume its output but CI does not impose host-dependent latency thresholds. Frontend visibility becomes derived state and disclosure ownership moves to `CollocationItem`. Corpus adapters continue to abort on malformed identity/structure while bounded rejections remain document-denominated.

**Tech Stack:** Python 3.12 standard library, FastAPI service, DuckDB artifacts, Svelte 5 runes, Playwright, pytest, Nix flakes.

## Global Constraints

- Do not rebuild, rewrite, or commit either corpus artifact.
- Do not optimize a query until matched per-endpoint measurements and a profile name the dominant bound.
- Keep benchmark evidence diagnostic and host-labelled; do not add a flaky Nix latency threshold.
- Preserve autocomplete selection, submission, Escape, focus-within, and delayed-response behavior.
- Preserve the public API, artifact schema, source-content hash, and corpus identities.
- Retire this plan and its design spec after implementation evidence has permanent owners.

---

### Task 1: Reproducible service benchmark

**Files:**
- Create: `src/natsume_simple/benchmark_service.py`
- Create: `tests/test_benchmark_service.py`
- Modify: `README.md`

**Interfaces:**
- Produces: `build_request_family(base_url: str, corpus_ids: tuple[str, ...]) -> tuple[BenchmarkEndpoint, ...]`
- Produces: `run_benchmark(endpoints: tuple[BenchmarkEndpoint, ...], *, request_count: int, concurrency: int, timeout: float) -> dict[str, object]`
- Produces: `python -m natsume_simple.benchmark_service --base-url URL --corpus-id ID ... --requests 500 --concurrency 10 --output PATH`

- [ ] **Step 1: Write request-family and summary tests**

Add tests that assert the six endpoint names, repeated `corpusId` query values,
both rank modes, stable round-robin scheduling, linear p50/p95 calculation,
response-size range, and failure when any response is not HTTP 200. Use a local
`ThreadingHTTPServer`; do not mock time or HTTP internals.

- [ ] **Step 2: Run the focused test and verify red**

Run:

```bash
nix develop .#test --command pytest tests/test_benchmark_service.py -q
```

Expected: collection fails because `natsume_simple.benchmark_service` does not exist.

- [ ] **Step 3: Implement the minimal standard-library harness**

Use `urllib.parse.urlencode(..., doseq=True)`, `urllib.request.urlopen`,
`ThreadPoolExecutor(max_workers=concurrency)`, and `time.perf_counter`. Warm each
URL once before starting the measured wall clock. Keep individual observations
as endpoint name, elapsed milliseconds, status, and body bytes; reject invalid
arguments and any non-200 result. Summaries contain `requests`, `successes`,
`bodyBytesMin`, `bodyBytesMax`, `p50Ms`, `p95Ms`, `maxMs`, `wallSeconds`, and
`requestsPerSecond`, both overall and under `endpoints`.

- [ ] **Step 4: Verify focused tests and source quality**

Run:

```bash
nix develop .#test --command pytest tests/test_benchmark_service.py -q
nix develop .#test --command ruff format --check src/natsume_simple/benchmark_service.py tests/test_benchmark_service.py
nix develop .#test --command ruff check src/natsume_simple/benchmark_service.py tests/test_benchmark_service.py
nix develop .#test --command mypy src
```

Expected: all pass.

- [ ] **Step 5: Document the exact operator command**

Add a README release-validation example using `nix develop .#test --command
python -m natsume_simple.benchmark_service`, explicitly stating that results are
host-specific evidence and not a CI gate.

- [ ] **Step 6: Commit**

```bash
git add src/natsume_simple/benchmark_service.py tests/test_benchmark_service.py README.md
git commit -m "feat: add reproducible service benchmark"
```

### Task 2: Matched latency diagnosis

**Files:**
- Modify: `docs/releases/20260814T003410Z-c8a55484c9c15371c88417240b288754.md`
- Runtime-only output: `/tmp/natsume-benchmark-*.json`

**Interfaces:**
- Consumes: Task 1 benchmark CLI.
- Produces: matched, per-endpoint evidence comparing old/two, new/two, and new/three corpus cases at concurrency 1 and 10.

- [ ] **Step 1: Start direct servers for both immutable artifacts**

Within one shell command, start the packaged server for the 2026-08-13 artifact
on port 18010 and the 2026-08-14 artifact on port 18011. Trap EXIT to terminate
both exact child PIDs. Wait for readiness and assert the expected build ID at
each port before sending benchmark traffic.

- [ ] **Step 2: Run balanced matched cases**

Run two balanced blocks in `old2, new2, new3, new3, new2, old2` order at
concurrency 1, then the same order at concurrency 10. Every run uses 500
round-robin requests, the same six endpoint definitions, and a distinct JSON
file under `/tmp`. Keep the machine otherwise unchanged.

- [ ] **Step 3: Compare per-endpoint distributions**

Report medians across the four runs per applicable case and concurrency. Do not
call growth super-linear from two aggregate points. Name whether data growth,
the third selected corpus, or concurrency/queueing accounts for the largest
measured delta. If attribution remains unclear, say so and stop before query
changes.

- [ ] **Step 4: Update the release record**

Record the harness command, matched table, the prior 1,092.335 ms maximum, and
the trigger: investigate when a fixed matched case reaches p95 >= 600 ms, any
maximum exceeds 2,000 ms, or before adding another corpus. State explicitly
that the original mixed benchmark remains a valid release pass but was not a
complexity measurement.

- [ ] **Step 5: Commit evidence only**

```bash
git add docs/releases/20260814T003410Z-c8a55484c9c15371c88417240b288754.md
git commit -m "docs: record matched service latency"
```

### Task 3: Derive autocomplete visibility

**Files:**
- Modify: `natsume-frontend/src/lib/components/SearchControls.svelte`
- Modify: `natsume-frontend/tests/test.ts`

**Interfaces:**
- Preserves: the existing `SearchControls` prop contract and DOM accessibility contract.
- Produces: one `$derived` owner for autocomplete visibility; event handlers update only its causes.

- [ ] **Step 1: Strengthen characterization for focus within the widget**

Extend the existing autocomplete browser test to prove that focusing the search
direction control while populated suggestions exist follows the documented
focus-within rule, and that submission keeps the popup dismissed after a
pending response. Keep the existing delayed Escape and suggestion-selection
coverage.

- [ ] **Step 2: Run the focused Playwright tests before refactoring**

Run the autocomplete tests with the fixture server through the frontend Nix
shell. Record whether the new focus-within characterization exposes the current
missing transition; if it does, treat that behavior correction separately in
the same frontend commit because the approved contract already requires it.

- [ ] **Step 3: Replace mutable `open` state**

Make `dismissedQuery` reactive and define:

```ts
let open = $derived(
  focusedWithin && suggestions.length > 0 && dismissedQuery !== term.trim()
);
```

Remove all assignments to `open`. Keep request-generation invalidation and
active-option reset behavior. `focusout`, input, choose, submit, and Escape
update only `focusedWithin`, `dismissedQuery`, `requestGeneration`,
`suggestions`, or `active`.

- [ ] **Step 4: Verify frontend behavior**

Run:

```bash
nix build .#checks.x86_64-linux.playwright
nix build .#checks.x86_64-linux.frontend
```

Expected: all browser, lint, type, unit, and build checks pass.

### Task 4: Restore disclosure ownership

**Files:**
- Modify: `natsume-frontend/src/lib/components/CollocationItem.svelte`
- Modify: `natsume-frontend/src/lib/components/SentenceExamples.svelte`
- Modify: `natsume-frontend/tests/test.ts`

**Interfaces:**
- `SentenceExamples` consumes only `client`, `item`, `selectedCorpusIds`, and
  `expanded`.
- `CollocationItem` owns `<details>`, `<summary>`, stacked bar, label, and toggle-triggered lazy loading boundary.

- [ ] **Step 1: Make the existing layout assertions structural**

Extend the example-expansion browser test to assert that the frequency bar and
label are inside the collocation summary while the examples body remains full
disclosure width. Change `expectSpreadsheet` to assert overflow and keyboard
scrolling at both 375 px and 1280 px; remove the misleading boolean parameter.

- [ ] **Step 2: Run Playwright before the refactor**

Run the focused search and spreadsheet tests and confirm the characterization
passes against the current DOM.

- [ ] **Step 3: Move the disclosure shell**

Move `<details>`, its toggle handler, `<summary>`, SVG bar, and label to
`CollocationItem`. The parent stores the current disclosure state in an
`expanded` boolean and passes it to `SentenceExamples`; the child reacts to the
first `true` value by calling its existing idempotent `load()` function. Remove
`segments`, `colors`, and `label` props from `SentenceExamples` while keeping
request and status state private there.

Render examples with an unkeyed `{#each examples as example}`. Keep keyed
sentence segments because their generated sequence is stable within a row.

- [ ] **Step 4: Verify and commit both frontend simplifications**

Run:

```bash
nix build .#checks.x86_64-linux.playwright
nix build .#checks.x86_64-linux.frontend
```

Then commit Tasks 3 and 4 together because both are one behavior-preserving UI
ownership cleanup protected by the same browser contract:

```bash
git add natsume-frontend/src/lib/components/SearchControls.svelte natsume-frontend/src/lib/components/CollocationItem.svelte natsume-frontend/src/lib/components/SentenceExamples.svelte natsume-frontend/tests/test.ts
git commit -m "refactor: simplify search result state ownership"
```

### Task 5: Harden TED and source-lock structure

**Files:**
- Modify: `tests/test_corpus_adapters.py`
- Modify: `src/natsume_simple/corpus_pipeline.py`
- Modify: `tests/test_release_inputs.py`
- Modify: `src/natsume_simple/release_inputs.py`
- Modify: `src/natsume_simple/builder_cli.py`

**Interfaces:**
- Preserves: `adapt_ted_iwslt_archive(Path) -> AdaptationResult`.
- Adds error reason: `source_lock_serving_corpus_mismatch`.

- [ ] **Step 1: Write malformed TED structure tests**

Parameterize five inputs: nested `<doc>`, stray `</doc>`, text before `<doc>`,
EOF before `</doc>`, and `<title>` before `<doc>`. Every case must raise exactly
`ted_archive_structure_invalid`.

- [ ] **Step 2: Verify the metadata-outside-document case fails**

Run:

```bash
nix develop .#test --command pytest tests/test_corpus_adapters.py -k ted_archive_structure -q
```

Expected: the metadata-before-document case does not raise on current code.

- [ ] **Step 3: Reject metadata outside a document**

Replace the silent `continue` with `raise
ValueError("ted_archive_structure_invalid")`. Keep all malformed-structure
raise sites on that single diagnostic reason.

- [ ] **Step 4: Write and implement the source-lock diagnostic test**

Mutate the ready TED fixture entry to `servingCorpusId: "other"` and require
`ReleaseInputError("source_lock_serving_corpus_mismatch")`. Validate this field
outside the missing-entry `try` block so an existing but misconfigured entry is
not reported as absent.

- [ ] **Step 5: Replace type-narrowing assertions**

In `_build`, bind `source_lock` and `release_sources` through explicit guarded
locals before the TED/Wikipedia branches. Preserve the existing user-facing
`Wikipedia requires ...` and `TED requires ...` errors; do not introduce a new
reachable state or load heavy modules earlier.

- [ ] **Step 6: Verify and commit**

Run:

```bash
nix develop .#test --command pytest tests/test_corpus_adapters.py tests/test_release_inputs.py tests/test_builder_cli.py -q
nix develop .#test --command ruff format --check src tests
nix develop .#test --command ruff check src tests
nix develop .#test --command mypy src
```

Then:

```bash
git add tests/test_corpus_adapters.py src/natsume_simple/corpus_pipeline.py tests/test_release_inputs.py src/natsume_simple/release_inputs.py src/natsume_simple/builder_cli.py
git commit -m "fix: reject malformed release inputs"
```

### Task 6: Final verification and scaffolding retirement

**Files:**
- Delete: `docs/superpowers/specs/2026-08-14-post-release-hardening-design.md`
- Delete: `docs/superpowers/plans/2026-08-14-post-release-hardening.md`

**Interfaces:**
- Produces: a clean branch containing only durable code, tests, README guidance, release evidence, and commit history.

- [ ] **Step 1: Run the complete gate**

```bash
nix flake check --print-build-logs
nix build .#frontend .#server .#corpus-builder-cpu
```

Expected: every check and package build succeeds.

- [ ] **Step 2: Re-run the real release structural check**

```bash
nix run .#build-corpus -- release-check /home/bor/Projects/natsume-simple/artifacts/20260814T003410Z-c8a55484c9c15371c88417240b288754 --source-lock docs/corpus-sources.lock.json --wikipedia-subset docs/wikipedia-ja-20231101-subset.json
```

Expected: three corpus IDs, 3,862 sources, and 971 Wikipedia sources.

- [ ] **Step 3: Audit repository boundaries**

Require clean `git diff --check`, no tracked database/archive/Parquet files, no
tracked file above 50 MiB, and only the owner's pre-existing root untracked files
outside the worktree.

- [ ] **Step 4: Retire implementation scaffolding and commit**

```bash
git rm docs/superpowers/specs/2026-08-14-post-release-hardening-design.md docs/superpowers/plans/2026-08-14-post-release-hardening.md
git commit -m "docs: retire post-release hardening scaffolding"
```

- [ ] **Step 5: Re-run cheap final-state checks**

Run `git diff --check`, `git status --short`, the real release check, and the
cached `nix flake check` once more after the deletion commit.
