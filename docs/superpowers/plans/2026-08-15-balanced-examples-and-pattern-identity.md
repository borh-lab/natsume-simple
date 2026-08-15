# Balanced Examples and Pattern Identity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Interleave examples evenly across selected corpora while preserving stable offset pagination, and replace ambiguous result-mode labels with explicit noun–particle–verb patterns.

**Architecture:** The examples endpoint remains the sole owner of eligibility, total ordering, pagination, and `hasMore`; a DuckDB window function changes only its final order. `SearchSummary.svelte` remains the owner of accepted-result identity and renders a compact visible pattern plus a semantic accessible label. No request, response, schema, controller, or client interface changes.

**Tech Stack:** Python 3.12, FastAPI, DuckDB, Pydantic, pytest, Svelte 5, TypeScript 6, Playwright, Nix flakes.

## Global Constraints

- Balance occurrence presentation in canonical corpus-ID order; do not change counts, deduplicate occurrences, or alter statistical ranking.
- Use `ROW_NUMBER` partitioned by corpus with the existing source/sentence/span/extractor total order.
- Keep `limit`, `offset`, `hasMore`, selected-corpus identity, timeout, and error envelopes unchanged.
- When a corpus exhausts, remaining corpora fill subsequent result positions.
- Visible noun identity is `N matches · “term”–particle–verb`; visible verb identity is `N matches · noun–particle–“term”`.
- Use singular `match` for exactly one result and plural `matches` otherwise.
- Bold only the searched term; keep grammatical placeholders neutral.
- Give the identity a semantic accessible label rather than relying on punctuation pronunciation.
- Add no schema, index, cache, cursor, API field, dependency, store, controller method, ordering module, or pagination abstraction.
- Preserve owner files `AGENDA.md` and `container.nix`; never stage or modify them.

---

### Task 1: Balanced, stable example pagination

**Files:**
- Modify: `tests/test_api_contract.py:562-625`
- Modify: `src/natsume_simple/api.py:577-640`

**Interfaces:**
- Consumes: existing `GET /api/examples` parameters `noun`, `particle`, `verb`, repeated `corpusId`, `limit`, and `offset`.
- Produces: the unchanged `ExamplesResponse`, ordered by `(corpus_row, corpus_id)` where `corpus_row` is deterministic within each corpus.

- [ ] **Step 1: Add a test-local three-corpus artifact helper**

In `tests/test_api_contract.py`, add a private helper beside `fixture_client` that calls `build_search_artifact`, opens its DuckDB file, and inserts:

```python
INSERT INTO corpus VALUES ('gamma', 'Gamma');
INSERT INTO source (id, corpus_id, external_id, title, content_sha256)
VALUES (4, 'gamma', 'g1', 'Gamma one', 'sha-g1');
INSERT INTO sentence VALUES
    (17, 4, 1, '情報を集める。'),
    (18, 4, 2, '情報を集める。');
INSERT INTO collocation_occurrence VALUES
    (17, '情報', 'を', '集める', 0, 2, 2, 3, 3, 6, 'fixture-extractor'),
    (18, '情報', 'を', '集める', 0, 2, 2, 3, 3, 6, 'fixture-extractor');
INSERT INTO corpus_stats VALUES ('gamma', 1, 2, 2);
```

After closing DuckDB, recompute `databaseSha256` in `manifest.json`. Return `TestClient(create_app(artifact_dir))`. Keep this helper private to the API contract file; do not modify the shared browser fixture.

- [ ] **Step 2: Write `test_balanced_examples_round_robin_and_fill_exhausted`**

Request all corpora for `情報–を–集める` with `limit=5`. Assert corpus IDs are:

```python
["alpha", "beta", "gamma", "alpha", "gamma"]
```

and `hasMore is True`. Request `offset=5, limit=5` and assert the remaining corpus IDs are `['alpha']` with `hasMore is False`. This proves one-result `beta` and two-result `gamma` drop out without reserving empty slots.

- [ ] **Step 3: Write `test_example_pages_compose_and_repeat`**

Fetch offsets `0`, `2`, and `4` at `limit=2`, concatenate occurrence identities `(corpusId, sourceId, sentenceId, nounSpan, particleSpan, verbSpan)`, and compare them with one `limit=20` request. Repeat the full request and assert exact response equality. Assert all six occurrence identities are unique.

- [ ] **Step 4: Run the focused API tests and verify RED**

Run:

```bash
nix develop .#test --command pytest tests/test_api_contract.py -k 'balanced_examples or example_pages_compose' -vv
```

Expected: the first page is corpus-first (`alpha`, `alpha`, `alpha`, `beta`, `gamma`) rather than round-robin.

- [ ] **Step 5: Implement the windowed order directly in the endpoint query**

Replace the examples SELECT with one CTE:

```sql
WITH ranked_examples AS (
    SELECT src.corpus_id,
           src.id AS source_id,
           src.title AS source_title,
           s.id AS sentence_id,
           s.text,
           o.n_begin, o.n_end,
           o.p_begin, o.p_end,
           o.v_begin, o.v_end,
           ROW_NUMBER() OVER (
               PARTITION BY src.corpus_id
               ORDER BY src.id, s.id,
                        o.n_begin, o.n_end,
                        o.p_begin, o.p_end,
                        o.v_begin, o.v_end,
                        o.extractor_id
           ) AS corpus_row
    FROM collocation_occurrence o
    JOIN sentence s ON s.id = o.sentence_id
    JOIN source src ON src.id = s.source_id
    WHERE o.noun = ? AND o.particle = ? AND o.verb = ?
      AND src.corpus_id IN ({placeholders})
)
SELECT corpus_id, source_id, source_title, sentence_id, text,
       n_begin, n_end, p_begin, p_end, v_begin, v_end
FROM ranked_examples
ORDER BY corpus_row, corpus_id
LIMIT ? OFFSET ?
```

Keep parameter order, `limit + 1`, row-to-response mapping, query limiter, timeout, logging, and `hasMore` calculation unchanged.

- [ ] **Step 6: Run the focused and existing examples contract tests GREEN**

Run:

```bash
nix develop .#test --command pytest tests/test_api_contract.py -k 'example' -vv
```

Expected: balancing, single-corpus total order, offset bounds, exhausted pages, response size, and typed-span tests all pass.

- [ ] **Step 7: Commit Task 1**

```bash
git add src/natsume_simple/api.py tests/test_api_contract.py
git commit -m "feat: balance examples across corpora"
```

### Task 2: Explicit grammatical-pattern result identity

**Files:**
- Modify: `natsume-frontend/src/lib/components/SearchSummary.svelte`
- Modify: `natsume-frontend/tests/test.ts:623-687`

**Interfaces:**
- Consumes: existing `VisibleSearchResult.input.term`, `.pos`, response particle groups, `stale`, and `draftDiffers`.
- Produces: compact visible pattern markup and a semantic `aria-label`; component props remain unchanged.

- [ ] **Step 1: Write failing noun-pattern assertions**

After accepting noun search `情報`, assert the summary contains the exact visible text `10 matches · “情報”–particle–verb`, its `<strong>` contains exactly `“情報”`, and its accessible label is `10 matches; searched noun “情報”; pattern noun, particle, verb`.

- [ ] **Step 2: Write failing verb-pattern and draft-stability assertions**

Select verb mode, search `集める`, accept it, and assert visible `1 match · noun–particle–“集める”`, bold `“集める”`, and accessible label `1 match; searched verb “集める”; pattern noun, particle, verb`. Then edit the term and switch mode without submitting; assert the accepted verb pattern remains visible beside `Not applied`.

- [ ] **Step 3: Run the focused browser tests and verify RED**

Run:

```bash
NATSUME_TEST_API_PORT=8011 NATSUME_TEST_FRONTEND_PORT=4184 nix develop .#frontend --command bash -lc 'cd natsume-frontend && npm run test:integration -- --grep "grammatical pattern|displayed results tied"'
```

Expected: current summaries end in the bare labels `Noun` or `Verb` and contain no semantic pattern label.

- [ ] **Step 4: Implement explicit pattern values and markup**

In `SearchSummary.svelte`, derive:

```ts
const countLabel = $derived(
  `${count.toLocaleString()} ${count === 1 ? 'match' : 'matches'}`
);
const pattern = $derived(
  result?.input.pos === 'verb'
    ? `noun–particle–“${result.input.term}”`
    : `“${result.input.term}”–particle–verb`
);
const accessibleIdentity = $derived(
  result
    ? `${countLabel}; searched ${result.input.pos} “${result.input.term}”; pattern noun, particle, verb`
    : ''
);
```

Render the visible identity as count text plus conditional markup so only the quoted term is inside `<strong>`. Put the compact full string in `title` and `accessibleIdentity` in `aria-label`. Preserve truncation, `aria-live`, `Not applied`, `Previous result`, and all neutral styling.

- [ ] **Step 5: Migrate existing summary selectors**

Replace every browser expectation ending in `· Noun` or `· Verb` with the matching visible pattern. Keep draft-stability assertions tied to the accepted input and do not derive expectations from the current draft controls.

- [ ] **Step 6: Run focused browser tests and Svelte diagnostics GREEN**

Run:

```bash
NATSUME_TEST_API_PORT=8011 NATSUME_TEST_FRONTEND_PORT=4184 nix develop .#frontend --command bash -lc 'cd natsume-frontend && npm run test:integration -- --grep "grammatical pattern|displayed results tied|responsive results toolbar" && npm run check'
```

Expected: noun, verb, accepted/draft identity, toolbar geometry, and accessibility assertions pass with zero Svelte warnings.

- [ ] **Step 7: Commit Task 2**

```bash
git add natsume-frontend/src/lib/components/SearchSummary.svelte natsume-frontend/tests/test.ts
git commit -m "feat: show grammatical search patterns"
```

### Task 3: Full verification and decision-record lifecycle

**Files:**
- Modify: `docs/superpowers/specs/2026-08-15-balanced-examples-and-pattern-identity-design.md`
- Delete after verification: `docs/superpowers/plans/2026-08-15-balanced-examples-and-pattern-identity.md`

**Interfaces:**
- Consumes: Tasks 1 and 2.
- Produces: an implemented retained design record and a clean, fully gated branch.

- [ ] **Step 1: Run Python quality and test gates**

Run:

```bash
nix build .#checks.x86_64-linux.source-quality .#checks.x86_64-linux.backend
```

Expected: formatting, Ruff, mypy, doctests, model-free tests, and API contracts pass.

- [ ] **Step 2: Run frontend and browser gates**

Run:

```bash
nix build .#checks.x86_64-linux.frontend .#checks.x86_64-linux.playwright .#checks.x86_64-linux.package-frontend .#checks.x86_64-linux.server-smoke
```

Expected: frontend lint/type/unit/build, all Playwright scenarios, packaged frontend, and server smoke pass.

- [ ] **Step 3: Confirm minimum-sufficient scope**

Run:

```bash
git diff --name-only main...HEAD
git diff --check main...HEAD
git ls-tree -rl HEAD | sort -k4 -n | tail -5
```

Confirm only the endpoint SQL, API contract test, summary component, browser tests, and decision record changed; no database, corpus archive, model, generated frontend build, dependency, schema, API type, controller, or client file entered the diff.

- [ ] **Step 4: Mark the design implemented and retire this plan**

Change the design status to `Implemented`, add one evidence sentence naming the passing API, browser, and Nix gates, then delete this execution plan. Do not copy task steps into the retained design record.

- [ ] **Step 5: Commit lifecycle documentation**

```bash
git add docs/superpowers/specs/2026-08-15-balanced-examples-and-pattern-identity-design.md
git add -u docs/superpowers/plans/2026-08-15-balanced-examples-and-pattern-identity.md
git commit -m "docs: record balanced example implementation"
```

- [ ] **Step 6: Review final branch state**

Run:

```bash
git status --short
git log --oneline main..HEAD
```

Expected: only owner files `AGENDA.md` and `container.nix` are untracked in the main checkout; the feature worktree is clean and contains focused implementation commits.
