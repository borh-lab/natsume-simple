# Incremental Results Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let users append independently paged collocations in one particle column and append additional examples in one disclosure without mixing search or artifact identities.

**Architecture:** Extend the existing read-only endpoints with deterministic offsets. The search controller continues to own the submitted query identity; each particle column and example disclosure owns only its local request/accumulation state, guarded by a shared pure build/corpus identity predicate. No generic pagination store, cursor, cache, or virtualization layer is introduced.

**Tech Stack:** Python 3.12, FastAPI, DuckDB, Pydantic, Svelte 5 runes, TypeScript 6, Vitest, Playwright, Nix flakes.

## Global Constraints

- Collocation ordering remains descending mean frequency per million, descending raw frequency, then noun/particle/verb lexical order.
- Example ordering must be total: corpus, source, sentence, all six span bounds, then extractor ID.
- Collocation page size is `min(200, floor(450 / selectedCorpusIds.length))`.
- Initial examples request five rows; later example requests use the existing maximum of 20.
- Responses from a different `databaseBuildId` or canonical corpus selection are never appended.
- Existing ranking, corpus distributions, highlighting, bar scales, raw-frequency tooltips, keyboard scrolling, dark mode, and response-size limits remain protected.
- Use Nix development shells and flake checks; do not create a checkout-local virtual environment.
- Preserve untracked owner files such as `AGENDA.md` and `container.nix`.

---

## File Structure

- `src/natsume_simple/api.py`: owns public parameter validation, deterministic database ordering, paging slices, and response envelopes.
- `tests/test_api_contract.py`: owns backend paging, ordering, bounds, totals, and response-size contracts.
- `natsume-frontend/src/lib/api/types.ts`: owns wire types, including `Particle` and `ExamplesResponse.hasMore`.
- `natsume-frontend/src/lib/api/client.ts`: serializes the new paging parameters.
- `natsume-frontend/src/lib/search/controller.svelte.ts`: owns full-search page-size policy and shared result-identity comparison.
- `ParticleOverview.svelte` and `ParticleColumn.svelte`: key/reset one visible result and own per-particle accumulation.
- `CollocationItem.svelte` and `SentenceExamples.svelte`: own disclosure presentation and per-collocation accumulation.
- `SearchSummary.svelte`: reports total matches, not only the loaded prefix.
- `src/natsume_simple/benchmark_service.py`: keeps later-page requests in the release performance family and rejects empty later-page measurements.
- `README.md`: documents the public API contract after implementation.
- `docs/superpowers/specs/2026-08-14-incremental-results-design.md`: remains as the durable decision and measurement record; only this task plan is retired.

### Task 1: Collocation Offset Contract

**Files:**
- Modify: `src/natsume_simple/api.py:28-120,426-549`
- Test: `tests/test_api_contract.py:300-470`

**Interfaces:**
- Produces: optional query parameters `particle: Particle | None` and `offsetPerParticle: int = 0` on `GET /api/collocations`.
- Preserves: `CollocationsResponse` and `ParticleGroupResponse` wire shapes.
- Rule: `offsetPerParticle > 0` without `particle` returns `400 invalid_parameter`.

- [ ] **Step 1: Write failing targeted-page contract tests**

Add tests that use the existing two `情報を…` rows in corpus `beta`:

```python
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
```

Add a second test for `offsetPerParticle=2`: the `を` group remains present with the full total/distribution, `returnedCount == 0`, and `items == []`. Add parameter tests proving an unknown particle is `422` and a positive offset without a particle returns the common `400 invalid_parameter` envelope. Keep the existing capacity cases as a cross-layer contract: one and two corpora accept 200, three corpora accept 150, and three corpora reject 151.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
nix develop .#test --command pytest tests/test_api_contract.py \
  -k 'collocations_page or collocation_offset' -q
```

Expected: failures because the endpoint ignores or rejects the new parameters.

- [ ] **Step 3: Implement particle filtering and offset slicing**

In `api.py`, name both service-owned capacity values and define the particle vocabulary once:

```python
MAX_COLLOCATIONS_PER_PARTICLE = 200
COLLOCATION_ITEM_CORPUS_BUDGET = 450
Particle = Literal["が", "を", "に", "で", "から", "より", "と", "へ"]
PARTICLES: tuple[Particle, ...] = ("が", "を", "に", "で", "から", "より", "と", "へ")
```

Extend the handler signature:

```python
particle: Annotated[Particle | None, Query()] = None,
offsetPerParticle: Annotated[int, Query(ge=0)] = 0,
limitPerParticle: Annotated[
    int, Query(ge=1, le=MAX_COLLOCATIONS_PER_PARTICLE)
] = 100,
```

Before querying, raise:

```python
if offsetPerParticle and particle is None:
    raise PublicApiError(
        400,
        "invalid_parameter",
        "offsetPerParticle requires particle",
    )
```

When `particle` is present, append `AND particle = ?` to the `collocation_frequency` query and its value to the bound parameters. Iterate `PARTICLES` for the initial request and `(particle,)` for a targeted request. Keep full aggregation, count, and distribution calculation; change only the returned slice:

```python
returned_items = items[
    offsetPerParticle : offsetPerParticle + limitPerParticle
]
```

Do not skip a targeted group merely because its page slice is empty; skip it only when the full `items` list is empty.

Keep `request.state.result_count` equal to the sum of the page-level `returnedCount` values. A targeted request therefore logs only rows serialized in that page, matching the example endpoint's logging rule.

- [ ] **Step 4: Run the focused and complete API contract tests**

Run:

```bash
nix develop .#test --command pytest tests/test_api_contract.py -q
```

Expected: all API contract tests pass, including existing 200×2 and 150×3 response-size cases.

- [ ] **Step 5: Commit the backend collocation contract**

```bash
git add src/natsume_simple/api.py tests/test_api_contract.py
git commit -m "feat: page collocations by particle"
```

### Task 2: Stable Example Pages

**Files:**
- Modify: `src/natsume_simple/api.py:100-120,552-602`
- Test: `tests/test_api_contract.py:470-530`

**Interfaces:**
- Produces: `GET /api/examples?...&offset=N&limit=L` and `ExamplesResponse.hasMore: bool`.
- Ordering: corpus/source/sentence, noun span, particle span, verb span, extractor ID.
- Past-end contract: `examples == []` and `hasMore is False`.

- [ ] **Step 1: Write failing example-page tests around the existing same-sentence tie**

The fixture already contains two `情報を集める` occurrences in sentence 1 with different spans. Add:

```python
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
    assert [page["hasMore"] for page in pages] == [True, True, False, False, False]
    assert pages[3]["examples"] == []
    assert pages[4]["examples"] == []
```

The third occurrence is sentence 2 and proves sentence ordering remains ahead of spans. Update every existing exact `ExamplesResponse` assertion whose result count is below its requested limit to include `"hasMore": False`. Add `offset=-1` validation expecting `422`.

- [ ] **Step 2: Run focused example tests and verify RED**

```bash
nix develop .#test --command pytest tests/test_api_contract.py -k examples -q
```

Expected: failures for missing `offset`, partial tie ordering, and missing `hasMore`.

- [ ] **Step 3: Implement limit-plus-one paging with the total order**

Add `hasMore: bool` to `ExamplesResponse`, add:

```python
offset: Annotated[int, Query(ge=0)] = 0,
```

and change the SQL suffix to:

```sql
ORDER BY src.corpus_id, src.id, s.id,
         o.n_begin, o.n_end,
         o.p_begin, o.p_end,
         o.v_begin, o.v_end,
         o.extractor_id
LIMIT ? OFFSET ?
```

Bind `limit + 1` and `offset`. After the bounded query:

```python
has_more = len(rows) > limit
rows = rows[:limit]
```

Serialize only `rows`, set `hasMore=has_more`, and keep request logging's result count equal to the serialized page length.

- [ ] **Step 4: Run API and maximum-response tests**

```bash
nix develop .#test --command pytest tests/test_api_contract.py -q
```

Expected: all pass; the maximum 20-example serialized response remains below one mebibyte.

- [ ] **Step 5: Commit stable example pages**

```bash
git add src/natsume_simple/api.py tests/test_api_contract.py
git commit -m "feat: page examples in stable order"
```

### Task 3: Frontend Paging Protocol and Result Identity

**Files:**
- Modify: `natsume-frontend/src/lib/api/types.ts`
- Modify: `natsume-frontend/src/lib/api/client.ts`
- Test: `natsume-frontend/src/lib/api/client.test.ts`
- Modify: `natsume-frontend/src/lib/search/controller.svelte.ts`
- Test: `natsume-frontend/src/lib/search/controller.test.ts`

**Interfaces:**
- Produces: `Particle` wire type; optional `particle`/`offsetPerParticle` collocation arguments; optional example `offset`; `ExamplesResponse.hasMore`.
- Produces: `collocationPageSize(corpusCount: number): number`.
- Produces: `responseMatchesResult(response, databaseBuildId, corpusIds): boolean`.

- [ ] **Step 1: Write failing client serialization and policy tests**

Extend `client.test.ts` to assert a targeted request serializes:

```typescript
await client.getCollocations({
	term: '情報',
	pos: 'noun',
	corpusIds: ['alpha', 'beta', 'ted'],
	particle: 'を',
	offsetPerParticle: 150,
	limitPerParticle: 150
});
expect(url.searchParams.get('particle')).toBe('を');
expect(url.searchParams.get('offsetPerParticle')).toBe('150');
```

Add an example request assertion for `offset=5` and `limit=20`.

In `controller.test.ts`, add:

```typescript
expect(collocationPageSize(1)).toBe(200);
expect(collocationPageSize(2)).toBe(200);
expect(collocationPageSize(3)).toBe(150);
```

The controller cannot submit an empty selection: `submit()` returns early and `toggleCorpus()` refuses to deselect the last corpus, so do not add an unreachable zero-corpus branch to this helper. Assert `submit()` sends 150 for three selected corpora and 200 after selecting two. Add identity predicate tests for matching build/corpus order, changed build, changed membership, and changed canonical order.

- [ ] **Step 2: Run unit tests and verify RED**

```bash
nix develop .#frontend --command bash -lc \
  'cd natsume-frontend && npm run test:unit -- --run src/lib/api/client.test.ts src/lib/search/controller.test.ts'
```

Expected: missing types, parameters, and helper exports fail.

- [ ] **Step 3: Implement the wire types and client parameters**

In `types.ts` add:

```typescript
export type Particle = 'が' | 'を' | 'に' | 'で' | 'から' | 'より' | 'と' | 'へ';
```

Use it for `ParticleGroup.particle`. Add `hasMore: boolean` to `ExamplesResponse`.

Extend `ApiClient.getCollocations` arguments with `particle?: Particle` and `offsetPerParticle?: number`, appending both only when provided (`offsetPerParticle` must serialize when it is zero if the caller supplied it). Extend `getExamples` with `offset?: number` and serialize `offset ?? 0`.

- [ ] **Step 4: Implement page-size and identity primitives in the controller module**

Export:

```typescript
export const MAX_COLLOCATIONS_PER_PARTICLE = 200;
export const COLLOCATION_ITEM_CORPUS_BUDGET = 450;

export function collocationPageSize(corpusCount: number): number {
	return Math.min(
		MAX_COLLOCATIONS_PER_PARTICLE,
		Math.floor(COLLOCATION_ITEM_CORPUS_BUDGET / corpusCount)
	);
}

export function responseMatchesResult(
	response: { databaseBuildId: string; selectedCorpusIds: readonly string[] },
	databaseBuildId: string,
	corpusIds: readonly string[]
): boolean {
	return response.databaseBuildId === databaseBuildId && sameValues(response.selectedCorpusIds, corpusIds);
}
```

Extend `SearchApi.getCollocations`' structural argument type for the optional paging fields. In `submit()`, pass `limitPerParticle: collocationPageSize(input.corpusIds.length)`. The TypeScript unit expectations and Task 1's API capacity tests deliberately pin the same public 200/450 contract on both sides of the language boundary. Do not add page accumulation to the controller.

- [ ] **Step 5: Run frontend unit tests and type checking**

```bash
nix develop .#frontend --command bash -lc \
  'cd natsume-frontend && npm run test:unit -- --run && npm run check'
```

Expected: all unit tests pass and Svelte reports zero errors/warnings.

- [ ] **Step 6: Commit the frontend protocol**

```bash
git add natsume-frontend/src/lib/api natsume-frontend/src/lib/search
git commit -m "feat: add incremental result protocol"
```

### Task 4: Per-Particle Accumulation

**Files:**
- Modify: `natsume-frontend/src/routes/+page.svelte:61-68`
- Modify: `natsume-frontend/src/lib/components/ParticleOverview.svelte`
- Modify: `natsume-frontend/src/lib/components/ParticleColumn.svelte`
- Modify: `natsume-frontend/src/lib/components/SearchSummary.svelte`
- Test: `natsume-frontend/tests/test.ts`

**Interfaces:**
- Consumes: `VisibleSearchResult`, `collocationPageSize`, and `responseMatchesResult` from Task 3.
- Produces: independently appended items and local `idle | loading | request-error | identity-error` state per particle.

- [ ] **Step 1: Write a failing browser test for one-column accumulation**

Add a Playwright route that delegates the initial collocation request with `route.fetch()`, changes one returned group's `totalMatchingCollocations` to `items.length + 1`, records the sum of every group's resulting `totalMatchingCollocations`, and returns a synthetic targeted second page only when `particle=を` and `offsetPerParticle` equals the initial item count. The synthetic item must have a lower score than the initial last item.

Assert:

```typescript
await expect(
	page.getByText(`${expectedTotal} matching results for “情報” · Noun–particle search`, {
		exact: true
	})
).toBeVisible();
await expect(woColumn.getByText(/Showing \d+ of \d+/)).toBeVisible();
const otherCount = await gaColumn.locator('summary').count();
await woColumn.getByRole('button', { name: /Load 1 more/ }).click();
await expect(woColumn.getByText('追加する', { exact: true })).toBeVisible();
expect(await gaColumn.locator('summary').count()).toBe(otherCount);
await expect(
	page.getByText(`${expectedTotal} matching results for “情報” · Noun–particle search`, {
		exact: true
	})
).toBeVisible();
await expect(page.getByRole('combobox', { name: 'Bar scale' })).toHaveValue('particle');
```

Also record the particle overview's `scrollLeft` before the click and assert it is unchanged afterward.

- [ ] **Step 2: Run the browser test and verify RED**

Use the hermetic check so Chromium libraries and fixture services are owned by Nix:

```bash
nix build --no-link -L .#checks.x86_64-linux.playwright
```

Expected: the new load-more button cannot be found.

- [ ] **Step 3: Key the overview by visible-result identity**

In `+page.svelte`, replace the current block with:

```svelte
{#if controller.result}
	{#key controller.result}
		<ParticleOverview {client} result={controller.result} corpora={controller.corpora} />
	{/key}
{/if}
```

Change `ParticleOverview` to accept `VisibleSearchResult`, derive response and position from it, and pass `searchInput`, `databaseBuildId`, and canonical corpus IDs to each column. This remounts local pages on a successful full search but not when controls merely become dirty.

- [ ] **Step 4: Implement local column paging**

In `ParticleColumn`, initialize:

```typescript
let items = $state([...group.items]);
let pageStatus = $state<'idle' | 'loading' | 'request-error' | 'identity-error'>('idle');
let request: AbortController | null = null;
const pageSize = $derived(collocationPageSize(selectedCorpusIds.length));
```

`loadMore()` must request the exact submitted query, this particle, `offsetPerParticle: items.length`, and `limitPerParticle: pageSize`. Before appending, require `responseMatchesResult(...)`, the requested group, the unchanged total, and a non-empty page while `items.length < totalMatchingCollocations`. An identity mismatch uses the distinct message `Data changed — update the search before loading more.` A request failure preserves items and exposes `Try again`.

Render `Showing {items.length} of {group.totalMatchingCollocations}` and a column footer button named `Load {Math.min(pageSize, total - items.length)} more`. Disable it during loading. Iterate `items`, keyed by object identity because the list only appends and is reset with the visible result. Abort on component teardown.

- [ ] **Step 5: Report total matches in the summary**

Change `SearchSummary`'s count reduction from `returnedCount` to `totalMatchingCollocations` and label it `matching results for …`. The exact assertions from Step 1 fail under both the old reduction and old label, then prove that this value remains stable while one column appends.

- [ ] **Step 6: Run frontend and browser gates**

```bash
nix build --no-link -L \
  .#checks.x86_64-linux.frontend \
  .#checks.x86_64-linux.playwright
```

Expected: unit/type/build checks and every Playwright scenario pass.

- [ ] **Step 7: Commit per-particle loading**

```bash
git add natsume-frontend/src/routes/+page.svelte \
  natsume-frontend/src/lib/components/ParticleOverview.svelte \
  natsume-frontend/src/lib/components/ParticleColumn.svelte \
  natsume-frontend/src/lib/components/SearchSummary.svelte \
  natsume-frontend/tests/test.ts
git commit -m "feat: load more collocations per particle"
```

### Task 5: Example Accumulation

**Files:**
- Modify: `natsume-frontend/src/lib/components/ParticleColumn.svelte`
- Modify: `natsume-frontend/src/lib/components/CollocationItem.svelte`
- Modify: `natsume-frontend/src/lib/components/SentenceExamples.svelte`
- Test: `natsume-frontend/tests/test.ts`

**Interfaces:**
- Consumes: `ExamplesResponse.hasMore` and `responseMatchesResult` from Task 3.
- Produces: initial page of five, later pages of 20, preserved loaded examples on retry, and distinct more/empty/exhausted states.

- [ ] **Step 1: Write failing browser tests for example paging**

Route example requests for `情報を集める`. Let the initial real response through but override `hasMore: true`; on `offset` equal to its length, return one distinct synthetic example with `hasMore: false` and the same build/corpus identity.

Assert the initial disclosure shows `4 examples shown` and an accent `Load more examples` footer, then after activation shows the new sentence and `All examples shown` on a neutral footer.

Add an identity-mismatch route returning a different `databaseBuildId`; assert the synthetic sentence is not appended and `Data changed — update the search before loading more.` appears. Add a request-failure route and assert existing examples remain with a `Try again` action.

- [ ] **Step 2: Run Playwright and verify RED**

```bash
nix build --no-link -L .#checks.x86_64-linux.playwright
```

Expected: the example load-more control cannot be found.

- [ ] **Step 3: Pass build identity through the existing component chain**

From `ParticleColumn`, pass `databaseBuildId` through `CollocationItem` to `SentenceExamples`. Keep corpus IDs canonical and unchanged. Do not introduce context, a store, or a paging interface.

- [ ] **Step 4: Implement incremental examples**

Refactor `SentenceExamples.load(limit)` to return immediately when `status === 'loading'` and otherwise use `offset: examples.length`. Initial expansion calls `load(5)` once; the footer calls `load(20)`. Store `hasMore` from every accepted response. Validate build/corpus identity before appending. The loading branch must replace the load-more button, so both the function guard and the rendered state enforce one in-flight request per disclosure.

Render loaded examples before transient footer state so loading or retry never hides them. Keep the existing empty branch and example `<ul>` unchanged. Immediately after that list in the non-empty branch, render these terminal states:

```svelte
{#if status === 'loading'}
	<div
		class="mt-2 rounded border border-blue-200 bg-blue-50 p-2 text-blue-900 dark:border-blue-800 dark:bg-blue-950 dark:text-blue-100"
	>
		Loading more examples…
	</div>
{:else if status === 'request-error'}
	<button onclick={() => load(20)}>Try again</button>
{:else if status === 'identity-error'}
	<p>Data changed — update the search before loading more.</p>
{:else if hasMore}
	<button
		class="mt-2 w-full rounded border border-blue-200 bg-blue-50 p-2 text-blue-900 hover:bg-blue-100 dark:border-blue-800 dark:bg-blue-950 dark:text-blue-100 dark:hover:bg-blue-900"
		onclick={() => load(20)}
	>
		Load more examples
	</button>
{:else}
	<div class="mt-2 rounded bg-gray-100 p-2 text-gray-600 dark:bg-gray-800 dark:text-gray-300">
		All examples shown
	</div>
{/if}
```

An empty accepted later page sets `hasMore=false` and becomes exhausted. Abort an active request on unmount.

- [ ] **Step 5: Run all frontend gates**

```bash
nix build --no-link -L \
  .#checks.x86_64-linux.frontend \
  .#checks.x86_64-linux.playwright
```

Expected: lint, Svelte diagnostics,  unit tests, build, and browser tests pass.

- [ ] **Step 6: Commit example loading**

```bash
git add natsume-frontend/src/lib/components/ParticleColumn.svelte \
  natsume-frontend/src/lib/components/CollocationItem.svelte \
  natsume-frontend/src/lib/components/SentenceExamples.svelte \
  natsume-frontend/tests/test.ts
git commit -m "feat: load more examples inline"
```

### Task 6: Expandable-Row Surfaces

**Files:**
- Modify: `natsume-frontend/src/lib/components/CollocationItem.svelte`
- Test: `natsume-frontend/tests/test.ts`

**Interfaces:**
- Preserves: example paging and disclosure ownership from Task 5.
- Produces: visually explicit collapsed/open/focus states and an immediately adjacent next row.

- [ ] **Step 1: Write failing browser tests for row surfaces**

Expand a collocation with multiple examples. Assert the next collocation's `<summary>` is the next summary after the expanded `<details>` in the same column. Compare computed summary backgrounds to prove collapsed and open summaries differ, toggle dark mode, and repeat the distinction. Focus the next summary and assert its computed `outlineStyle` is not `none`. Assert the summary contains an `aria-hidden="true"` chevron whose transform changes when the disclosure opens.

- [ ] **Step 2: Run Playwright and verify RED**

```bash
nix build --no-link -L .#checks.x86_64-linux.playwright
```

Expected: background, focus-outline, and custom-chevron assertions fail.

- [ ] **Step 3: Make each summary a full-width explicit surface**

Give `<details>` a `group` class. Hide the native marker with `[&>summary]:list-none` and `[&>summary::-webkit-details-marker]:hidden`. Add this chevron as the first summary child:

```svelte
<span
	aria-hidden="true"
	class="inline-block transition-transform group-open:rotate-90"
>
	▶
</span>
```

Change the summary grid to include the chevron and use a complete surface class:

```svelte
class="grid w-full cursor-pointer grid-cols-[auto_minmax(6rem,2fr)_minmax(0,3fr)] items-center gap-2 rounded border border-gray-200 bg-gray-50 p-2 font-medium hover:bg-blue-50 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-blue-600 group-open:border-blue-300 group-open:bg-blue-100 dark:border-gray-700 dark:bg-gray-900 dark:hover:bg-blue-950 dark:group-open:border-blue-700 dark:group-open:bg-blue-950"
```

Keep examples directly under their summary and the next `<details>` directly after the expanded one; do not add indentation, absolute positioning, sticky behavior, or a nested vertical scroller.

- [ ] **Step 4: Run frontend and browser gates**

```bash
nix build --no-link -L \
  .#checks.x86_64-linux.frontend \
  .#checks.x86_64-linux.playwright
```

Expected: all frontend and browser checks pass.

- [ ] **Step 5: Commit the row restyle separately**

```bash
git add natsume-frontend/src/lib/components/CollocationItem.svelte \
  natsume-frontend/tests/test.ts
git commit -m "style: clarify expandable collocation rows"
```

### Task 7: Benchmark, Documentation, and Scaffolding Retirement

**Files:**
- Modify: `src/natsume_simple/benchmark_service.py:35-90`
- Test: `tests/test_benchmark_service.py:45-80`
- Modify: `README.md:215-232`
- Delete after all gates pass: `docs/superpowers/plans/2026-08-14-incremental-results.md`

**Interfaces:**
- Produces: fixed benchmark requests for a later collocation page and a late example page.
- Produces: a hard failure when either curated later-page request returns zero rows.
- Produces: durable README parameter/response documentation.
- Preserves: the design document as the durable measurement, premise, tradeoff, and revisit-trigger record.
- Retires: only the completed implementation checklist after tests and documentation own the behavior.

- [ ] **Step 1: Write failing benchmark-family assertions**

Extend `test_request_family_is_fixed_and_repeats_corpus_ids` to expect two new endpoint names:

```python
"collocations-page-verb",
"examples-page",
```

Assert the first uses `term=する`, `pos=verb`, `particle=を`, `offsetPerParticle=4500`, and the selection-derived limit supplied by the benchmark family. Assert the second uses `noun=必要`, `particle=が`, `verb=ある`, `offset=4000`, and `limit=20`.

Extend the test HTTP handler with `/empty-collocations` returning `{"particleGroups": [{"returnedCount": 0}]}` and `/empty-examples` returning `{"examples": []}`. Add a parametrized test that runs one request named `collocations-page-verb` or `examples-page` against those paths and expects `BenchmarkError("benchmark_empty_result:<name>")`. This prevents a changed artifact from turning the late offsets into fast empty-page measurements.

- [ ] **Step 2: Run the benchmark unit test and verify RED**

```bash
nix develop .#test --command pytest tests/test_benchmark_service.py \
  -k 'request_family or empty_later_page' -q
```

Expected: the endpoint-name/query assertions fail and empty later pages are not rejected.

- [ ] **Step 3: Add the two fixed later-page requests**

Import the backend capacity constants from `natsume_simple.api` and compute:

```python
from natsume_simple.api import (
    COLLOCATION_ITEM_CORPUS_BUDGET,
    MAX_COLLOCATIONS_PER_PARTICLE,
)

page_size = min(
    MAX_COLLOCATIONS_PER_PARTICLE,
    COLLOCATION_ITEM_CORPUS_BUDGET // len(corpus_ids),
)
```

Reject an empty corpus tuple. Add the two named endpoints with the parameters above and repeated corpus IDs.

After reading the response body but outside the elapsed-time measurement, validate only the two late-page names:

```python
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
```

In `_fetch`, raise `BenchmarkError(f"benchmark_empty_result:{endpoint.name}")` when this function returns zero. Keep the existing warmup, concurrency, success, p50, p95, max, and body-size reporting unchanged.

- [ ] **Step 4: Run benchmark and backend suites**

```bash
nix develop .#test --command pytest tests/test_benchmark_service.py tests/test_api_contract.py -q
```

Expected: both suites pass.

- [ ] **Step 5: Document the durable API contract**

Expand README's API section with:

```markdown
- `GET /api/collocations?term=...&pos=noun&particle=を&offsetPerParticle=150&limitPerParticle=150`
- `GET /api/examples?noun=...&particle=を&verb=...&offset=5&limit=20`

Collocation offsets require `particle`; `totalMatchingCollocations` remains the full
selection-specific count. Example responses include `hasMore`. Pages are stable only
within one immutable `databaseBuildId`; clients must not combine different build or
corpus identities.
```

- [ ] **Step 6: Run fresh complete verification**

```bash
nix build --no-link -L \
  .#checks.x86_64-linux.source-quality \
  .#checks.x86_64-linux.backend \
  .#checks.x86_64-linux.frontend \
  .#checks.x86_64-linux.playwright \
  .#checks.x86_64-linux.server-smoke
```

Then verify no development service remains:

```bash
if ss -ltn '( sport = :8000 or sport = :4173 )' | grep -q LISTEN; then
  ss -ltnp '( sport = :8000 or sport = :4173 )'
  exit 1
fi
```

Expected: every derivation succeeds and neither test port has a listener.

- [ ] **Step 7: Retire scaffolding and commit the completed feature**

Only after Step 6 succeeds, remove this completed plan with `apply_patch`. Retain `docs/superpowers/specs/2026-08-14-incremental-results-design.md`; it owns the measured repeated-page costs, total-order proof, immutability premise/disconfirmer, accepted no-total tradeoff, and one-second revisit triggers. Then commit explicit paths:

```bash
git add README.md src/natsume_simple/benchmark_service.py tests/test_benchmark_service.py
git add -u docs/superpowers/plans/2026-08-14-incremental-results.md
git commit -m "docs: publish incremental result contract"
```

Confirm `git status --short` lists only the owner's pre-existing untracked files.
