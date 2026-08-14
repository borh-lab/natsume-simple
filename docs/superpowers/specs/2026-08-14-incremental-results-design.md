# Incremental Collocation and Example Results

Status: Approved for implementation

## Purpose

Users can inspect more than the first 150 collocations in any particle column and more than the first five examples for any collocation. Loading additional data affects only the column or disclosure the user chose. The interface continues to identify every visible result with the submitted search that produced it.

## Current behavior

- The frontend requests 150 collocations per particle.
- `/api/collocations` accepts at most 200 items per particle and restricts the item/corpus product to 450 so a response stays below one mebibyte.
- `totalMatchingCollocations` already reports how many items exist, but the frontend has no way to request a later page.
- `/api/examples` returns at most 20 occurrences and the frontend requests five once per expanded row.
- Collocation and example ordering is deterministic over one immutable database artifact.

## Constraints

- Pagination is independent per particle and per expanded collocation.
- Initial search responses retain the current 150-item page size.
- A page must remain below the existing response-size bound.
- Pages from different `databaseBuildId` values are never combined.
- Corpus selection, submitted term, and search direction remain part of result identity.
- Existing ranking, corpus distributions, highlighting, bar scales, and raw-frequency tooltips do not change.
- The database remains read-only and immutable for the lifetime of a server process.
- The page remains usable by keyboard and in dark mode.

## Decision

Use targeted offset pagination through the existing endpoints.

Offset pagination is sufficient because the underlying artifact is immutable and both result types have a complete deterministic order. An opaque cursor would add encoding, validation, and versioning without protecting a present failure mode. Re-fetching a growing prefix would repeatedly transfer and render rows the client already has and would still collide with the whole-response size bound.

## Collocation API

`GET /api/collocations` gains two optional parameters:

- `particle`: one of `が`, `を`, `に`, `で`, `から`, `より`, `と`, or `へ`.
- `offset`: an integer greater than or equal to zero, defaulting to zero.

`offset > 0` requires `particle`; otherwise the endpoint returns the common `400 invalid_parameter` envelope. Omitting `particle` retains the current initial-search behavior and returns every non-empty particle group. Supplying `particle` filters the database query by particle before aggregation. It returns exactly that group when the particle has matches before pagination, including when the requested page itself is empty; it returns no groups only when the particle has no matches.

Each group retains the existing fields:

- `totalMatchingCollocations` is the full selection-specific count for that particle.
- `returnedCount` is the number of items in this page.
- `items` is the deterministic slice `[offset:offset + limitPerParticle]`.
- `corpusDistribution` is calculated over every matching item, not only the page.

Ordering remains:

1. descending `meanFrequencyPerMillion`;
2. descending `totalRawFrequency`;
3. ascending noun;
4. ascending particle;
5. ascending verb.

The existing `limitPerParticle` validation and item/corpus budget apply to every page. The frontend uses pages of 150, so a three-corpus page remains within the measured bound.

## Example API

`GET /api/examples` gains `offset`, an integer greater than or equal to zero that defaults to zero. `limit` retains its existing bounds and default.

`ExamplesResponse` gains `hasMore`. The endpoint queries `limit + 1` deterministic occurrences, returns at most `limit` in `examples`, and sets `hasMore` when the extra occurrence exists. This proves whether another page exists without adding a full-count query or a total that no present interaction needs.

The complete example order is:

1. corpus ID;
2. source ID;
3. sentence ID;
4. noun start and end;
5. particle start and end;
6. verb start and end;
7. extractor ID.

The final tie-breaker is not exposed, but it is stored and participates in the occurrence uniqueness constraint.

## Frontend data flow

### Initial search

The search controller continues to own the visible response and its immutable submitted input. The summary changes from the number currently returned to the sum of `totalMatchingCollocations`, labelled as matching results.

The particle overview is keyed by the visible-result object. A successful full search therefore creates fresh particle-column pagination state, while draft edits leave the currently displayed result and its loaded pages intact.

### Per-particle loading

Each `ParticleColumn` owns only its incremental interaction state:

- the appended item list;
- `idle | loading | error` page status;
- the current page request and abort controller.

The request uses the visible result's submitted term, direction, and canonical corpus IDs; the target particle; `offset = items.length`; and `limitPerParticle = 150`.

Only one request per column may run at once. A successful response is appended only when:

- its `databaseBuildId` equals the visible result's build ID;
- its echoed corpus IDs equal the visible result's canonical corpus IDs; and
- it contains the requested particle group.

A mismatch is not merged. The column reports that the data changed and asks the user to update the search. A network or capacity error preserves the rows already loaded and exposes a retry action.

The column displays `Showing N of M`. While `N < M`, its footer offers `Load min(150, M - N) more`. The button is disabled and labelled as loading while its request is active. It disappears when all rows are shown.

Appending lower-ranked rows cannot change either bar reference: the within-particle maximum and across-particle maximum are already present in the first descending page. Corpus distribution is also stable because the server calculates it over the complete matching set.

### Per-collocation examples

`SentenceExamples` keeps its existing local ownership and initially requests five examples at offset zero. It records `hasMore` and appends subsequent pages of five using `offset = examples.length`.

Loaded examples remain visible during an additional request or a retryable error. Responses with a different `databaseBuildId` or corpus selection are not appended. The request is aborted when the row unmounts.

The disclosure displays `N examples shown`. While `hasMore` is true, a full-width footer offers `Load more examples`. When at least one example has loaded and `hasMore` is false, the footer changes to `All examples shown`. A zero-result initial request instead keeps the existing `No examples found` state.

Both incremental owners use one shared pure identity predicate for `databaseBuildId` and canonical corpus IDs. Request lifecycle and accumulated rows remain local because the two consumers have different page sizes, response shapes, and visual states. A generic pagination store or loader interface would hide those differences without serving a second implementation.

## Visual and interaction design

Every collocation summary is a full-width, button-like surface rather than text distinguished only by the browser disclosure marker:

- a clear chevron communicates collapsed/open state;
- the full summary has a neutral background and border;
- hover and keyboard focus use an accent background and visible focus ring;
- the open summary retains a stronger accent background;
- dark-mode equivalents preserve the same distinctions.

Examples render directly below the open summary. The example pagination footer has an accent surface while more examples remain and a quiet neutral surface when the list is exhausted. The next collapsed collocation summary follows immediately after the expanded content and has its own full-width surface, so a reader reaching the end of a long example list can identify and expand the next row without returning to the top.

The particle-level load-more footer is visually separate from collocation summaries and stays within its particle column. Loading one column neither moves another column's pagination state nor resets horizontal scrolling.

## Error handling

- Invalid particle, offset, or limit values use the existing validation/error envelopes.
- A collocation offset without a target particle is rejected before querying.
- Page request errors are local; existing rows and other columns remain usable.
- Artifact or corpus-identity mismatch never produces a mixed list.
- An empty collocation page is treated as exhausted only when the column's known total has already been reached; otherwise it is reported as an inconsistent response.
- Example exhaustion follows the server's `hasMore` value; an empty page with `hasMore = true` is inconsistent.

## Tests

Backend contract tests cover:

- targeted particle filtering;
- the second page's exact order and absence of overlap;
- full-set totals and corpus distribution on later pages;
- rejection of an offset without a particle;
- unchanged response-size enforcement;
- example `hasMore` semantics, successive pages, and complete tie ordering;
- parameter bounds and common error envelopes.

Frontend unit tests cover:

- client serialization of particle and offset parameters;
- total-matching collocation summary math;
- append/exhaustion decisions as pure presentation calculations where useful.

Browser tests cover:

- loading a second page in one particle without changing another column;
- preserved bar scale, horizontal position, and existing expanded rows;
- visible loading, retry, and exhausted states;
- loading successive example pages without duplicates;
- rejecting an artifact-identity mismatch;
- full-summary hover, focus, open, dark-mode, and next-row visual states;
- keyboard activation of both load-more controls.

## Non-goals

- Infinite scrolling.
- A pagination framework or shared page cache.
- Opaque cursor encoding.
- Virtualized rows.
- Loading more than one particle from one incremental request.
- Persisting pagination across a new full search or page reload.

Virtualization is reconsidered only if a measured browser trace shows unacceptable rendering or interaction latency after several pages have been appended to one column.

## Architecture review

Architecture triage classifies this change primarily as **Protocol Design** with a contained **Trust/State Review**:

- The wire contract states target, ordering, bounds, error semantics, and page identity.
- Mutable page state has one owner per user interaction and is discarded with its visible result.
- Build and corpus identity are values checked before mutation, not ambient state assumed by the component.
- Offset pagination is justified by artifact immutability; a mutable serving database would disconfirm that premise and trigger a cursor redesign.

The review rejects deepening pagination into a shared module. Collocation pages and example pages share only a small identity predicate; their accumulation, exhaustion, and rendering rules are different. Keeping those places local makes their transitions visible and avoids a one-implementation paging abstraction.

It also rejects a total example count. The present consumer needs to know only whether another page exists and when to change the footer state. `hasMore` supplies exactly that fact with one bounded query.

## Acceptance criteria

- A user can display every matching collocation in one particle through repeated bounded requests.
- A user can display every example for one collocation through repeated bounded requests.
- No request response exceeds the existing one-mebibyte target in the maximum-size fixture.
- Pages are stable, non-overlapping, and never merged across artifact or corpus identities.
- Other particle columns and previously loaded rows retain their state during a targeted load.
- The next expandable collocation is visually clear after a long example list.
- The complete frontend, backend, and browser check suites pass.

## Lifecycle

After implementation, the durable API parameters and response field move into the README endpoint documentation, and behavior is owned by contract/browser tests. This design document is retired with the implementation scaffolding rather than becoming a parallel operational manual.
