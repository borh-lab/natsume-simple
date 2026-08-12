# Frontend State and Component Simplification Design

**Status:** Revised draft after written review

**Date:** 2026-08-12
**Boundary:** Svelte application structure and browser behavior after the typed API
contract is stable

## Problem

The current 750-line page braids API calls, search state, global stores, derived
frequency calculations, colors, direct DOM mutation, scroll coordination,
desktop/mobile markup, menu state, tooltips, and error logging. The combined-mode
migration left a parallel obsolete result path and inaccurate types. Child
components receive writable stores and untyped callbacks, preventing their
interfaces from describing what they actually need.

This is a **Decomplect** candidate first. Merely splitting the page into more
files would preserve the same ordering and state coupling. The accepted API
contract supplies the protocol seam; the refactor must separate request
lifecycle, small presentation calculations, and rendering.

## Goals

- Make `+page.svelte` page composition and layout rather than an application
  controller.
- Centralize coordinated page state and request succession in one page-scoped
  owner.
- Keep only stacked-bar math, corpus colors, and safe highlighting as pure data
  transformations; the API owns filtering, ranking, grouping, and counts.
- Pass typed values/callbacks to presentation components, not writable stores.
- Reuse controls across desktop/mobile layouts and meet keyboard/accessibility
  requirements.
- Preserve characterized behavior while landing semantic improvements in
  separate patches.

## Non-Goals

- Visual redesign, additional search modes, SSR, offline operation, persistent
  browser storage, a general state-management framework, or a generated runtime
  API SDK.
- Backend or schema changes beyond consuming Spec 2.

## Components

| Component               | Purpose                          | Inputs                                            | Outputs                              | Dependencies              | State / time / identity                              |
| ----------------------- | -------------------------------- | ------------------------------------------------- | ------------------------------------ | ------------------------- | ---------------------------------------------------- |
| API client              | HTTP transport and error parsing | Typed method args, abort signal                   | Typed response or typed public error | `fetch`, generated schema | Stateless; request ID returned with errors           |
| Search controller       | Coordinate page transitions      | User commands, API client                         | Reactive page view state             | Svelte 5 runes            | Owns current request generation and abort controller |
| Presentation math       | Compute bar segments and colors  | Contributions and corpus metadata                 | New immutable display values         | None                      | Pure values; no time/state                           |
| Page                    | Compose layout/components        | Controller view state                             | User-visible page                    | Components                | No duplicated domain state                           |
| Presentation components | Render and emit intent           | Typed values/callbacks                            | DOM events/intents                   | Svelte                    | Local ephemeral UI state only                        |

## Typed API Client

`src/lib/api/client.ts` owns:

- Same-origin base URL by default and explicit development override.
- `URLSearchParams` construction.
- `fetch`, abort signal propagation, `response.ok`, JSON content-type checking,
  and common error-envelope parsing.
- Methods `getCorpora`, `getSuggestions`, `getCollocations`, and `getExamples`.

`getCollocations` always sends selected corpus IDs and `rankBy`; `getExamples`
sends the same corpus selection.

It owns no Svelte state, ranking choice, colors, filtering, retries, notifications,
or cache. Unexpected/non-JSON responses become a stable client error without
exposing response bodies to users.

## Search Controller State Machine

State:

- Draft `term` and `pos`.
- `submittedQuery` from the last successful submission attempt.
- Corpus metadata and selected corpus IDs.
- Ranking mode `rankBy: raw | meanPerMillion`, using the API enum unchanged.
- Status `idle | loading | success | empty | error`.
- Last successful collocation response.
- Current public error.
- Monotonically increasing request generation and current abort controller.

Transitions:

```text
initialize ──► load corpora ──► submit default query
edit draft ──► state only
submit ──► abort previous ──► loading(generation N)
  ├─ latest success + items ──► success
  ├─ latest success + empty ──► empty
  ├─ latest expected failure ──► error, retain last successful result
  └─ stale completion ──► ignored
select corpus / rankBy ──► abort previous ──► request selection + rankBy
```

Only the current generation may change visible request state. Aborted requests
do not display an error. A failed request retains the last successful result but
clearly marks it as stale relative to the failed submitted query; the interface
must not imply the old results belong to the new term.

The controller is instantiated by the page, not exported as a process-global
singleton. This prevents state leaking across component tests or future SSR.
Corpus/ranking changes are debounced only enough to coalesce one interaction
turn; they never locally reinterpret a globally truncated page. While the new
request runs, the prior result is visibly stale and cannot be labelled with the
new selection.

## Presentation Math

`src/lib/presentation/search.ts` contains only two typed pure calculations used
across the particle presentation components:

- `stackSegments`: percentages and offsets from selected contributions.
- `corpusColorMap`: stable color slots from corpus metadata order.

They return new values and never sort or mutate API-owned arrays. The UI renders
server groups, order, totals, and returned counts directly. It requires at least
one selected corpus; deselecting the last corpus is disabled and explained
accessibly. For a non-zero stack, final percentage totals equal 100 within a
documented floating-point tolerance; the last segment may absorb rounding for
pixel display only.

## Safe Sentence Segmentation

A pure function accepts sentence text and three typed spans, validates their
bounds/order, and returns an ordered sequence of `{kind, text}` segments where
kind is `plain | noun | particle | verb`. Components render `text` through
ordinary Svelte interpolation and semantic markup. The function never returns
HTML.

Invalid or unsupported overlap yields one plain segment containing the complete
original text and signals a non-sensitive validation result to the caller.

## Local Example State

Each mounted `SentenceExamples` owns `idle | loading | success | empty | error`
state and one abort controller. Reopening the same disclosure while its result
item remains mounted reuses that local result. Result items use the stable string
key `JSON.stringify([selectedCorpusIds, noun, particle, verb])`, with the
server-echoed canonical corpus order. A corpus-selection change therefore
remounts each item, aborts its request during destruction, and drops stale
example state. A `rankBy` change preserves every keyed item still present after
reranking, including its open disclosure, because examples do not depend on
ranking. The frontend fixes the examples `limit` at 5; exposing that control
later requires adding it to the key. There is no page cache, LRU policy,
cross-result reuse, or persistent browser storage.

## Presentation Components

The intended component set is:

- `SearchControls`
- `CorpusOptions`
- `SearchSummary`
- `ParticleOverview`
- `ParticleColumn`
- `CollocationItem`
- `SentenceExamples`
- `MobileMenu`
- Existing scroll/theme primitives that remain cohesive

Component props are typed values and intent callbacks. Components do not import
global corpus/search stores. Desktop and mobile layouts reuse the same controls,
options, summary, and menu content instead of duplicating markup.

## Accessibility and Interaction Contract

- Search uses a form so Enter and the search button invoke one submission path.
- Suggestions implement combobox/listbox roles, active descendant/selection,
  Arrow Up/Down, Enter, Escape, and outside dismissal.
- Loading indicators retain accessible button/control names.
- Menus/disclosures expose labels, `aria-expanded`, and controlled region IDs.
- Example expansion announces loading, empty, and error states.
- Horizontal scroll buttons are keyboard buttons with directional labels.
- Focus is not moved on background search completion; explicit menu/dialog
  actions manage focus locally.
- Network failures are visible and associated with the attempted query.
- Corpus text is always escaped; no `{@html}` or `innerHTML` is used for API data.
- Production debug `console.log` calls are absent.

## Refactor Sequence and Behavior Boundary

1. Characterization tests from Spec 1 pin current behavior.
2. Adopt generated API types and handwritten client without changing layout.
3. Extract only shared presentation math and compare it against focused fixtures.
4. Introduce the page-scoped controller behind existing UI.
5. Convert component store props to values/callbacks.
6. Reuse the mobile/desktop content and delete the duplicate implementation.
7. Delete obsolete stores/types/functions whose last callers are removed.

Safe rendering, latest-request-wins semantics, and visible error behavior are
intentional semantic changes and land as separate patches from structural
steps. A structure patch must not opportunistically alter sort order, labels,
colors, or network timing.

## Test Strategy

### Pure unit/property tests

- Presentation math never mutates API responses.
- Percentage stacks reconcile to 100 within tolerance for positive totals.
- The last selected corpus cannot be deselected.
- Corpus color slots remain stable for fixed metadata order.
- Highlight segmentation preserves every original code point exactly once and
  cannot produce markup.
- Generated operation sequences over controller submissions prove stale
  completions cannot replace the latest generation.

### Component tests

- Form submission, disabled/loading labels, and errors.
- Suggestion keyboard navigation and dismissal.
- Corpus selection and `rankBy` intents.
- Corpus/ranking changes abort stale work and request the new corpus set/rank.
- A corpus change remounts example state; reranking preserves an open item that
  remains in the result set and its loaded examples.
- Menu/disclosure ARIA state.
- Example loading/empty/error/success rendering.
- Malicious text remains text.

### Browser test

Against the fixture backend:

1. Load corpus metadata.
2. Submit a noun search.
3. Filter one corpus.
4. Expand a collocation and load examples.
5. Switch `rankBy` and verify that a keyed item present in both responses
   preserves the disclosure.
6. Change corpus selection and verify that examples are requested for the new
   canonical selection.
7. Verify HTML-shaped example content is inert text.
8. Exercise empty results and a controlled API failure.
9. Repeat the primary controls at a mobile viewport.

## Acceptance Criteria

- `+page.svelte` contains composition/layout and no raw fetch, SQL-shaped data
  transformation, tooltip DOM construction, or duplicated mobile menu.
- Search state has one page-scoped owner and no global writable search stores.
- Every component passes typed values/callbacks and `svelte-check` is green.
- Presentation-math tests and controller succession tests pass.
- Example state is component-local, survives rank-only reordering, and cannot
  survive a corpus-selection change.
- No corpus/API text flows through HTML-string rendering.
- Desktop and mobile flows pass their accessibility and Playwright assertions.
- Obsolete combined-mode types/stores/functions are removed after their last
  characterized caller disappears.

## Decision Log

| Decision                                      | Status   | Reason                                                                        | Revisit trigger                                         |
| --------------------------------------------- | -------- | ----------------------------------------------------------------------------- | ------------------------------------------------------- |
| Page-scoped controller                        | Accepted | Coordinated state without process-global leakage/framework                    | Multiple routes need shared search state                |
| Small presentation-math module                | Accepted | Two shared pure calculations retain cheap tests without pretending to own domain ranking | The functions gain only one component consumer |
| Values/callbacks for component interfaces     | Accepted | Honest, independently testable components                                     | A component genuinely owns shared mutable state         |
| Component-local example state                 | Accepted | Dependency-shaped keys reuse visible examples across ranking changes and invalidate them on corpus changes without cache policy | Measured traffic shows cross-result caching is needed |
| Server owns selection-specific ranking        | Accepted | Client-side filtering cannot recover items omitted by server top-N truncation | API provides an unbounded or cursor-complete result set |
| Retain last success on error with stale label | Accepted | User can inspect prior data without mistaking query identity                  | User research prefers clearing immediately              |
