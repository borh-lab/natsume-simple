# Executable Quality Baseline Design

**Status:** Draft for written review

**Date:** 2026-08-12
**Boundary:** Repository checks and current behavior; no new public API or
database design

## Problem

The repository cannot currently distinguish a regression from existing failure.
Backend pytest has two failures, `svelte-check` reports 53 errors in six files,
frontend lint fails formatting checks, and CI omits frontend type checking,
tests, browser behavior, and an explicit production build. The passing Vitest
test asserts arithmetic rather than application behavior, and the Playwright
test checks only for an `h1`.

Vite currently produces a bundle despite TypeScript/Svelte diagnostics, so a
successful frontend build is not evidence that the source contract is valid.
CI also invokes commands with `--fix`/`--write`, making a quality gate capable of
changing the tree it is meant to assess.

## Evidence Ledger

| Claim | Type | Evidence | Confidence | Impact |
|---|---|---|---|---|
| Backend suite is red | Observation | `uv run pytest -q`: 6 pass, 2 fail | High | Refactors/upgrades lack a trustworthy baseline |
| Generic loader iterates a path string as characters | Observation | `_load_sentences(List[Path])` receives `row["file_path"]` string; warnings name `a`, `r`, `t`, etc. | High | Generic corpus yields no sentences |
| Japanese normalization changed under current model output | Observation | GiNZA emits `突入`, `し`, `ちゃう`; current loop concatenates them | High | Established normalization fixture fails |
| Expected `突入する` behavior was intentionally recorded | Observation | Doctest and history preserve the expectation across a span-return refactor | High | Updating the expected value would silently change domain behavior |
| Frontend static checking is red | Observation | `npm run check`: 53 errors | High | Wire and component props are inconsistent |
| Frontend tests prove little | Observation | Arithmetic unit test and `h1` browser assertion | High | Green tests would not protect search behavior |
| CI is incomplete and mutating | Observation | Workflow calls Nix wrappers whose lint commands use write/fix modes | High | CI result is not a clean-checkout proof |

## Goals

- Establish a green, non-mutating baseline before protocol, schema, refactor, or
  major-version work.
- Fix confirmed defects at their source without mixing them with structural
  cleanup.
- Characterize user-visible and domain behavior that later specs can preserve or
  deliberately replace.
- Make each check runnable without the production database or a downloaded
  browser unless the check explicitly owns that dependency.

## Non-Goals

- Introduce the new `/api` contract.
- Replace the database schema or ingestion architecture.
- Perform safe-rendering changes, dependency-major upgrades, visual redesign, or
  broad Svelte restructuring.
- Make GPU inference part of ordinary CI.

## Required Changes

### Confirmed defect track A: generic corpus paths

The loader boundary accepts a sequence of relative `Path` values. A CSV scalar
is converted once at the adapter boundary to `Path(row["file_path"])`, then
wrapped in a one-element sequence. `_load_sentences` does not accept ambiguous
`str | Sequence[Path]` input because strings are iterable and recreate the
defect.

Tests cover one file, multiple files, missing files, invalid metadata, and
preservation of source-to-sentence association. The existing test expectation
is not weakened.

### Confirmed defect track B: Japanese verb normalization

The required normalized value remains `突入する` for `突入しちゃう`.
Implementation must recognize the observed サ変-capable stem followed by the
`する` auxiliary and exclude following aspectual/contracted auxiliaries from
the lemma while retaining the complete matched source span needed for
highlighting.

A parameterized fixture pins at least:

- `飛び立つでしょう → 飛び立つ`
- `考えられませんでした → 考えられる`
- `扱うかです → 扱う`
- `突入しちゃう → 突入する`
- `で囲んである → 囲む`
- `たらしめている → たらしめる`

The fixture records normalized lemma and half-open source offsets. Model-token
snapshots are diagnostic fixtures, not the public contract; tokenization may
change again while normalized behavior remains stable.

### Frontend baseline cleanup

Before changing architecture, tests characterize:

- Noun and verb submission direction.
- Particle order.
- Corpus filtering.
- Raw/normalized display selection.
- Example expansion.
- Desktop/mobile availability of the same controls.
- Theme selection.

Then only internally proven remnants of the combined-mode transition are
removed: unused `results`, `d`, `computeDerivedData`, `getMax`,
`lastSearchedNoun`, unused `corpusStats`, obsolete result types, and either the
duplicate inline mobile markup or the unused `MobileMenu` implementation. The
surviving mobile implementation is shared by the page.

Live component props and the current live API shape receive accurate types.
Missing `$lib/types` is either defined for a live concept or eliminated with the
dead caller. JavaScript Svelte components with typed props migrate to
`lang="ts"`; `unknown` is not suppressed with broad `any` casts.

### Test fixtures

A small generated DuckDB fixture contains two corpora, several sources and
sentences, noun- and verb-directed collocations, a zero-result term, and a
sentence containing HTML-shaped text. It is built during tests from explicit
rows rather than committed as a binary.

Backend integration tests invoke the FastAPI application through its real
router and the fixture database. Frontend network behavior uses deterministic
fixtures; the one Playwright happy path starts both fixture backend and frontend.

### Non-mutating command contract

The baseline exposes check-only commands for:

```text
ruff format --check
ruff check
mypy
pytest
prettier/biome check
eslint
svelte-check
vitest --run
vite build
playwright test
nix flake check
```

Formatting/fixing remains available as a distinct developer command. No check
uses `--write` or `--fix`.

Playwright browser installation is an explicit environment/bootstrap concern.
CI installs the pinned browser before the test and caches it by Playwright
version; a missing browser is an environment failure, not an ignored test.

## Test Claims

- Generic metadata paths load the intended files and never iterate characters.
- The normalization table above remains stable under the pinned NLP model.
- Existing endpoint queries compose through a real temporary DuckDB connection.
- Frontend corpus filtering and normalization selection retain characterized
  behavior.
- Frontend component props and current wire values type-check without escape
  hatches.
- The production frontend bundle is created.
- The browser happy path performs a search and expands an example against the
  fixture backend.
- Running the entire check suite leaves `git status --short` unchanged, excluding
  pre-existing user changes captured before the run.

## Acceptance Criteria

- All commands in the non-mutating command contract exit zero from a clean
  checkout.
- Backend tests include no model-download-on-import behavior.
- Unit and integration tests require neither `data/corpus.db` nor a network
  corpus download.
- The two confirmed defects have focused regression tests and separate commits.
- Placeholder arithmetic and `h1`-only tests are removed when stronger tests
  cover their test layers.
- CI runs every declared check and uploads useful failure artifacts without
  modifying tracked files.
- No public endpoint, persisted schema, normalization formula, or rendering
  trust behavior changes in this spec.

## Rollback

Each confirmed defect, characterization layer, dead-code cleanup, type cleanup,
and CI change is independently revertible. A CI expansion may initially expose
another pre-existing failure; that failure is fixed in its own commit rather
than suppressing or weakening the new check.

## Decision Log

| Decision | Status | Reason | Revisit trigger |
|---|---|---|---|
| Preserve documented Japanese normalized lemmas | Accepted | History and doctests show intentional domain behavior | Domain owner explicitly changes normalization policy |
| Use generated fixture databases | Accepted | Fast, deterministic, schema-visible integration seam | Fixture construction becomes slower than a validated binary fixture |
| Remove only proven combined-mode remnants | Accepted | Avoids mixing a rewrite into baseline repair | Characterization shows a supposedly dead path is reachable |
| Require non-mutating CI | Accepted | A check must observe, not repair, the submitted tree | None |
