# Executable Quality Baseline Design

**Status:** Revised draft after written review

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

| Claim                                                     | Type        | Evidence                                                                                            | Confidence | Impact                                                            |
| --------------------------------------------------------- | ----------- | --------------------------------------------------------------------------------------------------- | ---------- | ----------------------------------------------------------------- |
| Backend suite is red                                      | Observation | `uv run pytest -q`: 6 pass, 2 fail                                                                  | High       | Refactors/upgrades lack a trustworthy baseline                    |
| Generic loader iterates a path string as characters       | Observation | `_load_sentences(List[Path])` receives `row["file_path"]` string; warnings name `a`, `r`, `t`, etc. | High       | Generic corpus yields no sentences                                |
| Japanese normalization changed under current model output | Observation | GiNZA emits `突入`, `し`, `ちゃう`; current loop concatenates them                                  | High       | Established normalization fixture fails                           |
| Expected `突入する` behavior was intentionally recorded   | Observation | Doctest and history preserve the expectation across a span-return refactor                          | High       | Updating the expected value would silently change domain behavior |
| Frontend static checking is red                           | Observation | `npm run check`: 53 errors                                                                          | High       | Wire and component props are inconsistent                         |
| Frontend tests prove little                               | Observation | Arithmetic unit test and `h1` browser assertion                                                     | High       | Green tests would not protect search behavior                     |
| CI is incomplete and mutating                             | Observation | Workflow calls Nix wrappers whose lint commands use write/fix modes                                 | High       | CI result is not a clean-checkout proof                           |
| Executable examples are an active teaching surface        | Observation | `AGENDA.md` walks the backend and doctests; pytest enables `--doctest-modules`; 26 prompt examples exist | High    | A rewrite can stay green after silently deleting the walkthrough |

## Goals

- Establish a green, non-mutating baseline before protocol, schema, refactor, or
  major-version work.
- Fix confirmed defects at their source without mixing them with structural
  cleanup.
- Characterize user-visible and domain behavior that later specs can preserve or
  deliberately replace.
- Make each check runnable without the production database or a downloaded
  browser unless the check explicitly owns that dependency.
- Inventory and preserve the executable examples used to teach the backend
  walkthrough.

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
- `いるからで → いる`
- `いるという → いる`
- `語ります → 語る`
- `しました。 → する`
- `作り上げたか → 作り上げる`
- `見られなかったが → 見られない`

The fixture records normalized lemma and half-open source offsets. Model-token
snapshots are diagnostic fixtures, not the public contract; tokenization may
change again while normalized behavior remains stable.

The last six cases migrate from the notebook before its unittest class is
deleted. The notebook explicitly labels `見られないが` as an intentionally
failing expected value and says `見られない` is correct, so the deliberate
failure is not promoted into the domain contract.

### Executable teaching contract

Spec 1 records executable coverage as a boundary → test target → taught behavior
mapping. The initial 26 `>>>` prompts cover only Japanese-text filtering and
normalization; source adaptation, segmentation, occurrence construction,
aggregation, and query semantics start uncovered. Spec 1 is not accepted until
all seven rows name and execute at least one fixture-sized example. CI validates
the mapping targets, while the total prompt count is informational only.

One such decision is made here: replace the local `pairwise` implementation with
Python 3.12's `itertools.pairwise`. Its three examples teach a reimplementation
of a standard primitive, so the prompt inventory falls from 26 to 23 before the
five coverage gaps are filled. The boundary mapping, not the resulting total,
guards the teaching surface.

Every public transformation boundary on the documented backend walkthrough has
at least one short executable example using fixture-sized values. This includes
source adaptation, Japanese-text filtering, segmentation, normalization,
occurrence construction, aggregation, and request/query parameter semantics.
HTTP lifecycle orchestration and trivial getters use integration tests instead
of ceremonial doctests.

Examples are adjacent to the code they explain, deterministic, and readable
without the production corpus. A reviewer may reject a helper or abstraction
that makes the walkthrough harder to read even when it reduces local line count.

The pattern-extraction notebook demonstrates the module rather than forking it.
Before deletion, its normalization expectations migrate to the table above and
its extraction expectations migrate to a parameterized characterization table:

| Input                                        | Expected normalized triples                                  |
| -------------------------------------------- | ------------------------------------------------------------ |
| `東京では，銀座でランチをたべよう。`         | `銀座・で・食べる`, `ランチ・を・食べる`                     |
| `京都にも行く。`                             | none                                                         |
| `ことを説明するならば`                       | `こと・を・説明する`                                         |
| `ことにならない`                             | `こと・に・ならない`                                         |

The first two cases preserve compound-particle exclusion and the table also
preserves multi-result and ordinary extraction. Model-free token observations
add explicit `では`, `には`, `をも`, and `へと` exclusion cases. The last row is
characterization of current GiNZA fixed/compound handling, not a claim that it
is the preferred linguistic analysis; changing it requires a reviewed semantic
decision.

Only after both tables pass does the notebook import `simple_lemma`,
`normalize_verb_span`, and extraction functions from
`natsume_simple.pattern_extraction` and delete its duplicate implementations and
unittest classes. Narration, dependency visualization, KWIC, and `Counter`
exploration remain. The pytest tables are authoritative; adjacent module
doctests may retain representative teaching examples without becoming a second
implementation.

### Proven dead-code and primitive cleanup

Small verified deletions remain separate from the two defect fixes:

- Delete `utils.py`, its imports, and both ineffective `--seed` flags. The data
  command performs no random work; the extraction command's DuckDB `USING
  SAMPLE` is not controlled by Python/NumPy/Torch seeding. Spec 3 separately
  prohibits random sampling in publishable builds and owns real deterministic
  controls.
- Delete uncalled `filter_non_japanese`; retain the live `is_japanese` predicate.
- Replace `setup_logger` with module-level `logging.getLogger(__name__)` and one
  `logging.basicConfig(...)` call in each CLI entry point. Server logging remains
  owned by its ASGI runner. This also removes duplicate handlers on repeated
  setup.
- Delete the uncalled `/corpus/stats` route and its README example. Keep
  `calculate_corpus_stats` temporarily because `/corpus/norm` calls its result;
  Spec 2 removes both with the complete legacy API.
- Remove unused `pkgs.bun`, `pkgs.sqlite`, Biome ownership, and the Bun editor
  recommendation from the flake, tracked devcontainers, and VS Code settings.
- Delete stale `CHANGELOG.md` and `pkgs.git-cliff`. There is no release workflow
  or cliff configuration, the file has only two historical updates and only an
  unreleased section, and repository versioning already moved beyond its sole
  release tag. A changelog returns only with a named release consumer.

README structure, command, logging, and endpoint examples are updated in the
same focused cleanup commits so the teaching entry point never documents deleted
surfaces.

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

The tracked legacy `static/index.html` and `static/app.js` are removed after the
baseline proves no runtime path serves them; the server mounts only
`natsume-frontend/build` and the legacy page's CDN-loaded Vega application has no
consumer.

The current frontend receives the narrowest honest annotations needed for
`svelte-check` to pass. Missing `$lib/types` is defined only for a live concept or
eliminated with its dead caller. This spec does not exhaustively model response
fields or component interfaces that Specs 2 and 5 replace; transitional unknown
data is narrowed at its use site, never hidden by project-wide `any` declarations
or disabled checking.

### Model-free and model-dependent checks

Default tests exercise normalization policy from small captured token-observation
fixtures and require no GiNZA model. A separately marked `nlp_model` integration
tier loads the locked `ja-ginza` package and proves that actual model output maps
to the same normalization table. The model tier performs no runtime download and
runs as a release/scheduled gate through a declared Nix output in Spec 6; it is
not part of ordinary `nix flake check`.

This split deliberately allows the default gate to remain green when a new
GiNZA release changes tokenization but the captured policy examples still pass.
The model-dependent release gate is the owner that detects that drift, and it
must run before any NLP dependency or production-corpus release.

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
pytest -m "not nlp_model"
prettier --check
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

Prettier owns frontend formatting and ESLint owns frontend linting. Biome is
removed from the frontend gate and configuration rather than retained as a
second owner. Spec 1 also replaces mutating Nix wrappers with check-only
commands, makes the development-shell hook setup-free, and adds a minimal
`checks` output delegating to this baseline. Spec 6 later replaces those command
wrappers with full derivations and production closures.

## Test Claims

- Generic metadata paths load the intended files and never iterate characters.
- The normalization table remains stable in model-free policy fixtures and the
  separately declared pinned-model integration tier.
- Walkthrough examples execute and every taught boundary retains a mapped
  behavior target.
- Existing endpoint queries compose through a real temporary DuckDB connection.
- Frontend corpus filtering and normalization selection retain characterized
  behavior.
- Live frontend use sites type-check without project-wide escape hatches; fields
  already scheduled for removal need no polished transitional public model.
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
- Ordinary `nix flake check` requires no large NLP model; the explicit model
  integration output is green before release or an NLP dependency update.
- The boundary inventory has no uncovered row, every mapped example executes,
  and its informational count accounts explicitly for the three `pairwise`
  examples removed before new boundary examples are added.
- The two confirmed defects have focused regression tests and separate commits.
- Placeholder arithmetic and `h1`-only tests are removed when stronger tests
  cover their test layers.
- CI runs every declared check and uploads useful failure artifacts without
  modifying tracked files.
- No consumed public endpoint, persisted schema, normalization formula, or
  rendering trust behavior changes in this spec; `/corpus/stats` is the sole
  endpoint deletion, after repository-wide reachability evidence and the
  no-external-consumer premise establish that it is dead.

## Rollback

Each confirmed defect, characterization layer, dead-code cleanup, type cleanup,
and CI change is independently revertible. A CI expansion may initially expose
another pre-existing failure; that failure is fixed in its own commit rather
than suppressing or weakening the new check.

## Decision Log

| Decision                                                     | Status   | Reason                                                                                                                         | Revisit trigger                                                     |
| ------------------------------------------------------------ | -------- | ------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------- |
| Preserve documented Japanese normalized lemmas               | Accepted | History and doctests show intentional domain behavior                                                                          | Domain owner explicitly changes normalization policy                |
| Normalize `見られなかったが` to `見られない`                 | Accepted | The notebook marked its trailing-`が` expectation as a known failure; conjunctive particles are excluded consistently elsewhere | The domain owner chooses to retain a trailing conjunctive particle |
| Use generated fixture databases                              | Accepted | Fast, deterministic, schema-visible integration seam                                                                           | Fixture construction becomes slower than a validated binary fixture |
| Remove only proven combined-mode remnants                    | Accepted | Avoids mixing a rewrite into baseline repair                                                                                   | Characterization shows a supposedly dead path is reachable          |
| Require non-mutating CI                                      | Accepted | A check must observe, not repair, the submitted tree                                                                           | None                                                                |
| Prettier plus ESLint own frontend checks                     | Accepted | They are the repository's Svelte-aware configured stack; current Biome Svelte support is experimental and duplicates ownership | The selected stack becomes unsupported or demonstrably inferior     |
| Split model-free policy from model integration               | Accepted | Ordinary checks stay small while release evidence still covers actual GiNZA output                                             | The model becomes cheap enough for every default check              |
| Baseline owns non-mutating wrappers and minimal flake checks | Accepted | Its acceptance criteria otherwise depend circularly on Spec 6                                                                  | Full derivations land in Spec 6                                     |
| Preserve boundary-mapped walkthrough examples               | Accepted | Coverage by taught boundary detects loss or concentration that a total prompt count cannot                                      | The repository is no longer used for instruction                    |
| Do not require doctests on orchestration/getters             | Accepted | A per-function quota creates ceremonial examples; integration tests explain lifecycle behavior better                           | Learner feedback identifies a missing executable seam               |
| Replace local `pairwise` with the standard primitive         | Accepted | Python 3.12 is required; teaching the standard API is clearer and explicitly accounts for three removed examples                 | A domain-specific pairing rule appears                              |
| Migrate notebook behavior before making it a consumer        | Accepted | Twelve normalization cases and extraction/exclusion cases must survive removal of duplicate implementations and unittest classes | The notebook needs a genuinely experimental algorithm             |
| Retire the unconsumed generated changelog                    | Accepted | No release automation/configuration consumes it, and its all-unreleased contents are stale                                      | A release process names changelog output as a deliverable            |
