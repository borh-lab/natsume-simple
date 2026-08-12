# Natsume Simple Modernization Specifications

**Status:** Revised draft after written review

**Date:** 2026-08-12
**Decision owner:** Repository owner

## Objective

Turn Natsume Simple into a reproducible, publicly hosted, anonymous read-only
search service backed by an immutable DuckDB corpus artifact without losing its
role as an executable teaching repository. The work also
establishes meaningful quality gates, simplifies the Svelte application,
upgrades supported dependency majors, and makes Nix the authoritative build and
deployment interface.

This is a greenfield modernization. A recoverability gate must first prove that
each corpus can be reacquired under the new no-remote-code policy or converted
once from the legacy database. After that gate, the current database may be
replaced rather than migrated in place, and the FastAPI and Svelte contracts may
change atomically. There are no external API consumers to preserve.

## Product Boundary

- The production service is a publicly hosted single instance.
- Public behavior is anonymous and read-only.
- Corpus acquisition, NLP processing, and database publication are offline
  operator actions.
- The server consumes a validated immutable database artifact and never mutates
  it.
- Nix defines development, testing, packaging, and production artifacts.
- An OCI image is derived from the Nix server package rather than maintained as
  an independent Docker build.
- The backend walkthrough remains readable in source order and its executable
  examples remain a supported learning interface.

## Actors and Use Cases

| Actor                 | Objective                                                       | Current obstacle                                                                   | Capability after modernization                                                              |
| --------------------- | --------------------------------------------------------------- | ---------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| Public visitor        | Find Japanese noun/particle/verb relations and inspect examples | Inconsistent API types, unbounded work, weak failure UI, and unsafe text rendering | Bounded search with interpretable frequencies, corpus filters, examples, and visible errors |
| Corpus operator       | Acquire sources and publish a new corpus safely                 | In-place mutation, duplicate risk, unclear provenance, and no atomic publication   | Reproducible offline build, validation report, immutable publication, and pointer rollback  |
| Developer             | Change backend/frontend with fast feedback                      | Red baseline and checks that omit or mutate important surfaces                     | Fixture-backed, non-mutating checks covering types, protocol, behavior, and build outputs   |
| Dependency maintainer | Upgrade major versions without losing domain behavior           | One broad environment and trivial tests obscure compatibility regressions          | Independently verified compatibility cohorts and accelerator evidence                       |
| Service operator      | Run and roll back one public instance                           | Runtime setup scripts and no minimal production artifact                           | Nix server package and derived non-root OCI image consuming a read-only artifact            |
| Student or instructor | Read, run, and explain the corpus-to-query path                  | Production concerns can obscure the walkthrough and silently erase doctests        | Small modules with executable examples at each taught transformation boundary                |

## Glossary

| Term                       | Definition                                                                                                            |
| -------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| Corpus input               | Pinned, checksum-validated source material consumed by the offline builder                                            |
| Serving artifact           | Versioned directory containing `corpus.duckdb`, `manifest.json`, and validation evidence                              |
| Artifact instance ID       | Unique identity of one artifact execution; it names the flat directory and appears on the wire                        |
| Identity inputs            | Structured source, transformation, model, execution, schema, and builder provenance recorded in the artifact manifest |
| Database file checksum     | Integrity hash of the completed DuckDB file                                                                           |
| Occurrence                 | One extracted noun-particle-verb relation tied to a sentence and source spans                                         |
| Raw frequency              | Count of occurrences for a collocation in a corpus                                                                    |
| Frequency per million      | Raw frequency divided by that corpus's total collocation count, multiplied by 1,000,000                               |
| Mean frequency per million | Arithmetic mean of the selected corpora's per-million rates, including zero for a selected corpus with no occurrence  |
| Characterization test      | Test that records current required behavior before a structure-only change                                            |
| Compatibility cohort       | Dependency set upgraded, reviewed, verified, and reverted as one unit                                                 |

## Specification Set

| Order | Specification                                                                                 | Outcome                                                                                         | Depends on                                                          |
| ----- | --------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| 1     | [Executable Quality Baseline](./2026-08-12-quality-baseline-design.md)                        | Truthful, non-mutating green checks and characterization rails                                  | None                                                                |
| 2     | [Safe Public Service and Typed API](./2026-08-12-public-service-api-design.md)                | Bounded anonymous API, safe rendering contract, and explicit runtime ownership                  | Spec 1                                                              |
| 3     | [Deterministic Corpus Artifact Builder](./2026-08-12-corpus-builder-design.md)                | Recoverability gate plus rebuildable immutable search projection with provenance and validation | Spec 1; must satisfy Spec 2 fixture contract                        |
| 4     | [Major-Version Dependency Migration](./2026-08-12-dependency-migration-design.md)             | Supported modern dependency cohorts with reproducible lockfiles                                 | Frontend phase: Spec 1; serving/data phases: Specs 2/3 respectively |
| 5     | [Frontend State and Component Simplification](./2026-08-12-frontend-simplification-design.md) | Page-scoped state, server-ranked selection, typed components, and accessible behavior           | Specs 1 and 2; Spec 4 frontend cohorts                              |
| 6     | [Nix Packages and OCI Image](./2026-08-12-nix-delivery-design.md)                             | Real derivations for frontend, server, builder, checks, and container                           | Specs 1–5                                                           |

Specs 2 and 3 may be developed in parallel against the same fixture schema, but
Spec 3 implementation cannot begin until its corpus recoverability gate passes,
and production cutover requires both. Dependency Spec 4 is phased: frontend
cohorts 1–4 run after the baseline and before frontend Spec 5; serving and NLP
cohorts wait for their protocol and corpus characterization. Final Nix packaging
follows stable application closures. Spec 1 creates the minimal non-mutating
flake checks and shell behavior that later specs extend.

## Specification Lifecycle

These documents are implementation scaffolding, not permanent prerequisite
reading. When the final acceptance criteria land, durable choices collapse into
a short ADR set, the learning/walkthrough contract and current commands move to
`README.md` and `AGENDA.md`, and this superseded working set is retired with a
completion record linking the implementation commits and surviving ADRs.

## Written Review Disposition

| Finding                                                         | Resolution                                                                                                                     | Owning specification |
| --------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------ | -------------------- |
| Selection-specific top N was impossible after global truncation | `corpusId` and `rankBy` are server inputs; selection/ranking precede limiting and UI toggles refetch                           | 2 and 5              |
| Equivalent builds collided on one directory                     | Flat unique no-overwrite instance directories; structured provenance and execution profile remain manifest metadata            | 3                    |
| Rebuildability was assumed                                      | Gate 3A requires a passing reacquire or validated conversion outcome for every corpus                                          | 3                    |
| Large NLP model had no check owner                              | Default checks use model-free observations; an explicit locked model derivation is a release/scheduled gate                    | 1 and 6              |
| Formatter ownership was undecided                               | Prettier formats and ESLint lints; duplicate Biome ownership is removed                                                        | 1 and 4              |
| Baseline and Nix specs owned the same mutation cleanup          | Spec 1 owns non-mutating wrappers/shell and minimal checks; Spec 6 owns derivations and closures                               | 1 and 6              |
| Baseline perfected types and tooling later replaced             | Spec 1 uses narrow transitional types; frontend cohorts precede the component refactor                                         | 1, 4, and 5          |
| Public abuse control was absent                                 | Bounded application admission plus a rate/connection-limited reverse proxy are release prerequisites                           | 2 and 6              |
| Aggregate rate name/meaning was ambiguous                       | Per-corpus `frequencyPerMillion` and selected `meanFrequencyPerMillion` are distinct                                           | 2                    |
| p95 equalled timeout and cancellation was unspecified           | p95 is below 1 second; an event-loop timer interrupts and the request discards its local connection                            | 2                    |
| Determinism criterion partly restated ID construction           | Repeated builds compare ordered relational exports and structured input fields independently                                   | 3                    |
| Educational purpose was absent                                  | Student/instructor is an actor; walkthrough readability and executable examples are global constraints                          | 1–6                  |
| Example cache omitted corpus selection                          | Cross-result caching is removed; component keys follow corpus/example identity while rank-only reorders preserve state           | 5                    |
| Persisted aggregates created avoidable drift                    | Immutable build-time corpus/lemma facts are reconciled once; only filtered collocation frequency remains a view                 | 3                    |
| Identity nesting complicated retention and cache invalidation   | Artifact directories are flat and the instance ID is the wire `databaseBuildId`                                                 | 2 and 3              |
| Semantic build hash lost its structural consumers               | Structured manifest inputs serve equivalence comparison directly; the instance ID remains the only build identifier            | 3                    |
| Small dead helpers and duplicate wrappers remained              | Spec 1 deletes dead seed/filter/route surfaces and uses stdlib pairing/logging; the live normalization helper waits for Spec 2  | 1 and 2              |
| Notebook forked the extraction contract                         | Its distinct normalization/extraction cases migrate first; then it imports the tested module and retains exploration           | 1                    |
| JNLP conversion relied on ambient executables                    | Gate 3A runs `nkf`/`pandoc` end-to-end; Spec 6 includes both in the declared builder closure                                    | 3 and 6              |
| Generated changelog had no release consumer                     | The stale file and `git-cliff` are removed until a release process names a changelog deliverable                                | 1                    |
| Development entry points duplicated environment ownership       | Default Codespaces delegates to Nix; Dockerfile/rootless retire; in-flight ROCm survives only as Cohort 7's tested evidence harness | 4 and 6           |

Review follow-ups also assign legacy static deletion to Spec 1, acknowledge the
incumbent prerelease manifest range, require cumulative intermediate-major
migration review, and permit only coarse query-length buckets in logs.

## System Flow

```text
Pinned corpus inputs
        │
        ▼
Offline corpus builder ──► versioned artifact instance directory
                              ├── corpus.duckdb
                              └── manifest.json
                                        │
                                        ▼ read-only
Browser ──► rate-limited reverse proxy ──same origin──► FastAPI ──► DuckDB
   ▲                         │
   └──── static Svelte app ──┘

Nix flake ──► frontend package
          ├─► server package
          ├─► corpus-builder packages
          ├─► checks
          └─► OCI image derived from server package
```

## Global Constraints

- Security/correctness changes and structural refactors are separate review
  units.
- Characterization tests land before behavior-preserving deletion.
- Every new abstraction, cache, persisted derivative, identity, or workflow names
  its present consumer. Prefer a language/runtime/database primitive and require
  measurement before adding performance machinery.
- Modules on the documented backend walkthrough remain readable end-to-end.
  Executable examples are a maintained interface, not incidental comments; an
  example may disappear only with the behavior it teaches or an explicit
  teaching-contract decision.
- Generated lockfile changes never share a commit with unrelated cleanup.
- Each dependency cohort is independently locked, verified, and reviewable.
- No production build or test requires the full production corpus unless it is
  explicitly a scheduled corpus compatibility job.
- Corpus content, raw search text, and sentence text do not enter logs. Search
  diagnostics may record only a coarse query-length bucket.
- A merge leaves all declared checks green and does not mutate the checkout.

## Shared Success Criteria

- A clean checkout can run the complete default check suite without a production
  database or large NLP model. Model-dependent compatibility is a separate
  declared release/scheduled gate with a pinned model closure.
- A fresh corpus artifact can be built, validated, published, served, and rolled
  back without in-place database mutation.
- Every valid public request is bounded by input, result count, response size,
  query timeout, and server resource configuration.
- Corpus text is never interpreted as HTML.
- Frontend and backend share a checked OpenAPI-derived contract.
- The minimal production server and OCI image contain no notebook, Node.js,
  corpus acquisition, NLP model, Torch, CUDA, or ROCm closure.
- A learner can follow acquisition, transformation, persistence, and serving in
  source order and run the executable examples without the production corpus.

## Shared Revisit Triggers

- More than one server instance requires a new review of database connection,
  artifact rollout, and availability ownership.
- Online corpus writes require a new persistence and synchronization design.
- An external API consumer requires explicit compatibility/versioning policy.
- A consumer needing more than 200 collocations per particle triggers cursor
  pagination design.
- A supported token-level research workflow triggers a separate analytical
  artifact rather than widening the serving schema.
- Embedding corpus data in an OCI image triggers a separate, explicitly
  identified image variant.
- Removing or materially loosening edge request/connection limits requires a
  new public-abuse and capacity review.

## Classified Architecture Review

**Mitigated by the specifications**

- Stored-content XSS through structured escaped rendering.
- Shared thread-unsafe database state through lifespan ownership and
  request-local read-only connections.
- Silent duplicate ingestion through immutable replacement and uniqueness.
- Implicit wire shapes through Pydantic/OpenAPI and checked TypeScript types.
- Braided frontend state through a page controller, small presentation math, and
  typed presentation interfaces.
- Environment drift through committed locks and real Nix derivations.

**Accepted tradeoffs**

- The initial service has single-instance availability and no failover.
- Anonymous public access relies on an operator-managed reverse proxy with
  request-rate and connection limits plus the application's bounded query
  concurrency; the application does not implement accounts or distributed rate
  limiting.
- Whole-artifact rebuilding is preferred over incremental mutation.
- Results are limited to 200 collocations per particle until a present consumer
  justifies pagination.
- CPU is the mandatory full CI path; accelerator claims require scheduled
  matching-hardware evidence rather than every-commit full runs.
- Filtered collocation aggregation remains a view; recorded corpus and lemma
  facts keep artifact-wide scans off page-load, normalization, and typeahead
  paths. The production-host benchmark verifies that split.

**Blocking inputs owned by later specs**

- Spec 2 records the production host class and proves memory/query limits with
  its curated benchmark before public release.
- Spec 3 first proves reacquisition or a validated legacy conversion for every
  corpus, then records license/redistribution status and resolvable immutable
  source revisions before publishing each corpus.
- Spec 4 selects and proves the exact common CPU/CUDA/ROCm PyTorch matrix before
  advertising accelerator support.

## Decision Log

| Decision                                              | Status   | Date       | Reversibility | Evidence / reason                                                                                                                               | Revisit trigger                                                             |
| ----------------------------------------------------- | -------- | ---------- | ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------- |
| Use six capability specs with explicit phase gates    | Accepted | 2026-08-12 | Easy          | Separates defects, protocol, data lifecycle, refactor, upgrades, and packaging without forcing unsafe total ordering                            | Specs repeatedly require coupled changes                                    |
| Replace the database only after a recoverability gate | Accepted | 2026-08-12 | Moderate      | Greenfield replacement is acceptable, but TED/Wiki inputs are not presently available in the working tree and remote dataset code is prohibited | Every corpus has a durable pinned source and conversion fallback is retired |
| Rank over the requested corpus set and metric         | Accepted | 2026-08-12 | Moderate      | Client-side refinement of a globally truncated page omits valid top results                                                                     | Cursor pagination or a new analytical ranking is required                   |
| Change frontend/backend atomically                    | Accepted | 2026-08-12 | Moderate      | Owner confirmed no external API consumers                                                                                                       | First external consumer appears                                             |
| Target an anonymous read-only single instance         | Accepted | 2026-08-12 | Moderate      | Owner confirmed intended runtime                                                                                                                | Authentication, writes, or horizontal scaling is required                   |
| Make Nix authoritative                                | Accepted | 2026-08-12 | Moderate      | Owner selected Nix for production                                                                                                               | Deployment platform cannot consume Nix outputs                              |
| Derive OCI from Nix server package                    | Accepted | 2026-08-12 | Easy          | Prevents two competing production definitions                                                                                                   | Measured OCI constraints require a specialized builder                      |
| Preserve the executable teaching walkthrough         | Accepted | 2026-08-12 | Moderate      | `AGENDA.md`, module doctests, pytest configuration, and Codespaces setup identify learning as a present consumer                                | The repository is no longer used for instruction                            |
