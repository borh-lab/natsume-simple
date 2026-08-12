# Deterministic Corpus Artifact Builder Design

**Status:** Revised draft after written review

**Date:** 2026-08-12
**Boundary:** Offline acquisition, transformation, serving projection,
validation, publication, and rollback

## Problem

Current corpus preparation and pattern extraction mutate one database in place,
can duplicate sources/sentences/derived facts on rerun, use schema APIs that
contradict ID ownership, and mix acquisition, parsing, splitting, model loading,
transformation, persistence, and publication. The serving database retains more
than 13 million token-position rows although the public product searches
collocations and examples rather than arbitrary tokens.

The project permits greenfield replacement, but rebuildability is a claim to
prove rather than a premise. The working tree contains the legacy JNLP archive
but no TED or Wiki source files, and current TED acquisition executes a remote
dataset script that the target policy prohibits. A mandatory recoverability gate
therefore precedes implementation. It may select clean reacquisition, a one-time
legacy conversion, or a documented mixture by corpus; none requires an in-place
migration framework.

## Goals

- Make acquisition and deterministic transformation explicit, separable
  operator commands.
- Build a fresh immutable serving artifact and publish only after validation.
- Preserve source provenance, extraction identity, and rejection evidence.
- Store only the facts required by the accepted public API.
- Make reruns safe by replacement rather than incremental mutation.
- Support fast fixture builds without large models or network access.
- Keep acquisition → adaptation → segmentation → extraction → persistence
  readable as a linear walkthrough with executable examples at its public
  transformation boundaries.

## Gate 3A: Corpus Recoverability Spike

Before schema/builder implementation begins, a bounded spike inventories JNLP,
TED, and Wiki independently and produces `docs/corpus-recoverability.md` plus
machine-readable source-lock candidates. For each corpus it must:

1. identify the exact content currently represented in the legacy database;
2. locate an immutable, legally usable source and acquire a representative
   sample using data-only code with no `trust_remote_code`;
3. prove a local adapter can reproduce stable source identities, ordered text
   units, and representative sentences; and
4. record checksum, revision, license/redistribution status, and the command used.

The spike passes only when every corpus has one of these recorded outcomes:

- **reacquire**: a pinned data-only source and adapter are viable; or
- **convert**: the existing database is the seed of record and the conversion
  path below passes its validation fixture.

Failure to obtain either outcome blocks Spec 3 and data/NLP Cohort 6. Owner
confidence that conversion is acceptable is not evidence that corpus content is
recoverable.

### One-time legacy conversion

`convert-legacy --database <read-only legacy.db> --output inputs/legacy-v1/`
extracts canonical source, sentence, occurrence, span, and corpus-count records
without modifying the source database. The output manifest includes the legacy
database checksum, inspected legacy schema version/fingerprint, conversion
revision, per-table source counts, and explicit fields that could not be
recovered.

Conversion validation reconciles corpus/source/sentence/occurrence counts,
checks every span against sentence text, compares a curated set of noun, verb,
frequency, and example queries against the legacy database, and samples stable
identities for manual review. Before the spike or conversion, the only legacy
database is copied to checksum-verified operator backup storage outside any
builder output or retention path. Conversion is removed only after durable
pinned sources exist for every converted corpus.

## Non-Goals

- Online writes, incremental mutation of the deployed database, distributed
  building, arbitrary token research queries, data lake infrastructure, or a
  general schema migration framework.
- Byte-identical DuckDB files where engine metadata prevents it; semantic
  relational determinism is required.
- Embedding the production database into the default server/container closure.

## Pipeline

```text
acquire --manifest sources.lock.json --output inputs/
       │
       ▼
validated pinned inputs
       │
       ▼
build --inputs inputs/ --output artifacts/<instance-id>.staging/
       │
       ├─► source adapters → SourceDocument
       ├─► segmenter      → SentenceRecord
       ├─► extractor      → CollocationOccurrence
       ├─► persist        → canonical facts + recorded aggregates
       ├─► define view    → filtered collocation frequency
       └─► validate       → manifest + validation report
                                │
                                ▼
publish --artifact artifacts/<instance-id>/ --pointer deploy/current
```

Acquisition is network-capable. Transformation accepts already acquired inputs
and does not fetch datasets, models, or code. Publication is a distinct command
and cannot target a `.staging` or failed artifact.

## Canonical Values

### `SourceDocument`

- `corpus_id: str`: stable lowercase ASCII identifier.
- `external_id: str`: stable identifier in that source collection.
- `title: str`.
- `year: int | None`.
- `author`, `publisher`, `url`: optional strings.
- `text_units: tuple[str, ...]`: ordered source text units.
- `content_sha256: str`.

Identity is `(corpus_id, external_id)`. Display title is not identity.

### `SentenceRecord`

- `source_identity: (corpus_id, external_id)`.
- `ordinal: int`: zero-based order within the source.
- `text: str`: plain Unicode text.

Identity is source identity plus ordinal. Empty sentences are rejected before
construction.

### `CollocationOccurrence`

- Sentence identity.
- `noun`, `particle`, `verb`: non-empty normalized lemmas.
- `noun_span`, `particle_span`, `verb_span`: half-open character offsets into
  the original sentence.
- `extractor_id`: model and extraction-policy identity.

Construction requires a configured particle; in-range, non-empty spans; and
source substrings consistent with extraction evidence. Overlapping spans are
allowed only if an explicit linguistic fixture demonstrates a valid case;
otherwise they are rejected.

These values contain no database-generated IDs. Persistence assigns surrogate
keys only after stable identities are known.

## Source Acquisition Contract

`sources.lock.json` records for each corpus:

- Stable corpus ID and display label.
- Dataset/archive URL or registry identifier.
- Configuration and immutable revision.
- Expected checksum for acquired content or a checksum manifest for a directory.
- Adapter version/configuration.
- License identifier, source URL, and redistribution notes.

Archive downloads are checksum-verified before replacing a validated cache.
Dataset revisions are pinned. Normal acquisition does not execute remote dataset
code; `trust_remote_code=True` is prohibited. If a source has no non-executable
loader under the selected datasets release, Spec 3 remains blocked until an
explicit local adapter or separately audited acquisition tool exists.

License/redistribution status is a publication prerequisite. Unknown license is
not silently represented as permissive; it blocks public artifact publication
for that corpus while still permitting local fixture work.

## Serving Schema Version 1

### `build_metadata`

Exactly one row:

- `schema_version INTEGER NOT NULL` equal to `1`.
- `artifact_instance_id TEXT NOT NULL UNIQUE`.
- `builder_version TEXT NOT NULL`.
- `extractor_id TEXT NOT NULL`.
- `execution_profile_json TEXT NOT NULL`.
- `source_manifest_sha256 TEXT NOT NULL`.
- `built_at_utc TIMESTAMP NOT NULL` for observability.

### `corpus`

- `id TEXT PRIMARY KEY`.
- `label TEXT NOT NULL`.

### `source`

- Surrogate `id` primary key.
- `corpus_id` foreign key.
- `external_id`, `title`, optional metadata, and `content_sha256`.
- Unique `(corpus_id, external_id)`.

### `sentence`

- Surrogate `id` primary key.
- `source_id` foreign key.
- `ordinal` and `text`.
- Unique `(source_id, ordinal)`.

### `collocation_occurrence`

- `sentence_id` foreign key.
- Normalized noun, particle and verb.
- Six span offsets.
- `extractor_id`.
- Unique on sentence, three lemmas, six offsets, and extractor identity.

### Recorded build-time facts

- `corpus_stats` table: exactly one row per corpus with source, sentence, and
  collocation counts.
- `lemma_frequency` table: noun and verb lemma occurrence counts derived from the
  two occurrence roles, unique on `(part_of_speech, lemma)`.
- `collocation_frequency` view: `corpus_id`, noun, particle, verb, and positive
  `raw_frequency`, grouped from `collocation_occurrence`.

`corpus_stats` and `lemma_frequency` are recorded facts about one immutable
build, not mutable caches. The builder writes them once, validates exact equality
against occurrences, then publishes a checksummed read-only file; no refresh
protocol exists or is needed. They avoid full-artifact aggregation on page load,
per-million calculation, and debounced typeahead. `collocation_frequency` stays a
view because every serving query supplies noun/verb and corpus predicates. Its
retention requires `EXPLAIN`/benchmark evidence that the selected DuckDB release
pushes those predicates beneath aggregation and restricts work to the indexed
slice; otherwise the measured design must change before publication.

The schema omits general `lemma`, `word`, and `sentence_word` tables. If
token-level research becomes supported, it receives a separate analytical
artifact and contract.

## Query-Oriented Physical Design

Candidate indexes on base tables align with endpoint predicates:

- Occurrences by `(noun, particle)` and `(verb, particle)`.
- Examples by `(noun, particle, verb)` on occurrence.
- Source by `(corpus_id, id)` and sentence by `(source_id, id)` for corpus joins.
- Suggestions by `(part_of_speech, lemma)` on `lemma_frequency`; the curated
  benchmark confirms whether the selected DuckDB release uses it for the
  substring query before the index is retained.

For the selected DuckDB release, the builder captures `EXPLAIN` output for the
curated query set with and without candidate indexes. An index is retained only
when the plan or measured fixture/production benchmark demonstrates benefit.
Index count is not a success metric.

## Normalization and Aggregation

`corpus_stats.collocation_count` and `lemma_frequency.occurrence_count` are
recorded at build time and reconciled before publication.
`collocation_frequency.raw_frequency` is derived by the filtered view.

Per-million values are not persisted because they are exactly derived from raw
counts and corpus totals:

```text
frequency_per_million = raw_frequency / corpus_stats.collocation_count × 1,000,000
```

Immutability makes post-publication drift impossible. Build-time reconciliation
makes the two recorded tables equivalent to recomputation at the point the file
is sealed.

## Build Provenance and Artifact Layout

`manifest.json` records structured `identity_inputs` containing:

- Sorted source identities and content checksums.
- Source adapter versions/configuration.
- Sentence splitter name, version, model checksum, and configuration.
- NLP model name, package/model checksum, and extraction-policy version.
- Execution backend (`cpu | cuda | rocm`), numeric precision, deterministic
  settings, relevant runtime versions, device class, and all thread/process
  counts that can affect extraction output.
- Serving schema version.
- Builder application revision.

Artifact layout:

```text
artifacts/<artifact-instance-id>/
  corpus.duckdb
  manifest.json
  validation.json
```

`artifact-instance-id` is an operator-readable UTC basic timestamp plus 128 bits
of randomness generated once before staging. It is the only build identifier.
Creation uses an atomic exclusive directory operation and refuses any existing
staging or final path; neither build nor publish overwrites an artifact.

`manifest.json` records the structured identity inputs, instance ID, relation
counts, corpus counts, license/provenance records, and the completed
`corpus.duckdb` SHA-256. The file checksum is not embedded in the database,
avoiding a circular identity. Equivalence checks and operator comparisons use
the recorded fields directly instead of introducing a hash with no independent
consumer. Equivalent instances may have different DuckDB checksums, which
validation reports rather than conceals. The API exposes the instance ID as
`databaseBuildId`.

The initial publishable extraction profile is CPU, float32, deterministic mode,
and fixed thread/process counts. CUDA/ROCm packages may build diagnostic
artifacts with distinct recorded execution profiles, but an accelerated profile
becomes publishable only after repeated fixture and representative-sample builds
prove identical ordered relational exports under its declared deterministic
settings. Resolver success or a single extraction smoke test is not publication
evidence.

## Validation

Publication requires all of the following:

- Exactly one compatible `build_metadata` row.
- All foreign keys and stable identity uniqueness constraints hold.
- Every sentence belongs to exactly one source/corpus.
- Every occurrence references a sentence and has valid spans.
- Every normalized lemma/particle is non-empty and every particle is configured.
- Recorded corpus and lemma counts are non-negative and equal exact
  recomputation from canonical relations; the collocation-frequency view returns
  non-negative counts and references only canonical relations.
- Every selected corpus has at least one source, sentence, and occurrence.
- Representative noun/verb/example fixture queries match expected relations.
- The Spec 2 API integration suite passes against the built artifact.
- Two small fixture builds from the same inputs and execution profile produce
  different artifact instance IDs, equal structured `identity_inputs`, and
  identical ordered relational exports after excluding observational fields.
  Changing an identity input, including backend/precision/thread count, changes
  the corresponding recorded field.
- The certified CPU publication profile passes the repeated-build comparison;
  accelerated profiles fail publication until they independently pass it.

Adapters emit bounded reason counts for rejected sources, text units, sentences,
and occurrences. Configuration defines both an absolute and percentage maximum
per reason/corpus. Crossing either threshold fails the build. Logs and reports
contain identifiers and counts, never corpus text.

## Publication and Rollback

The builder writes the exclusively created `<instance-id>.staging`,
fsyncs/finishes files as supported by the platform, validates, then renames to
the absent final instance directory. A pre-existing staging or final instance
path is a hard failure. It never opens the deployed artifact for writing.

Publication validates the final directory again, then atomically changes an
explicit `deploy/current` pointer and restarts the single server. The server
resolves the pointer only at startup. Rollback restores the prior pointer and
restarts. `list-artifacts` reads flat-directory manifests and can compare or
group instances by their recorded input fields. An explicit retention command
sorts all instances by `built_at_utc`, protects `deploy/current`, and removes
only non-current instances beyond the overall configured keep count; removal is
not part of publication.

Partial corpus success exits non-zero. Building an intentional subset requires
the operator to explicitly select that subset, producing a different source
manifest and recorded provenance.

## Failure Semantics

- Acquisition failure does not replace validated cached input.
- Transformation or validation failure leaves a diagnostic staging directory
  but no publishable final artifact.
- Publication failure leaves `deploy/current` on the previous artifact.
- Existing staging/final instance targets are never replaced, even when all
  recorded identity inputs match.
- A server incompatible with the selected schema remains unready rather than
  attempting migration.
- Rebuilding never deletes or modifies the current production artifact.

## Test Strategy

- Adapter contract tests for every source using tiny local fixtures.
- Property tests for stable identity, span validation, aggregate reconciliation,
  view definitions, and structured provenance comparison.
- Fixture end-to-end build with no network/model download.
- Failure-injection tests at acquisition, transformation, validation, rename,
  pointer swap, and server restart boundaries.
- Duplicate-input tests prove base-table uniqueness failures are loud.
- Compatibility probes against selected datasets/GiNZA/spaCy/wtpsplit versions.
- Recoverability-gate tests for data-only acquisition and the legacy conversion
  reconciliations before real corpus replacement.
- Production-like benchmark records build duration, peak memory, artifact size,
  rejection counts, and curated query plans without making an optimization claim
  before measurement.
- Doctests walk fixture-sized canonical values through adaptation, segmentation,
  occurrence construction, recorded/derived frequency, and build provenance.

## Acceptance Criteria

- Gate 3A records a passing reacquire or convert path for JNLP, TED, and Wiki
  before builder implementation or legacy database replacement begins.
- A fixture artifact can be built twice into distinct immutable instance paths
  with equal structured identity inputs and relational contents.
- The built artifact passes all schema, protocol, and query fixture validations.
- Re-running or republishing never duplicates persisted facts.
- No normal transformation command uses network access or remote code execution.
- Failed builds/publications cannot change the artifact used by the server.
- The serving database contains no general token tables.
- Manifest provenance and license fields are complete for every publicly
  published corpus.
- Operators can publish and roll back by selecting immutable artifact versions.
- A learner can run the builder walkthrough examples and trace the corresponding
  implementation without following a framework or repository hierarchy.

## Decision Log

| Decision                                         | Status   | Reason                                                                                                              | Revisit trigger                                                      |
| ------------------------------------------------ | -------- | ------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| Whole immutable rebuild                          | Accepted | Greenfield, simple rollback, eliminates duplicate/migration state                                                   | Build time exceeds operational window                                |
| Separate acquisition from transformation         | Accepted | Reproducibility and remote-code control                                                                             | Inputs cannot legally/technically be cached                          |
| Serve a projection, not token graph              | Accepted | Public consumers need collocations/examples only                                                                    | Token research becomes a supported product                           |
| Derive per-million values at query time          | Accepted | Prevents denominator drift                                                                                          | Measured query cost is material                                      |
| Publish via versioned directory pointer          | Accepted | Database and manifest switch together and roll back cheaply                                                         | Deployment filesystem cannot provide atomic pointer replacement      |
| Gate replacement on corpus recoverability        | Accepted | TED/Wiki source material is absent locally and the target datasets stack cannot execute the incumbent remote loader | All corpora have durable pinned data-only sources                    |
| Use one artifact instance identity                | Accepted | Structured manifest fields already support equivalence comparison; a second hashed identity has no independent consumer | A real consumer requires a compact equivalence key                 |
| Record the execution profile as provenance        | Accepted | Device, precision, and concurrency can change NLP extraction results                                                | Extraction becomes proven invariant across profiles                  |
| Materialize hot, artifact-wide aggregate facts    | Accepted | Immutable builder-written corpus/lemma counts avoid repeated full scans; filtered collocation aggregation stays a view | Benchmarks show either recorded table is unnecessary               |
| Use flat instance directories                    | Accepted | Instance IDs are globally unique; flat layout makes listing and retention one-dimensional                           | Artifact volume requires a measured sharding strategy                |
