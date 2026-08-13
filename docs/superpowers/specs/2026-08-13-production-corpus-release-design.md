# Production Corpus Release Design

**Date:** 2026-08-13
**Status:** Revised after review
**Scope:** Build and select the first schema-v1 JNLP + Wikipedia artifact

## Outcome

Produce a real immutable artifact containing the 2026 JNLP release and the
legacy-compatible 971-article Wikipedia subset. Validate it, compare it with the
legacy database as one-time evidence, and select it at `deploy/current` only
after the repository owner, Bor Hodošček, has reviewed the release evidence.

The pipeline is the durable product. Exact reproduction of old Pandoc,
sentence-splitter, NLP-model, sentence, or collocation output is not required. A
source, tool, package, model, or policy change creates a new artifact whose
applied inputs are recorded.

## Decisions and Non-goals

- Keep the existing `--splitter-model PATH` builder interface, local `SaT(PATH)`
  loading, model-directory checksum, README command, and hashing test. The
  production build does not replace this explicit input with ambient network or
  Hugging Face cache state.
- Keep the existing eager builder for this bounded two-corpus release. Full-Wikipedia
  streaming, checkpointing, and resumption wait until full-Wikipedia coverage
  becomes a product requirement.
- Treat `data/corpus.db` only as a one-time comparison oracle. It is neither a
  build input nor a recurring publication prerequisite.
- Do not add a report schema that `publish` parses, a previous-pointer protocol,
  a backup system, a retention service, or a TED path for this single release.

## Source Inputs

### JNLP

Use the `2026-06-15` archive marked `ready` in
`docs/corpus-sources.lock.json`, not the local 2020 legacy archive. Acquisition
must verify the locked size `19,166,717` and SHA-256
`8610f8c391634de11a816950008d63675c52e940c6c0c7df29d7ded005547fdb`
before `prepare-jnlp` extracts or converts it.

The adapter dry run records the resulting source count and every rejection
reason before the NLP pass. The count is evidence, not a frozen product quota:
the newer archive and current Pandoc are expected to change it. The one-time
legacy comparison accounts explicitly for all 459 legacy JNLP identities that
remain in the new archive and reports additional, missing, and changed sources.

### Wikipedia

Use only `train-00000-of-00015.parquet` from the locked
`wikimedia/wikipedia` `20231101.ja` revision. Acquisition verifies the size and
SHA-256 already owned by the `wikipedia-ja-20231101` lock entry. Any additional
`--wikipedia-parquet` argument is rejected for this release rather than ignored.

`docs/wikipedia-ja-20231101-subset.json` contains only:

- `sourceLockCorpusId: "wikipedia-ja-20231101"`; and
- `articleIds`: the 971 upstream IDs as JSON strings in shard file order.

Repository, revision, shard, size, and source checksum remain single-owned by
`docs/corpus-sources.lock.json`; the subset file does not duplicate them.

The historical selection was the first 1,000 rows of shard 0 in file order,
keeping rows for which `is_japanese(text, min_length=200)` returns true. The
implementation was removed in commit `8ad9aff`; the predicate remains in
`src/natsume_simple/data.py`. This sentence documents history; the frozen ID
list, not a versioned policy registry, is the production input.

The identity checksum is SHA-256 over exactly these bytes:

```python
json.dumps(article_ids, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
```

There is no BOM, trailing newline, or numeric coercion. The existing lock value
`249dc639f646da4db3711571ea97231a22421c22a5a90ebedf86e7ee3b471991`
is checked using this definition. If the canonical value differs while the 971
unique titles still match the legacy evidence, amend the lock with the newly
derived checksum and a note explaining the former undocumented encoding; do not
fit serialization to the old hash.

The CLI boundary loads the source lock and subset JSON, validates their
relationship, count, uniqueness, and checksum, then passes `article_ids` as a
value to `adapt_wikipedia_parquet`. The adapter uses
`pl.scan_parquet(...).filter(pl.col("id").cast(pl.String).is_in(article_ids)).collect()`
and returns exactly that membership or raises. It need not preserve manifest
order because persistence already sorts by source identity.

## Sentence and Extraction Policy

Preserve the legacy public-content rule: after `sat-3l-sm` segmentation, retain
only sentences for which `is_japanese(sentence, min_length=5)` returns true.
Compose this filter in the builder CLI's `split` callable; keep
`segment_documents` a language-agnostic split/identity transform.

Record the applied sentence filter name and `minLength: 5` in
`identityInputs`, alongside the existing model checksum, `wtpsplit` version,
GiNZA/spaCy versions, extraction policy, builder revision, and execution
profile. This prevents English abstracts, title blocks, reference fragments,
and formula-only fragments from entering examples without hiding a policy
change inside segmentation.

## Acquisition, Dry Run, and Build

Acquisition is a small network-capable CLI operation. It downloads the locked
2026 JNLP archive and the one Wikipedia shard to caller-selected paths, verifies
temporary files, and atomically exposes only valid files. A valid existing file
is reused. Transformation remains local and uses the existing explicitly
provided sentence-splitter model path.

Before the expensive build, `natsume-corpus inspect-inputs` runs both adapters,
validates the Wikipedia subset, and prints source counts plus bounded rejection
counts. The operator selects rejection limits from this evidence and passes
them to `build`; the applied limits and actual counts are retained in the
artifact manifest and release notes. Extraction-limit failure still occurs
after extraction because the missing dependency annotation is not knowable
earlier.

The build emits stage counts and periodic segmentation and extraction progress
so a multi-hour, fixed-profile CPU run is distinguishable from a hung process. This release
does not add checkpoints: interruption restarts the build. The release notes
record start time, finish time, wall-clock duration, peak resident memory, and
host CPU/memory.

## Artifact Stability and Cutover

`create_app` resolves the selected artifact directory once at startup and
validates that resolved directory. Every request in that process opens the same
validated database. A direct server and a container therefore have the same
semantics: changing `deploy/current` does nothing until restart/redeployment.

`publish` remains the small atomic current-pointer operation. For this first
release, the operator records any former artifact instance ID in the release
notes; no `deploy/previous` state is added.

Cutover is:

1. build an immutable artifact;
2. run the structural release check and review operator evidence;
3. record the current artifact ID, if one exists;
4. publish the new pointer;
5. restart/redeploy the service;
6. run readiness and representative browser/API smoke tests against the live
   edge endpoint;
7. on failure, publish the recorded former artifact, restart, and repeat
   readiness.

## Release Checks and Evidence Ownership

`natsume-corpus release-check ARTIFACT --source-lock PATH --wikipedia-subset PATH`
owns cheap structural release assertions separate from API startup validation:

- ordinary artifact validation passes;
- `LICENSE-CONTENT.txt` and `ATTRIBUTION.md` exist and are non-empty;
- corpus IDs are exactly `jnlp` and `wiki`;
- Wikipedia has exactly the 971 expected external IDs;
- manifest source identities and source checksums reconcile with the database;
- rejection counts do not exceed the limits recorded in the manifest.

It exits nonzero on failure and prints a bounded summary. It does not parse
legacy data, run a browser, benchmark HTTP, or produce a report protocol.
`validate_artifact` remains the schema/startup validator and continues to accept
synthetic fixture artifacts.

The release operator owns the remaining one-time evidence in
`docs/releases/<artifact-instance-id>.md`:

- adapter dry-run counts and chosen rejection limits;
- the 971-title and 459-overlap legacy comparisons, clearly marked as one-time
  evidence requiring the untracked legacy database;
- recorded `EXPLAIN` plans for suggestion, noun-collocation, and
  verb-collocation queries; plans are observations, not pass/fail gates;
- the application capacity benchmark; and
- post-restart live edge readiness/browser/API smoke results.

Existing API contract tests own response cardinality and serialized-size limits;
the release process does not duplicate them.

The benchmark uses this fixed request family with both `rankBy` values and both
corpora selected: suggestions for `情報`, noun collocations for `情報`, verb
collocations for `行う`, and examples for the first returned collocation. It
runs directly against the application because the public per-IP edge limit is
intended to reject ten simultaneous requests from one benchmark source. The
report names the host, artifact ID, DuckDB settings, and exact resolved URLs;
requires 100% successful responses with no 5xx; and requires p95 below one
second. The separate live smoke runs through the configured edge.

The human operator runs `publish` last after reviewing this evidence. A
machine-bound report/publish protocol is added only when a second production
artifact creates a recurring consumer for that machinery.

## Content Notices and Public Attribution

Track `corpus-notices/LICENSE-CONTENT.txt` and
`corpus-notices/ATTRIBUTION.md`. They distinguish MIT-licensed software from
corpus content, name ANLP and Wikimedia/Wikipedia, link source and license
pages, identify the snapshot, disclose conversion/segmentation/normalization/
extraction, and provide the correction/takedown contact `dev@bor.space`.

The combined database and corpus-derived distributed content are treated as CC
BY-SA 4.0 as recorded in the source lock. The builder copies the notices into
the artifact. The Svelte page adds a concise footer linking the licenses and
attribution/contact information; the existing Playwright flow asserts that the
public service exposes it.

## Tests and Acceptance

Automated tests cover verified acquisition/reuse and checksum failure; subset
count/duplicate/missing/checksum/shard failures; filtered Parquet membership;
sentence-level Japanese filtering outside `segment_documents`; structural
release-check failures; startup-time symlink resolution; and public attribution.

The ordinary model-free, fixture, browser, package, and scheduled GiNZA gates
remain unchanged. The real corpus run is release evidence, not a default CI
input.

The tranche is complete when:

- the locked 2026 JNLP archive and one Wikipedia shard are verified;
- the committed 971-ID subset reproduces its documented canonical checksum and
  one-time legacy title comparison;
- input inspection passes before NLP work;
- one real two-corpus schema-v1 artifact passes structural release checks and
  the operator evidence above;
- publish plus restart serves exactly the artifact validated at startup;
- live smoke tests pass, rollback instructions name the former artifact if one
  existed, and stable commands are reflected in README's Corpus pipeline, OCI
  image, and Checks sections.

## Lifecycle

This document is implementation scaffolding. After cutover, durable source
facts remain in `docs/corpus-sources.lock.json` and
`docs/corpus-recoverability.md`; measured evidence remains in the release note;
the README sections named above own commands and operations; and this spec is
retired.
