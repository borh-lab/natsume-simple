# Production Corpus Release Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build, verify, publish, and serve the first schema-v1 production artifact from the locked 2026 JNLP archive and the frozen 971-article Japanese Wikipedia subset.

**Architecture:** Keep acquisition, adaptation, immutable artifact construction, startup validation, and pointer publication as separate operations. A small release-input module owns lock/subset parsing and byte verification; the existing adapters continue to accept values; a separate release checker owns first-release structural policy without changing the generic API startup validator.

**Tech Stack:** Python 3.12, pytest, DuckDB, Polars, FastAPI, Svelte 5, Playwright, Nix flakes, `nkf`, Pandoc, SaT/wtpsplit, spaCy/GiNZA.

## Global Constraints

- Preserve the user-owned untracked paths: `AGENDA.md`, `container.nix`, `data/`, and `svelte-llms-short.txt`.
- Keep `--splitter-model PATH`; do not introduce model download, cache, bundle, or tokenizer protocols.
- Keep `validate_artifact` generic enough for fixture artifacts. Release-only policy belongs in `release_check.py`.
- The production artifact contains exactly `jnlp` and `wiki`; TED remains synthetic-fixture-only.
- The pipeline, source lock, frozen subset, notices, and release evidence are durable. The legacy database is optional one-time comparison evidence.
- Use test-first commits. Run the focused red test before implementation and the focused green test after it.
- Do not add a report schema parsed by `publish`, a previous-pointer protocol, checkpointing, a backup system, or a benchmark framework.

---

### Task 0: Expose the repository's existing test and release environments

**Files:**

- Modify: `flake.nix`

- [ ] Add a `test` development shell backed by the already-defined `testPython` environment and a small release-operator shell.

```nix
test = pkgs.mkShell {
  packages = [
    testPython
    pkgs.mypy
    pkgs.ruff
  ];
};

release = pkgs.mkShell {
  packages = [
    corpusBuilder
    server
    pkgs.curl
    pkgs.nginx
    pkgs.time
  ];
};
```

The `test` shell is the one environment for focused Python tests and static checks. It selects the CPU dependency set, contains backend, builder, and test extras together, and does not create `.venv` or resolve packages from the network. The `release` shell provides only packaged operator commands plus the reference edge and measurement tools; it does not create a second build implementation.

- [ ] Verify the environment before using it elsewhere in the plan.

```bash
nix develop .#test --command python -c 'import fastapi, polars, spacy, torch, pytest, httpx2'
nix develop .#test --command mypy --version
nix develop .#test --command ruff --version
nix develop .#release --command sh -c 'command -v natsume-corpus; command -v natsume-serve; command -v nginx; command -v time; command -v curl'
```

- [ ] Commit.

```bash
git add flake.nix
git commit -m "build: expose hermetic test and release shells"
```

### Task 1: Pin one validated artifact per server process

**Files:**

- Modify: `src/natsume_simple/api.py`
- Modify: `tests/test_api_contract.py`

- [ ] Add a failing lifespan regression test.

Build two distinguishable fixture artifacts, create `deploy/current` pointing at the first, start `create_app(deploy / "current")`, and replace the symlink with the second after startup. Assert readiness still reports the first artifact ID and `/api/corpora` still reads the first database's distinct corpus label/count. The readiness assertion is the stale-metadata half of the regression and already passes before the fix; the distinct `/api/corpora` assertion is what must fail red when request connections follow the swapped symlink.

- [ ] Run the focused test and confirm it fails because request connections follow the changed symlink.

```bash
nix develop .#test --command pytest tests/test_api_contract.py -k startup_resolves_artifact_once -q
```

- [ ] Resolve before validation in `artifact_lifespan`.

```python
selected_artifact = artifact_dir.resolve(strict=True)
database_path, manifest = validate_artifact(selected_artifact)
```

Catch `OSError` alongside `ArtifactValidationError`, log a bounded `artifact_path_unresolvable` reason, and leave readiness unavailable. Store only the database path returned from the resolved directory.

- [ ] Run the focused test, then the API contract suite.

```bash
nix develop .#test --command pytest tests/test_api_contract.py -q
```

- [ ] Commit.

```bash
git add src/natsume_simple/api.py tests/test_api_contract.py
git commit -m "fix: pin the validated artifact at startup"
```

### Task 2: Give release source facts one code owner

**Files:**

- Create: `src/natsume_simple/release_inputs.py`
- Create: `tests/test_release_inputs.py`

- [ ] Write failing tests for the production source lock and subset contracts.

Cover:

- the ready `jnlp` candidate resolves to the locked URL, size, and SHA-256;
- Wikipedia resolves only `train-00000-of-00015.parquet` from the lock entry;
- the subset references `wikipedia-ja-20231101`;
- article IDs are strings, unique, and exactly 971 entries;
- canonical checksum uses compact UTF-8 JSON with no sort/reformat step;
- wrong count, duplicate IDs, unknown source entry, wrong checksum, or an extra shard raises a bounded `ReleaseInputError` reason.

Use temporary fixture JSON in unit tests; the committed real subset is added in Task 4.

- [ ] Run the tests and confirm import/contract failures.

```bash
nix develop .#test --command pytest tests/test_release_inputs.py -q
```

- [ ] Implement only the shared values and checks.

```python
@dataclass(frozen=True)
class LockedFile:
    source_lock_corpus_id: str
    name: str
    local_name: str
    url: str
    size: int
    sha256: str


@dataclass(frozen=True)
class ReleaseSources:
    jnlp_archive: LockedFile
    wikipedia_shard: LockedFile
    wikipedia_identity_sha256: str


@dataclass(frozen=True)
class WikipediaSubset:
    source_lock_corpus_id: str
    article_ids: tuple[str, ...]
```

Expose:

```python
load_release_sources(source_lock: Path) -> ReleaseSources
canonical_article_ids_sha256(article_ids: Sequence[str]) -> str
load_wikipedia_subset(path: Path, *, sources: ReleaseSources) -> WikipediaSubset
validate_wikipedia_paths(paths: Sequence[Path], *, sources: ReleaseSources) -> Path
verify_file(path: Path, locked: LockedFile) -> None
```

The lock remains the sole owner of repository/revision/URL/size/checksum. `local_name` is derived while loading: the Wikipedia shard keeps its locked name, and JNLP uses `NLP_LATEX_CORPUS-<observedRelease>.zip`. The subset owns only `sourceLockCorpusId` and `articleIds`.

- [ ] Run focused tests and static checks.

```bash
nix develop .#test --command pytest tests/test_release_inputs.py -q
nix develop .#test --command mypy --ignore-missing-imports --show-error-context src/natsume_simple/release_inputs.py
nix develop .#test --command ruff check src/natsume_simple/release_inputs.py tests/test_release_inputs.py
nix develop .#test --command ruff format --check src/natsume_simple/release_inputs.py tests/test_release_inputs.py
```

- [ ] Commit.

```bash
git add src/natsume_simple/release_inputs.py tests/test_release_inputs.py
git commit -m "feat: validate locked release inputs"
```

### Task 3: Acquire the two locked files atomically

**Files:**

- Modify: `src/natsume_simple/release_inputs.py`
- Modify: `src/natsume_simple/builder_cli.py`
- Modify: `tests/test_release_inputs.py`
- Modify: `tests/test_builder_cli.py`

- [ ] Add failing tests for download, reuse, and rejection.

Use an injected byte-stream opener in unit tests. Prove that acquisition:

- writes to a temporary sibling before verification;
- exposes the final filename only after size and SHA-256 pass;
- reuses a valid existing file without opening the network;
- rejects and removes an invalid temporary file;
- rejects an invalid pre-existing destination rather than overwriting it.
- opens the source with the explicit 60-second socket timeout.

- [ ] Add the CLI parser test.

The command is:

```text
natsume-corpus acquire-release-inputs \
  --source-lock docs/corpus-sources.lock.json \
  --output-directory data/release-inputs
```

It derives the JNLP local name from the lock's `observedRelease` as `NLP_LATEX_CORPUS-<observed-release>.zip`, preserves the locked Wikipedia shard name, and prints both paths. The release date is not repeated as a Python constant.

- [ ] Confirm the focused tests fail.

```bash
nix develop .#test --command pytest tests/test_release_inputs.py tests/test_builder_cli.py -k acquire -q
```

- [ ] Implement `acquire_locked_file` with `urllib.request.urlopen(..., timeout=60)`, a unique `.part` sibling, streaming SHA-256/byte count, `os.replace`, and cleanup in `finally`.

Do not add retry, resumable-download, cache, mirror, or parallel-download abstractions. A rerun reuses a fully valid destination.

- [ ] Wire `acquire-release-inputs` at the CLI boundary and convert `ReleaseInputError`, `OSError`, and `FileExistsError` into `parser.error` messages without tracebacks.

- [ ] Run focused tests and the builder smoke-level Python suite.

```bash
nix develop .#test --command pytest tests/test_release_inputs.py tests/test_builder_cli.py -q
```

- [ ] Commit.

```bash
git add src/natsume_simple/release_inputs.py src/natsume_simple/builder_cli.py tests/test_release_inputs.py tests/test_builder_cli.py
git commit -m "feat: acquire locked corpus sources"
```

### Task 4: Freeze and enforce the 971-article Wikipedia subset

**Files:**

- Create: `docs/wikipedia-ja-20231101-subset.json`
- Modify: `src/natsume_simple/corpus_pipeline.py`
- Modify: `src/natsume_simple/builder_cli.py`
- Modify: `tests/test_corpus_adapters.py`
- Modify: `tests/test_builder_cli.py`
- Modify: `tests/test_release_inputs.py`

- [ ] Acquire the verified real shard if it is not already present.

```bash
nix run .#build-corpus -- acquire-release-inputs \
  --source-lock docs/corpus-sources.lock.json \
  --output-directory data/release-inputs
```

- [ ] Generate the committed subset once from the documented historical rule.

In a temporary one-off Python program, read only the first 1,000 rows of the verified shard in file order, keep rows where `is_japanese(text, min_length=200)`, convert `id` to strings, assert 971 unique IDs, and serialize:

```json
{
  "sourceLockCorpusId": "wikipedia-ja-20231101",
  "articleIds": ["..."]
}
```

The file may be pretty-printed; its identity is the compact serialization of the `articleIds` value, not the bytes of the file.

- [ ] Compare the selected titles to the present legacy oracle before accepting or changing the identity checksum.

```sql
SELECT title FROM source WHERE corpus = 'Wikipedia' ORDER BY title
```

For this first release, `data/corpus.db` is present and the 971-title comparison is required evidence: stop if the titles differ. Only after all 971 titles match may the canonical ID checksum replace the currently locked undocumented value; update only `orderedIdentityListSha256` in `docs/corpus-sources.lock.json` and add a nearby explanatory note field. Future rebuilds use the committed subset without requiring the legacy database.

- [ ] Add failing adapter tests.

Change the adapter contract to:

```python
adapt_wikipedia_parquet(path: Path, *, article_ids: Collection[str]) -> AdaptationResult
```

Test that it returns exactly the requested IDs, rejects a missing requested ID, rejects duplicate rows for a requested ID, and does not return unselected rows.

- [ ] Confirm the focused tests fail.

```bash
nix develop .#test --command pytest tests/test_corpus_adapters.py tests/test_release_inputs.py tests/test_builder_cli.py -k wikipedia -q
```

- [ ] Implement the lazy filtered read.

```python
selected = (
    pl.scan_parquet(path)
    .select("id", "url", "title", "text")
    .filter(pl.col("id").is_in(list(article_ids)))
    .collect()
)
```

Compare the returned ID multiset with the requested set before constructing documents. Continue sorting adapted documents by `external_id`; do not promise manifest order from the adapter.

- [ ] Update the only production caller in the same commit.

Add `--source-lock` and `--wikipedia-subset` build options. When Wikipedia is present, `builder_cli._build` requires them, loads the validated subset once, validates that exactly one Wikipedia path was supplied, and calls `adapt_wikipedia_parquet(path, article_ids=subset.article_ids)`. Keep the build command type-correct and runnable at the end of Task 4; Task 5 adds checksum validation, inspection, and recorded policy metadata.

- [ ] Validate the committed real subset against the real source lock in a non-network test.

```bash
nix develop .#test --command pytest tests/test_release_inputs.py tests/test_corpus_adapters.py tests/test_builder_cli.py -q
nix develop .#test --command mypy --ignore-missing-imports --show-error-context src
```

- [ ] Commit.

```bash
git add docs/wikipedia-ja-20231101-subset.json docs/corpus-sources.lock.json src/natsume_simple/corpus_pipeline.py src/natsume_simple/builder_cli.py tests/test_corpus_adapters.py tests/test_builder_cli.py tests/test_release_inputs.py
git commit -m "feat: freeze the production Wikipedia subset"
```

### Task 5: Inspect inputs and build with the recorded content policy

**Files:**

- Modify: `src/natsume_simple/builder_cli.py`
- Modify: `src/natsume_simple/corpus_pipeline.py`
- Modify: `src/natsume_simple/artifact_builder.py`
- Modify: `tests/test_builder_cli.py`
- Modify: `tests/test_corpus_adapters.py`
- Modify: `tests/test_artifact_builder.py`

- [ ] Add failing CLI tests for production input validation.

The `build` command uses the release metadata options introduced in Task 4:

```text
--source-lock docs/corpus-sources.lock.json
--wikipedia-subset docs/wikipedia-ja-20231101-subset.json
--wikipedia-parquet data/release-inputs/train-00000-of-00015.parquet
```

Test rejection before model loading for a wrong Wikipedia shard name/checksum, extra shard, or invalid subset. Preserve the existing `--splitter-model PATH` test. The prepared JNLP directory is operator-produced transformation output; only acquisition claims byte verification of its source archive.

- [ ] Add failing `inspect-inputs` tests.

The command accepts `--source-lock`, `--wikipedia-subset`, `--jnlp-root`, and one `--wikipedia-parquet`. It runs both adapters without SaT, spaCy, or GiNZA and prints a JSON object containing each corpus's accepted source count and bounded rejection counts.

- [ ] Add a failing sentence-policy test at the CLI split boundary.

Extract a small pure helper:

```python
split_japanese_sentences(text_units, *, splitter) -> Iterable[str]
```

It strips paragraphs, delegates to the supplied splitter, and retains only `is_japanese(sentence, min_length=5)`. Do not change `segment_documents`.

Import `is_japanese` inside this helper. `data.py` imports Polars, Torch, and wtpsplit at module import time, so a module-level import would make light commands such as `natsume-corpus current` load the builder stack.

- [ ] Add failing manifest tests for applied limits and policy identity.

Extend `BuildMetadata` with:

```python
rejection_limits: dict[str, int | float] = field(default_factory=dict)
rejection_totals: dict[str, int] = field(default_factory=dict)
```

Write `rejectionLimits` and `rejectionTotals` beside `rejectionCounts` in `manifest.json`, so a release check can evaluate both absolute and fractional limits without guessing a denominator. Assert the production builder records:

```json
"sentenceFilter": {"name": "is_japanese", "minLength": 5}
```

and the existing sentence splitter model checksum.

- [ ] Confirm the focused tests fail.

```bash
nix develop .#test --command pytest tests/test_builder_cli.py tests/test_corpus_adapters.py tests/test_artifact_builder.py -q
```

- [ ] Implement the CLI-edge loading once, then pass `article_ids` into the adapter.

Acquisition verifies the local JNLP archive before preparation. Inspection and build verify the exact Wikipedia Parquet again before adaptation and record the selected JNLP lock identity; they do not pretend the prepared directory is cryptographically bound to the archive. No adapter reads source-lock JSON.

- [ ] Add simple progress logging without a callback abstraction.

Configure `logging.basicConfig` once in the CLI. At INFO level, log accepted source counts, sentence count after segmentation, every 1,000 parsed sentences, occurrence count, and artifact completion. Keep extraction single-threaded and eager.

- [ ] Run focused tests and the default backend gate.

```bash
nix develop .#test --command pytest tests/test_builder_cli.py tests/test_corpus_adapters.py tests/test_artifact_builder.py -q
nix build .#checks.x86_64-linux.source-quality --print-build-logs
```

- [ ] Commit.

```bash
git add src/natsume_simple/builder_cli.py src/natsume_simple/corpus_pipeline.py src/natsume_simple/artifact_builder.py tests/test_builder_cli.py tests/test_corpus_adapters.py tests/test_artifact_builder.py
git commit -m "feat: inspect and record production build inputs"
```

### Task 6: Add structural release checks and content notices

**Files:**

- Create: `src/natsume_simple/release_check.py`
- Create: `tests/test_release_check.py`
- Create: `corpus-notices/LICENSE-CONTENT.txt`
- Create: `corpus-notices/ATTRIBUTION.md`
- Modify: `src/natsume_simple/artifact_builder.py`
- Modify: `src/natsume_simple/builder_cli.py`
- Modify: `tests/test_artifact_builder.py`
- Modify: `tests/test_builder_cli.py`

- [ ] Write failing release-check tests using a two-corpus fixture artifact.

Cover one passing artifact and bounded failures for:

- missing or empty notice;
- corpus set other than exactly `{jnlp, wiki}`;
- Wikipedia external IDs differing from the subset;
- manifest `identityInputs.sources` differing from database source identity/content hashes;
- database `build_metadata.source_manifest_sha256` differing from the canonical manifest source list;
- a rejection count exceeding its recorded absolute or fractional limit.

Do not add these assertions to `validate_artifact`.

- [ ] Confirm the tests fail.

```bash
nix develop .#test --command pytest tests/test_release_check.py tests/test_builder_cli.py -k release -q
```

- [ ] Implement one public operation.

```python
check_release_artifact(
    artifact_dir: Path,
    *,
    source_lock: Path,
    wikipedia_subset: Path,
) -> dict[str, object]
```

It calls ordinary `validate_artifact`, performs the release assertions with read-only DuckDB queries, returns a bounded summary, and raises `ReleaseCheckError(reason)` on the first failed invariant. Promote `_source_manifest_sha256` to the public, tested `source_manifest_sha256` helper in `artifact_builder.py` and reuse it here rather than duplicating canonicalization.

Keep `release_check.py` on the lightweight JSON/DuckDB path: it may import `artifact_builder`, `artifact_validation`, and `release_inputs`, but not `corpus_pipeline`, Polars, spaCy, Torch, or wtpsplit.

- [ ] Add the CLI command.

```text
natsume-corpus release-check ARTIFACT \
  --source-lock docs/corpus-sources.lock.json \
  --wikipedia-subset docs/wikipedia-ja-20231101-subset.json
```

Print the summary as JSON and exit nonzero through the existing CLI error path.

- [ ] Write concise repository notices.

`LICENSE-CONTENT.txt` must state that software remains MIT-licensed while the combined database and distributed corpus-derived content are treated as CC BY-SA 4.0. `ATTRIBUTION.md` must name ANLP and Wikimedia/Wikipedia, link the locked source/license pages, disclose conversion/segmentation/normalization/extraction, and name `dev@bor.space` for correction/takedown requests.

- [ ] Run focused tests.

```bash
nix develop .#test --command pytest tests/test_release_check.py tests/test_builder_cli.py -q
```

- [ ] Commit.

```bash
git add src/natsume_simple/release_check.py src/natsume_simple/artifact_builder.py src/natsume_simple/builder_cli.py tests/test_release_check.py tests/test_artifact_builder.py tests/test_builder_cli.py corpus-notices/LICENSE-CONTENT.txt corpus-notices/ATTRIBUTION.md
git commit -m "feat: check production artifacts for release"
```

### Task 7: Expose attribution in the public service

**Files:**

- Modify: `natsume-frontend/src/routes/+page.svelte`
- Modify: `natsume-frontend/tests/test.ts`

- [ ] Add a failing Playwright assertion to the existing page-load flow.

Assert a footer is visible and contains links or text for ANLP, Wikipedia/Wikimedia, CC BY 4.0, CC BY-SA 4.0, and `dev@bor.space`.

- [ ] Run the focused browser test and confirm failure.

```bash
nix build .#checks.x86_64-linux.playwright --print-build-logs
```

- [ ] Add one concise footer after `<main>`.

Keep the full provenance in `ATTRIBUTION.md`; the UI needs only names, license links, modification disclosure, and contact. Do not add a new API endpoint or fetch artifact notice files at runtime.

- [ ] Run frontend gates.

```bash
nix build .#checks.x86_64-linux.frontend --print-build-logs
nix build .#checks.x86_64-linux.playwright --print-build-logs
```

- [ ] Commit.

```bash
git add natsume-frontend/src/routes/+page.svelte natsume-frontend/tests/test.ts
git commit -m "feat: expose corpus attribution publicly"
```

### Task 8: Document and verify the stable operator workflow

**Files:**

- Modify: `README.md`
- Modify: `flake.nix`
- Modify: `docs/corpus-recoverability.md`

- [ ] Update README's **Corpus pipeline** section with the exact sequence:

```bash
nix run .#build-corpus -- acquire-release-inputs --source-lock docs/corpus-sources.lock.json --output-directory data/release-inputs
nix run .#build-corpus -- prepare-jnlp data/release-inputs/NLP_LATEX_CORPUS-2026-06-15.zip data/prepared-jnlp-2026
nix run .#build-corpus -- inspect-inputs --source-lock docs/corpus-sources.lock.json --wikipedia-subset docs/wikipedia-ja-20231101-subset.json --jnlp-root data/prepared-jnlp-2026/NLP_LATEX_CORPUS --wikipedia-parquet data/release-inputs/train-00000-of-00015.parquet
nix run .#build-corpus -- build --artifacts-directory artifacts --source-lock docs/corpus-sources.lock.json --wikipedia-subset docs/wikipedia-ja-20231101-subset.json --jnlp-root data/prepared-jnlp-2026/NLP_LATEX_CORPUS --wikipedia-parquet data/release-inputs/train-00000-of-00015.parquet --splitter-model data/models/wtpsplit --content-license corpus-notices/LICENSE-CONTENT.txt --attribution corpus-notices/ATTRIBUTION.md
nix run .#build-corpus -- release-check artifacts/<instance-id> --source-lock docs/corpus-sources.lock.json --wikipedia-subset docs/wikipedia-ja-20231101-subset.json
```

State explicitly that `publish` changes the pointer and restart/redeployment changes the served artifact. Keep rollback as “publish the former instance, then restart.”

- [ ] Update README's **OCI image** section to preserve its resolved bind mount and require redeployment after pointer change.

- [ ] Update README's **Checks and dependency locks** section to identify the release check and real build as operator/release evidence, not default CI inputs.

- [ ] Update `docs/corpus-recoverability.md` only where commands or status changed; do not duplicate the source lock or subset.

- [ ] Extend `builderSmoke` minimally.

Keep it network-independent. Add fixture calls that prove the packaged CLI can parse/verify a tiny fixture source lock and subset and run `inspect-inputs`; do not download production data or load models in Nix checks.

- [ ] Format and run the whole repository gate.

```bash
nix fmt
nix flake check --print-build-logs
nix build .#frontend .#server .#corpus-builder-cpu
nix build .#container
```

- [ ] Commit.

```bash
git add README.md flake.nix docs/corpus-recoverability.md
git commit -m "docs: publish the production corpus release workflow"
```

### Task 9: Build the real artifact and record release evidence

**Files:**

- Create: `docs/releases/<artifact-instance-id>.md`
- Modify: `README.md` only if the real run disproves a command
- Modify: `docs/corpus-sources.lock.json` only if verified evidence corrects it

- [ ] Re-run verified acquisition and input inspection from Task 8.

Record the accepted JNLP/Wikipedia source counts and rejection reasons. Choose `--max-rejections` and `--max-rejection-fraction` above the observed adapter counts without making them unbounded. For each accepted rejection class, record why it is legitimate (for example, a `*NA*` metadata row names no source file), then record the limit and headroom; do not justify a threshold only by saying it exceeds the observation.

- [ ] Build the real artifact under resource measurement.

```bash
nix develop .#release --command time -v natsume-corpus build \
  --artifacts-directory artifacts \
  --source-lock docs/corpus-sources.lock.json \
  --wikipedia-subset docs/wikipedia-ja-20231101-subset.json \
  --jnlp-root data/prepared-jnlp-2026/NLP_LATEX_CORPUS \
  --wikipedia-parquet data/release-inputs/train-00000-of-00015.parquet \
  --splitter-model data/models/wtpsplit \
  --content-license corpus-notices/LICENSE-CONTENT.txt \
  --attribution corpus-notices/ATTRIBUTION.md \
  --max-rejections <chosen-count> \
  --max-rejection-fraction <chosen-fraction>
```

Record start/finish, elapsed time, maximum RSS, CPU/memory, artifact ID, and tool/model identities from the manifest.

- [ ] Run structural release checks.

```bash
nix run .#build-corpus -- release-check artifacts/<instance-id> \
  --source-lock docs/corpus-sources.lock.json \
  --wikipedia-subset docs/wikipedia-ja-20231101-subset.json
```

- [ ] Record optional one-time legacy comparisons.

When `data/corpus.db` exists, compare all 971 Wikipedia titles and the 459 legacy JNLP identities, reporting added/missing/changed results. Do not make future publication depend on the old database.

- [ ] Record query plans as observations.

Use the same suggestion, noun-collocation, and verb-collocation SQL paths exercised by the API for `情報` and `行う`. Store `EXPLAIN` output in the release note without inventing a pass predicate.

- [ ] Run the direct-application benchmark with 10 concurrent clients and 500 total requests.

Start the server directly against the candidate artifact, then distribute 500 requests across 10 concurrent clients and round-robin suggestions for `情報`, noun collocations for `情報`, verb collocations for `行う`, and examples for the first returned collocation, with both `rankBy` values and both corpora. Record exact URLs, host facts, DuckDB settings, concurrency 10, request count 500, success rate, errors, and p95. Acceptance is 100% success, no 5xx, and p95 below one second.

- [ ] Review the release note as Bor Hodošček, then record the former artifact ID if `deploy/current` exists.

- [ ] Publish, restart the repository-defined direct server, and smoke the reference edge.

```bash
nix run .#build-corpus -- publish artifacts/<instance-id> deploy
nix run .#build-corpus -- current deploy

release_runtime=$(mktemp -d)
mkdir -p "$release_runtime/logs"
nix develop .#release --command natsume-serve \
  --artifact-dir deploy/current --port 8000 \
  >"$release_runtime/server.log" 2>&1 &
server_pid=$!
nix develop .#release --command nginx \
  -p "$release_runtime" -c "$PWD/deploy/nginx.conf" -g 'daemon off;' \
  >"$release_runtime/nginx.log" 2>&1 &
edge_pid=$!
```

Wait for `http://127.0.0.1:8081/ready`, then verify `/api/corpora`, one representative search in both directions, one examples expansion, and the attribution footer through `http://127.0.0.1:8080`. Record the served artifact ID and results. Stop both recorded PIDs after the smoke.

The repository defines the packaged direct server, OCI image, and reference nginx configuration, but no long-lived host supervisor. This procedure is the executable repository-scoped release target. If the public host has an external systemd/container supervisor, record and run its actual restart command in the release note; do not claim that this plan provisioned that external runtime.

- [ ] If smoke fails, stop the test processes, republish the recorded former artifact, rerun the same direct-server/edge commands, and repeat readiness. Do not mutate either artifact.

- [ ] Commit durable evidence only.

```bash
git add docs/releases/<artifact-instance-id>.md
git commit -m "docs: record the first production corpus release"
```

### Task 10: Retire implementation scaffolding after cutover

**Files:**

- Delete: `docs/superpowers/specs/2026-08-13-production-corpus-release-design.md`
- Delete: `docs/superpowers/plans/2026-08-13-production-corpus-release.md`
- Modify: any incoming links found by `rg`; none exist at plan-review time

- [ ] Confirm every durable fact now has one surviving owner:

- source bytes and provenance: `docs/corpus-sources.lock.json`;
- frozen Wikipedia identities: `docs/wikipedia-ja-20231101-subset.json`;
- commands and rollback: `README.md`;
- measured release evidence: `docs/releases/<artifact-instance-id>.md`;
- content terms/contact: `corpus-notices/` and the public footer.

- [ ] Delete the fulfilled spec and this plan, then fix any incoming links found at execution time.

- [ ] Run final verification before claiming completion.

```bash
git status --short
nix flake check --print-build-logs
nix build .#frontend .#server .#corpus-builder-cpu .#container
```

- [ ] Commit.

```bash
git add -u docs/superpowers/specs/2026-08-13-production-corpus-release-design.md docs/superpowers/plans/2026-08-13-production-corpus-release.md
git status --short
git commit -m "docs: retire the completed release scaffolding"
```
