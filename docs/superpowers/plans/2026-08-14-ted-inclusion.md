# TED Corpus Inclusion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the locked IWSLT 2017 Japanese training corpus as a talk-level `ted` source, build and validate a three-corpus external artifact, and expose accurate mixed-license notices without committing source archives or databases.

**Architecture:** Extend the existing release-input, adapter, builder, and release-check path; do not create a second TED pipeline. The adapter reads one named ZIP member into talk-level `SourceDocument` values, all adapters share one framed content hash, and release checking binds the TED source-file identity directly to the lock. WIT³ remains evidence only. The final multi-hour build and benchmark are operator release evidence outside default CI.

**Tech Stack:** Python 3.12, stdlib `zipfile`/`html`/`hashlib`, Polars, SaT/wtpsplit, GiNZA/spaCy, DuckDB, pytest 9, Svelte/Playwright, Nix flakes.

## Global Constraints

- Production TED text comes only from `ja-en/train.tags.ja-en.ja` in IWSLT 2017 revision `c18a4f81a47ae6fa079fe9d32db288ddde38451d`.
- Locked archive size is `26,190,859`; SHA-256 is `a923cdaa5632e55a94e799d31468c58fd2eab290a34f8b950a562f6ff20046b6`.
- WIT³ is not acquired, adapted, or merged into the production `ted` corpus.
- One `<doc>` is one source; `<talkid>` is the external ID; non-tag rows remain ordered text units.
- Missing/duplicate IDs and conflicting metadata abort adaptation. A talk with no subtitle text is a counted talk-level rejection.
- All adapters use `natsume-source-content-v1`: prefix plus NUL, then each UTF-8 text unit framed by an unsigned eight-byte big-endian byte length.
- `is_japanese(..., min_length=5)`, SaT, GiNZA, schema v1, and the 450 item-corpus API budget remain unchanged.
- The generated database and all source/model archives remain under ignored external paths; never add them to Git or a Nix check/server closure.
- The artifact is mixed-term content with no single SPDX grant. TED is included by owner decision while permission remains unresolved.
- Run Python tests through `nix develop .#test`; run packaged operator commands through `nix run .#build-corpus` or `nix develop .#release`.

---

### Task 1: Uniform framed source-content identity

**Files:**
- Modify: `src/natsume_simple/corpus_pipeline.py`
- Modify: `src/natsume_simple/builder_cli.py`
- Test: `tests/test_corpus_adapters.py`
- Test: `tests/test_builder_cli.py`

**Interfaces:**
- Produces: `source_content_sha256(text_units: tuple[str, ...]) -> str`.
- Produces: `identityInputs.sourceContentHash == "natsume-source-content-v1"` for every new artifact.
- Consumes: existing `SourceDocument.text_units` tuples.

- [ ] **Step 1: Write fixed-vector failing tests**

Import `source_content_sha256` and assert:

```python
def test_source_content_hash_frames_ordered_utf8_units():
    assert source_content_sha256(("一行目。", "二行目。")) == (
        "2ed1b83c19688e8f8a24142f973ff992face1c71641953875b40cb54b3583a5f"
    )
    assert source_content_sha256(("一行目。二行目。",)) != source_content_sha256(
        ("一行目。", "二行目。")
    )
```

Update existing adapter expectations to:

```python
assert jnlp_document.content_sha256 == (
    "f0510a814ea358642b12f3490a1504b8b7583198a6d03c749aaba0e54d070131"
)
assert wikipedia_document.content_sha256 == (
    "8d6f8d6e7fbfc4a27e54b52f0c1707214ca66d343eb62ce739617b5af6589e42"
)
```

In `test_build_records_release_sources_and_sentence_policy`, add:

```python
assert identity["sourceContentHash"] == "natsume-source-content-v1"
```

- [ ] **Step 2: Verify the tests fail on the unframed hashes**

```bash
nix develop .#test --command pytest tests/test_corpus_adapters.py tests/test_builder_cli.py -q
```

Expected: FAIL because the helper and identity field do not exist and old adapters hash bare text.

- [ ] **Step 3: Implement the one hash helper and use it in both existing adapters**

Replace `_text_sha256` with:

```python
SOURCE_CONTENT_HASH = "natsume-source-content-v1"


def source_content_sha256(text_units: tuple[str, ...]) -> str:
    digest = hashlib.sha256(SOURCE_CONTENT_HASH.encode("ascii") + b"\0")
    for text_unit in text_units:
        encoded = text_unit.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big", signed=False))
        digest.update(encoded)
    return digest.hexdigest()
```

Call it with `(text,)` from JNLP and Wikipedia. Add
`"sourceContentHash": SOURCE_CONTENT_HASH` to the build identity inputs, importing the constant lazily with the other pipeline symbols.

- [ ] **Step 4: Run focused and static checks**

```bash
nix develop .#test --command pytest tests/test_corpus_adapters.py tests/test_builder_cli.py -q
nix develop .#test --command mypy --ignore-missing-imports --show-error-context src
```

Expected: PASS.

- [ ] **Step 5: Commit the uniform content identity**

```bash
git add src/natsume_simple/corpus_pipeline.py src/natsume_simple/builder_cli.py tests/test_corpus_adapters.py tests/test_builder_cli.py
git commit -m "feat: frame source content hashes uniformly"
```

---

### Task 2: IWSLT talk adapter

**Files:**
- Modify: `src/natsume_simple/corpus_pipeline.py`
- Test: `tests/test_corpus_adapters.py`

**Interfaces:**
- Produces: `adapt_ted_iwslt_archive(archive: Path) -> AdaptationResult`.
- Produces: `corpus_id="ted"`; talk-level sources sorted by `external_id`.
- Consumes: exact ZIP member `ja-en/train.tags.ja-en.ja` and Task 1's `source_content_sha256`.

- [ ] **Step 1: Add a local ZIP fixture helper**

In the test module:

```python
def write_iwslt_archive(path: Path, japanese_training: str) -> Path:
    import zipfile

    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("ja-en/train.tags.ja-en.ja", japanese_training)
    return path
```

- [ ] **Step 2: Write the successful talk grouping and bounded-empty test**

```python
def test_ted_adapter_groups_rows_by_talk_and_counts_empty_talks(tmp_path: Path):
    archive = write_iwslt_archive(
        tmp_path / "ja-en.zip",
        """<doc docid="1" genre="lectures">
<url>https://www.ted.com/talks/example</url>
<speaker>話者</speaker>
<talkid>42</talkid>
<title>例の講演</title>
一行目。
二行目。
</doc>
<doc docid="2" genre="lectures">
<talkid>43</talkid>
<title>空の講演</title>
</doc>
""",
    )

    result = adapt_ted_iwslt_archive(archive)

    assert result.corpus_id == "ted"
    assert result.rejections == {"empty_subtitle_text": 1}
    assert len(result.documents) == 1
    talk = result.documents[0]
    assert talk.external_id == "42"
    assert talk.title == "例の講演"
    assert talk.year is None
    assert talk.author == "話者"
    assert talk.publisher == "TED Conference LLC"
    assert talk.url == "https://www.ted.com/talks/example"
    assert talk.text_units == ("一行目。", "二行目。")
    assert talk.content_sha256 == source_content_sha256(talk.text_units)
```

- [ ] **Step 3: Write hard-abort identity tests**

Parameterize fixtures for:

```python
[
    (
        "<doc>\n<title>missing</title>\n本文。\n</doc>",
        "ted_identity_missing",
    ),
    (
        "<doc>\n<talkid>1</talkid>\n本文。\n</doc>\n"
        "<doc>\n<talkid>1</talkid>\n別文。\n</doc>",
        "ted_identity_duplicate",
    ),
    (
        "<doc>\n<talkid>1</talkid>\n<title>A</title>\n"
        "<title>B</title>\n本文。\n</doc>",
        "ted_metadata_conflict",
    ),
]
```

Each case must raise `ValueError` with the exact reason and must not return a counted rejection.

Add separate archive-shape cases asserting `ted_archive_member_invalid` for a missing named member and for a ZIP containing two entries with that same name.

- [ ] **Step 4: Confirm the adapter tests fail before implementation**

```bash
nix develop .#test --command pytest tests/test_corpus_adapters.py -k ted -q
```

Expected: FAIL because `adapt_ted_iwslt_archive` does not exist.

- [ ] **Step 5: Implement a single-pass state-machine parser**

Use `zipfile.ZipFile`, `io.TextIOWrapper`, `html.unescape`, and one current-talk dictionary. Strip each line; recognize a `<doc>` opening tag with optional attributes, `</doc>`, and complete metadata tags with a compiled expression:

```python
TED_TRAINING_MEMBER = "ja-en/train.tags.ja-en.ja"
TED_METADATA = re.compile(r"^<([a-z]+)>(.*)</\1>$")
```

For metadata keys `talkid`, `title`, `speaker`, and `url`, store the first unescaped value and raise `ted_metadata_conflict` if the same key repeats with a different value. Ignore known non-source metadata such as `keywords`, `description`, `reviewer`, and `translator`. A non-tag line inside a document is an unescaped text unit. Text outside a document raises `ted_archive_structure_invalid`.

On `</doc>`:

```python
if not talk_id:
    raise ValueError("ted_identity_missing")
if talk_id in seen_talk_ids:
    raise ValueError("ted_identity_duplicate")
if not text_units:
    rejections["empty_subtitle_text"] += 1
else:
    documents.append(
        SourceDocument(
            corpus_id="ted",
            external_id=talk_id,
            title=metadata.get("title", f"TED Talk {talk_id}"),
            year=None,
            author=_optional_text(metadata.get("speaker")),
            publisher="TED Conference LLC",
            url=_optional_text(metadata.get("url")),
            text_units=tuple(text_units),
            content_sha256=source_content_sha256(tuple(text_units)),
        )
    )
```

Reject nested/unclosed documents. Sort the final tuple by `external_id`. Do not extract ZIP members to disk.

- [ ] **Step 6: Run focused, full adapter, and formatting checks**

```bash
nix develop .#test --command pytest tests/test_corpus_adapters.py -q
nix develop .#test --command ruff format --check src/natsume_simple/corpus_pipeline.py tests/test_corpus_adapters.py
nix develop .#test --command ruff check src/natsume_simple/corpus_pipeline.py tests/test_corpus_adapters.py
```

Expected: PASS.

- [ ] **Step 7: Commit the adapter**

```bash
git add src/natsume_simple/corpus_pipeline.py tests/test_corpus_adapters.py
git commit -m "feat: adapt IWSLT talks as TED sources"
```

---

### Task 3: Lock and acquire the selected TED archive

**Files:**
- Modify: `docs/corpus-sources.lock.json`
- Modify: `src/natsume_simple/release_inputs.py`
- Modify: `tests/test_release_inputs.py`
- Modify: `tests/test_builder_cli.py`

**Interfaces:**
- Extends: `ReleaseSources(jnlp_archive, wikipedia_shard, ted_archive, wikipedia_identity_sha256)`.
- Produces: three-path `acquire_release_inputs(...) -> tuple[Path, Path, Path]` in JNLP, Wikipedia, TED order.
- Consumes: the IWSLT lock entry with `status="ready"`, `plannedPublic=true`, and `servingCorpusId="ted"`.

- [ ] **Step 1: Extend the source-lock test fixture and expectations**

Add this entry to `write_source_lock`:

```python
{
    "corpusId": "ted-iwslt-2017-ja-en",
    "servingCorpusId": "ted",
    "plannedPublic": True,
    "status": "ready",
    "candidate": {
        "path": "data/2017-01-trnted/texts/ja/en/ja-en.zip",
        "url": "https://example.test/ja-en.zip",
        "size": 3,
        "sha256": hashlib.sha256(b"ted").hexdigest(),
    },
}
```

Assert `sources.ted_archive.name == "ja-en.zip"`, and change acquisition expectations to three paths and three payload URLs.

- [ ] **Step 2: Confirm the release-input tests fail**

```bash
nix develop .#test --command pytest tests/test_release_inputs.py -q
```

Expected: FAIL because `ReleaseSources` has no TED field and acquisition returns two files.

- [ ] **Step 3: Extend the typed lock reader minimally**

Add `ted_archive: LockedFile` to `ReleaseSources`. In `load_release_sources`, select `entries["ted-iwslt-2017-ja-en"]`, require `servingCorpusId == "ted"`, and construct:

```python
LockedFile(
    source_lock_corpus_id="ted-iwslt-2017-ja-en",
    name="ja-en.zip",
    local_name="ja-en.zip",
    url=str(ted["candidate"]["url"]),
    size=int(ted["candidate"]["size"]),
    sha256=str(ted["candidate"]["sha256"]),
)
```

Return it as the third acquired file. Keep generic download verification unchanged.

Update the existing mocked builder fixture so its positional construction remains
type-correct after this task:

```python
ted = LockedFile(
    "ted-iwslt-2017-ja-en",
    "ja-en.zip",
    "ja-en.zip",
    "ted",
    3,
    hashlib.sha256(b"ted").hexdigest(),
)
sources = ReleaseSources(jnlp, wiki, ted, "c" * 64)
```

- [ ] **Step 4: Change the repository lock disposition**

For `ted-iwslt-2017-ja-en`, set:

```json
"servingCorpusId": "ted",
"plannedPublic": true,
"status": "ready"
```

Retain the immutable revision, URL, size, SHA-256, README license evidence, and unresolved-permission reason. Keep `ted-iwslt-2014-2016` excluded. Change planned corpus IDs to `jnlp`, `ted`, `wiki`; set top-level `gateStatus` to `owner-accepted-unresolved-permission`; replace the single `spdxExpression` conclusion with `singleLicenseAsserted: false` and explicit JNLP, Wikipedia, and TED conclusions from the design.

- [ ] **Step 5: Assert the real lock values**

Extend `test_repository_lock_selects_the_planned_release_files`:

```python
assert sources.ted_archive.size == 26_190_859
assert sources.ted_archive.sha256 == (
    "a923cdaa5632e55a94e799d31468c58fd2eab290a34f8b950a562f6ff20046b6"
)
```

- [ ] **Step 6: Run tests and JSON parsing**

```bash
nix develop .#test --command pytest tests/test_release_inputs.py tests/test_builder_cli.py -q
jq empty docs/corpus-sources.lock.json
```

Expected: PASS.

- [ ] **Step 7: Commit the source selection**

```bash
git add docs/corpus-sources.lock.json src/natsume_simple/release_inputs.py tests/test_release_inputs.py tests/test_builder_cli.py
git commit -m "feat: select locked IWSLT TED input"
```

---

### Task 4: Builder and inspection integration

**Files:**
- Modify: `src/natsume_simple/builder_cli.py`
- Modify: `tests/test_builder_cli.py`
- Modify: `README.md`

**Interfaces:**
- Adds: `--ted-iwslt-archive PATH` to `build` (optional local corpus) and `inspect-inputs` (required production input).
- Consumes: Tasks 2–3 `adapt_ted_iwslt_archive` and `ReleaseSources.ted_archive`.
- Produces: TED corpus record, verified TED `sourceFiles` identity, adapter policy/counts, and aggregate sentence-filter observations in operator logs.

- [ ] **Step 1: Write parser and source-lock requirement tests**

Assert the build parser accepts `--ted-iwslt-archive`, and add a CLI error test proving TED without `--source-lock` exits 2 with:

```text
TED requires --source-lock
```

Extend the mocked build test to include a `LockedFile` for TED, a mocked TED adaptation, and these assertions:

```python
assert [corpus.id for corpus in captured["corpora"]] == ["ted", "wiki"]
assert identity["sourceAdapters"] == ["ted", "wiki"]
assert identity["sourceContentHash"] == "natsume-source-content-v1"
assert identity["tedSelectionPolicy"] == "iwslt2017-ja-en-training-v1"
assert next(
    item for item in identity["sourceFiles"] if item["corpusId"].startswith("ted-")
) == {
    "corpusId": "ted-iwslt-2017-ja-en",
    "name": "ja-en.zip",
    "sha256": hashlib.sha256(b"ted").hexdigest(),
    "size": 3,
}
```

- [ ] **Step 2: Extend the inspection test**

Make the summary include:

```python
"ted": {
    "acceptedSources": 1,
    "textUnits": 2,
    "rejections": {"empty_subtitle_text": 1},
}
```

Assert `inspect-inputs` requires and forwards the TED archive while still completing before SaT/GiNZA imports.

- [ ] **Step 3: Verify the focused tests fail**

```bash
nix develop .#test --command pytest tests/test_builder_cli.py -q
```

Expected: FAIL because the CLI and metadata do not include TED.

- [ ] **Step 4: Add the TED CLI path and verify before model loading**

Add the arguments, load release sources once when either Wikipedia or TED is supplied, call `verify_file(args.ted_iwslt_archive, release_sources.ted_archive)`, adapt it, and add `CorpusRecord("ted", "TED Talks")`. Build `sourceFiles` from inputs actually selected rather than blindly recording every lock entry.

Record:

```python
"tedSelectionPolicy": "iwslt2017-ja-en-training-v1",
"sourceObservations": {
    adaptation.corpus_id: {
        "acceptedSources": len(adaptation.documents),
        "textUnits": sum(len(document.text_units) for document in adaptation.documents),
    }
    for adaptation in adaptations
},
```

- [ ] **Step 5: Count the existing sentence filter without adding a policy**

Add an optional standard-library counter to the existing helper:

```python
def split_japanese_sentences(
    text_units: tuple[str, ...],
    *,
    splitter: object,
    observations: Counter[str] | None = None,
) -> Iterable[str]:
```

Materialize each stripped candidate returned by SaT long enough to increment `candidate`, `retained`, and `dropped`; yield only the same `is_japanese(..., min_length=5)` results as today. In `_build`, pass one `Counter`, wait for `build_corpus_artifact` to finish consuming it, then log one structured line with those three totals. Do not persist the mutable counter in `identityInputs`.

Test with one Japanese and one English candidate:

```python
observations = Counter()
assert list(split_japanese_sentences(units, splitter=Splitter(), observations=observations)) == [
    "日本語の文章です。"
]
assert observations == {"candidate": 2, "retained": 1, "dropped": 1}
```

- [ ] **Step 6: Update documented commands**

Add `--ted-iwslt-archive data/release-inputs/ja-en.zip` to `inspect-inputs` and
`build`; keep the measured production release limits explicit as
`--max-rejections 200 --max-rejection-fraction 0.15`. State why the known JNLP
`missing_source_path` and `missing_plain_text` classes are legitimate, that
acquisition now returns three files, WIT³ is not a release input, and the real
database remains ignored/external.

- [ ] **Step 7: Run focused tests and source checks**

```bash
nix develop .#test --command pytest tests/test_builder_cli.py tests/test_corpus_adapters.py -q
nix develop .#test --command ruff format --check src tests
nix develop .#test --command ruff check src tests
nix develop .#test --command mypy --ignore-missing-imports --show-error-context src
```

Expected: PASS.

- [ ] **Step 8: Commit CLI integration**

```bash
git add src/natsume_simple/builder_cli.py tests/test_builder_cli.py README.md
git commit -m "feat: integrate TED with corpus builder"
```

---

### Task 5: Three-corpus release policy

**Files:**
- Modify: `src/natsume_simple/release_check.py`
- Modify: `tests/test_release_check.py`

**Interfaces:**
- Consumes: `ReleaseSources.ted_archive`, manifest `identityInputs.sourceFiles`, and `identityInputs.sourceContentHash`.
- Produces: exact corpus set `jnlp`, `ted`, `wiki`; TED source-file equality check; no frozen TED talk-list protocol.

- [ ] **Step 1: Extend the release fixture to three corpora**

Add one TED source, sentence, and occurrence to `release_records`; include `CorpusRecord("ted", "TED Talks")`. Build fixture metadata with:

```python
"sourceContentHash": "natsume-source-content-v1",
"sourceFiles": [
    {
        "corpusId": "jnlp",
        "name": "NLP_LATEX_CORPUS.zip",
        "sha256": hashlib.sha256(b"jnlp").hexdigest(),
        "size": 4,
    },
    {
        "corpusId": "ted-iwslt-2017-ja-en",
        "name": "ja-en.zip",
        "sha256": hashlib.sha256(b"ted").hexdigest(),
        "size": 3,
    },
    {
        "corpusId": "wikipedia-ja-20231101",
        "name": "train-00000-of-00015.parquet",
        "sha256": hashlib.sha256(b"wiki").hexdigest(),
        "size": 4,
    },
],
```

Update rejection counts/totals to include `"ted": {}` / `"ted": 1`, update
the success summary to `corpusIds == ["jnlp", "ted", "wiki"]`,
`sourceCount == 973`, and retain exactly 971 Wikipedia identities.

- [ ] **Step 2: Write lock-binding failure tests**

Parameterize mutations of the TED source-file dictionary for wrong `corpusId`, `name`, `size`, and `sha256`. Each must raise:

```text
ted_source_file_mismatch
```

Add a separate mutation of `sourceContentHash` that raises `source_content_hash_mismatch`.

- [ ] **Step 3: Verify focused release tests fail**

```bash
nix develop .#test --command pytest tests/test_release_check.py -q
```

Expected: FAIL because release checking still requires only JNLP and Wikipedia and never reads `sourceFiles`.

- [ ] **Step 4: Implement exact structural checks**

Require sorted corpus IDs `['jnlp', 'ted', 'wiki']`. Read the manifest identity inputs defensively; find exactly one TED entry and compare it to:

```python
{
    "corpusId": sources.ted_archive.source_lock_corpus_id,
    "name": sources.ted_archive.name,
    "sha256": sources.ted_archive.sha256,
    "size": sources.ted_archive.size,
}
```

Require `sourceContentHash == "natsume-source-content-v1"`. Keep the full manifest/database source reconciliation and frozen Wikipedia subset check unchanged.

- [ ] **Step 5: Run release and API contract tests**

```bash
nix develop .#test --command pytest tests/test_release_check.py tests/test_api_contract.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit release policy**

```bash
git add src/natsume_simple/release_check.py tests/test_release_check.py
git commit -m "feat: validate three-corpus release artifacts"
```

---

### Task 6: Mixed-term notices and public attribution

**Files:**
- Modify: `corpus-notices/LICENSE-CONTENT.txt`
- Modify: `corpus-notices/ATTRIBUTION.md`
- Modify: `natsume-frontend/src/routes/+page.svelte`
- Modify: `natsume-frontend/tests/test.ts`
- Modify: `docs/corpus-recoverability.md`

**Interfaces:**
- Produces: no artifact-wide content SPDX grant; per-corpus JNLP/Wikipedia/TED terms; public TED attribution and contact/takedown path.
- Consumes: owner decision already recorded in the approved design and source lock.

- [ ] **Step 1: Write the failing footer test**

Extend the attribution Playwright test:

```ts
await expect(footer.getByText('TED Talks')).toBeVisible();
await expect(footer.getByText('no license grant is asserted', { exact: false })).toBeVisible();
await expect(footer.getByText('No single license applies', { exact: false })).toBeVisible();
```

Keep JNLP, Wikipedia, CC links, and contact assertions.

- [ ] **Step 2: Verify the footer test fails**

```bash
nix develop .#frontend --command bash -lc 'cd natsume-frontend && npm ci && npm run test:integration -- --grep "corpus attribution"'
```

Expected: FAIL because TED is absent and the footer implies a two-license combined conclusion.

- [ ] **Step 3: Replace the artifact-wide license claim**

`LICENSE-CONTENT.txt` must state:

- MIT applies only to software;
- no single content license is asserted for the collection;
- JNLP is CC BY 4.0;
- Wikipedia is subject to Wikimedia terms, CC BY-SA 4.0, and applicable GFDL notices;
- the IWSLT archive says TED copyright and CC BY-NC-ND 3.0, current TED terms require separate permission for dataset uses, and this project asserts no TED grant while including it by owner decision pending permission.

Do not say the combined `corpus.duckdb` is offered under CC BY-SA 4.0.

- [ ] **Step 4: Add complete TED attribution**

Add to `ATTRIBUTION.md`: provider TED Conference LLC; IWSLT 2017 archive/revision; original TED URL carried per talk; archive README terms; current TED terms link; modifications (grouping, sentence filtering/splitting, lemma normalization, NPV extraction); unresolved-permission statement; and existing `dev@bor.space` correction/takedown contact.

- [ ] **Step 5: Update the public footer and gate record**

Render JNLP, Wikipedia, and TED as distinct sources. Add the exact phrases asserted in Step 1 without claiming clearance. In `docs/corpus-recoverability.md`, replace the excluded/two-corpus disposition, record IWSLT-only selection and the measured WIT³ overlap, reverse the randomized legacy-ID mapping requirement, and state that owner acceptance supersedes the prior public-exclusion gate without changing the evidence about permission.

- [ ] **Step 6: Run footer, static, and text checks**

```bash
nix develop .#frontend --command bash -lc 'cd natsume-frontend && npm run test:integration -- --grep "corpus attribution" && npm run check'
rg -n "TED|no single|no license grant|dev@bor.space" corpus-notices natsume-frontend/src/routes/+page.svelte docs/corpus-recoverability.md
```

Expected: browser test passes and every notice surface carries the unresolved status and contact.

- [ ] **Step 7: Commit notices and gate disposition**

```bash
git add corpus-notices/LICENSE-CONTENT.txt corpus-notices/ATTRIBUTION.md natsume-frontend/src/routes/+page.svelte natsume-frontend/tests/test.ts docs/corpus-recoverability.md
git commit -m "docs: record mixed terms for TED release"
```

---

### Task 7: Hermetic fixture and full pre-release gate

**Files:**
- Modify: `flake.nix`
- Test: existing Nix checks.

**Interfaces:**
- Consumes: the Task 3 lock schema and Task 4 CLI.
- Produces: network-independent builder smoke with a tiny IWSLT ZIP; unchanged server closure exclusions.

- [ ] **Step 1: Extend the builder-smoke fixture**

In `builderSmoke`, create `fixture-inputs/ja-en.zip` with the named member and one talk:

```python
with ZipFile(root / "ja-en.zip", "w") as archive:
    archive.writestr(
        "ja-en/train.tags.ja-en.ja",
        "<doc>\n<talkid>1</talkid>\n<title>教材</title>\n教材を読む。\n</doc>\n",
    )
```

Add its size/SHA-256/status/serving ID to the synthetic lock. Pass it to `inspect-inputs`, and assert the JSON reports one accepted TED source and one text unit. Do not put the real archive in the derivation.

The smoke command becomes:

```bash
natsume-corpus inspect-inputs \
  --source-lock fixture-inputs/sources.json \
  --wikipedia-subset fixture-inputs/subset.json \
  --jnlp-root fixture-inputs/jnlp \
  --wikipedia-parquet fixture-inputs/train-00000-of-00015.parquet \
  --ted-iwslt-archive fixture-inputs/ja-en.zip > inspection.json
python - <<'PY'
import json

inspection = json.load(open("inspection.json", encoding="utf-8"))
assert inspection["ted"] == {
    "acceptedSources": 1,
    "rejections": {},
    "textUnits": 1,
}
PY
```

Also add `pkgs.duckdb` and `pkgs.jq` to `devShells.release.packages`; the release
procedure uses them for read-only artifact facts and JSON assertions and must
not depend on ambient user installations.

- [ ] **Step 2: Run Nix formatting before evaluation**

```bash
nix fmt flake.nix
```

Expected: only `flake.nix` formatting may change.

- [ ] **Step 3: Run focused builder and package checks**

```bash
nix build .#checks.x86_64-linux.builder-smoke --print-build-logs
nix build .#checks.x86_64-linux.package-corpus-builder-cpu --print-build-logs
```

Expected: PASS without network access or large source files.

- [ ] **Step 4: Run the complete repository gate**

```bash
nix flake check --print-build-logs
nix build .#frontend .#server .#corpus-builder-cpu
```

Expected: all checks pass. Confirm `server-closure` still excludes Torch, spaCy, GiNZA, wtpsplit, Polars, Node, and CUDA.

- [ ] **Step 5: Commit Nix fixture changes**

```bash
git add flake.nix
git commit -m "test: smoke TED release inputs in Nix"
```

---

### Task 8: Inspect and project the real external build

**Files:**
- Create after evidence exists: `docs/releases/${artifact_instance_id}.md` during Task 9, where `artifact_instance_id` is the basename emitted by the build; not in this task.
- Do not modify tracked code.

**Interfaces:**
- Consumes: ignored files `data/release-inputs/ja-en.zip`, JNLP/Wikipedia inputs, and `data/models/wtpsplit`.
- Produces: verified adapter counts and a pre-build resource decision recorded in `data/ted-preflight-20260814.json` and `data/ted-sat-sample-20260814.json` (ignored operator evidence).

- [ ] **Step 1: Verify or acquire all three locked files**

```bash
nix run .#build-corpus -- acquire-release-inputs \
  --source-lock docs/corpus-sources.lock.json \
  --output-directory data/release-inputs
```

Expected: prints three verified paths; TED resolves to `data/release-inputs/ja-en.zip`. Existing correct files are reused without downloading.

- [ ] **Step 2: Run the complete adapter-only inspection**

```bash
nix run .#build-corpus -- inspect-inputs \
  --source-lock docs/corpus-sources.lock.json \
  --wikipedia-subset docs/wikipedia-ja-20231101-subset.json \
  --jnlp-root data/prepared-jnlp-2026/NLP_LATEX_CORPUS \
  --wikipedia-parquet data/release-inputs/train-00000-of-00015.parquet \
  --ted-iwslt-archive data/release-inputs/ja-en.zip \
  | tee data/ted-preflight-20260814.json
```

Expected: TED reports 1,863 accepted talks, 223,108 text units, and bounded talk-level rejections; no NLP model loads.

- [ ] **Step 3: Run a deterministic 50-talk SaT sample**

Run this inside the builder environment:

```bash
nix develop .#builder --command python - <<'PY' | tee data/ted-sat-sample-20260814.json
import json
from collections import Counter
from pathlib import Path

from wtpsplit import SaT
from natsume_simple.builder_cli import split_japanese_sentences
from natsume_simple.corpus_pipeline import adapt_ted_iwslt_archive

documents = adapt_ted_iwslt_archive(Path("data/release-inputs/ja-en.zip")).documents[:50]
splitter = SaT("data/models/wtpsplit").eval().to("cpu")
observations = Counter()
sentences = [
    sentence
    for document in documents
    for sentence in split_japanese_sentences(
        document.text_units, splitter=splitter, observations=observations
    )
]
print(json.dumps({
    "talks": len(documents),
    "textUnits": sum(len(document.text_units) for document in documents),
    "sentences": len(sentences),
    "filter": dict(sorted(observations.items())),
}, ensure_ascii=False, sort_keys=True))
PY
```

Expected: stable JSON with 50 talks and nonzero text units/sentences. Re-running produces identical counts.

- [ ] **Step 4: Record the go/no-go projection in the terminal log**

Using the full text-unit count and sample sentence multiplier, calculate projected total sentences. Use legacy TED occurrence density only as an assumption to project approximately 1.07 million total occurrences and roughly 300 MB DuckDB size. Record 6–7 hours and 8–10 GiB as the initial range, then confirm the host has more than 10 GiB available and enough external disk for sources, staging, and final artifact.

Run:

```bash
free -h
df -h data artifacts
```

Expected: sufficient memory and disk. If not, stop before Task 9; do not add checkpoint machinery as a workaround.

---

### Task 9: Build, validate, benchmark, and select the three-corpus artifact

**Files:**
- Create: `docs/releases/${artifact_instance_id}.md` after assigning `artifact_instance_id=$(basename "$natsume_candidate_artifact")` from the actual build output.
- Modify only after successful evidence: `deploy/current` symlink through the existing publish command (ignored deployment state).
- Never add: `artifacts/**`, `data/**`, `*.db`, or `*.duckdb`.

**Interfaces:**
- Consumes: all implementation tasks and Task 8 preflight evidence.
- Produces: one immutable external artifact, release record, selected deployment pointer, direct-app benchmark, and live UI/API smoke.

- [ ] **Step 1: Run the measured production build**

```bash
nix develop .#release --command bash -lc '
  set -o pipefail
  command time -v natsume-corpus build \
    --artifacts-directory artifacts \
    --source-lock docs/corpus-sources.lock.json \
    --wikipedia-subset docs/wikipedia-ja-20231101-subset.json \
    --jnlp-root data/prepared-jnlp-2026/NLP_LATEX_CORPUS \
    --wikipedia-parquet data/release-inputs/train-00000-of-00015.parquet \
    --ted-iwslt-archive data/release-inputs/ja-en.zip \
    --splitter-model data/models/wtpsplit \
    --content-license corpus-notices/LICENSE-CONTENT.txt \
    --attribution corpus-notices/ATTRIBUTION.md \
    --max-rejections 200 \
    --max-rejection-fraction 0.15 \
    2>data/ted-build-20260814.log | tee data/ted-artifact-path-20260814.txt
'
```

Expected: one new artifact path on stdout; log includes filter totals, elapsed time, maximum RSS, and no rejection-limit failure.

- [ ] **Step 2: Run structural release checking**

```bash
natsume_candidate_artifact=$(tail -n 1 data/ted-artifact-path-20260814.txt)
nix run .#build-corpus -- release-check "$natsume_candidate_artifact" \
  --source-lock docs/corpus-sources.lock.json \
  --wikipedia-subset docs/wikipedia-ja-20231101-subset.json
```

Expected: JSON lists exactly `jnlp`, `ted`, `wiki`, verifies 971 Wikipedia identities, and reports the new artifact ID.

- [ ] **Step 3: Record exact artifact facts**

Use read-only DuckDB queries and `stat`:

```bash
nix develop .#release --command bash -lc '
  natsume_candidate_artifact=$(tail -n 1 data/ted-artifact-path-20260814.txt)
  stat -c "%s %n" "$natsume_candidate_artifact/corpus.duckdb"
  duckdb "$natsume_candidate_artifact/corpus.duckdb" -c "
    SELECT c.id, cs.source_count, cs.sentence_count, cs.collocation_count
    FROM corpus c JOIN corpus_stats cs ON cs.corpus_id = c.id ORDER BY c.id;
    SELECT count(*) AS lemma_rows FROM lemma_frequency;
  "
'
```

Expected: all three corpora have nonzero counts. Assign
`artifact_instance_id=$(basename "$natsume_candidate_artifact")` and record exact
source, sentence, occurrence, lemma, file-size, build-time, RSS, and filter
totals in `docs/releases/${artifact_instance_id}.md`.

- [ ] **Step 4: Start the candidate server and run the fixed benchmark**

Start a release shell with `nix develop .#release --command bash`, then run this
complete block inside it so `curl` and the packaged server come from declared
inputs:

```bash
natsume_candidate_artifact=$(tail -n 1 data/ted-artifact-path-20260814.txt)
NATSUME_ARTIFACT_DIR="$natsume_candidate_artifact" nix run .#serve >data/ted-server-20260814.log 2>&1 &
natsume_server_pid=$!
trap 'kill "$natsume_server_pid" 2>/dev/null || true' EXIT
for attempt in $(seq 1 60); do
  curl --fail --silent http://127.0.0.1:8000/api/health/ready >/dev/null && break
  sleep 1
done
nix develop .#test --command python - <<'PY'
import asyncio
import statistics
import time

import httpx

base = "http://127.0.0.1:8000"
urls = [
    "/api/suggestions?q=%E6%83%85%E5%A0%B1&pos=noun",
    "/api/collocations?term=%E6%83%85%E5%A0%B1&pos=noun&rankBy=raw&corpusId=jnlp&corpusId=ted&corpusId=wiki&limitPerParticle=100",
    "/api/collocations?term=%E6%83%85%E5%A0%B1&pos=noun&rankBy=meanPerMillion&corpusId=jnlp&corpusId=ted&corpusId=wiki&limitPerParticle=100",
    "/api/collocations?term=%E8%A1%8C%E3%81%86&pos=verb&rankBy=raw&corpusId=jnlp&corpusId=ted&corpusId=wiki&limitPerParticle=100",
    "/api/collocations?term=%E8%A1%8C%E3%81%86&pos=verb&rankBy=meanPerMillion&corpusId=jnlp&corpusId=ted&corpusId=wiki&limitPerParticle=100",
    "/api/examples?noun=%E6%83%85%E5%A0%B1&particle=%E3%81%8C&verb=%E5%90%AB%E3%81%BE%E3%82%8C%E3%82%8B&corpusId=jnlp&corpusId=ted&corpusId=wiki&limit=5",
]

async def main():
    durations = []
    async with httpx.AsyncClient(base_url=base, timeout=3.0) as client:
        for url in urls:
            response = await client.get(url)
            response.raise_for_status()

        queue = asyncio.Queue()
        for index in range(500):
            queue.put_nowait(urls[index % len(urls)])

        async def worker():
            while not queue.empty():
                try:
                    url = queue.get_nowait()
                except asyncio.QueueEmpty:
                    return
                started = time.perf_counter()
                response = await client.get(url)
                elapsed = (time.perf_counter() - started) * 1000
                assert response.status_code == 200, (response.status_code, url)
                durations.append(elapsed)
                queue.task_done()

        started = time.perf_counter()
        await asyncio.gather(*(worker() for _ in range(10)))
        wall = time.perf_counter() - started

    ordered = sorted(durations)
    p95 = ordered[int(len(ordered) * 0.95) - 1]
    print({
        "success": len(durations),
        "p50Ms": statistics.median(durations),
        "p95Ms": p95,
        "maxMs": max(durations),
        "wallSeconds": wall,
    })
    assert len(durations) == 500
    assert p95 < 1000

asyncio.run(main())
PY
```

Expected: 500 successful responses, no 5xx, p95 below 1000ms on the named host. Record the exact output and host facts in the release record.

- [ ] **Step 5: Validate curated TED behavior and response size**

Query `/api/corpora`, then select one deterministic nonzero TED collocation:

```bash
natsume_candidate_artifact=$(tail -n 1 data/ted-artifact-path-20260814.txt)
IFS=$'\t' read -r natsume_ted_noun natsume_ted_particle natsume_ted_verb < <(
  duckdb -noheader -separator $'\t' "$natsume_candidate_artifact/corpus.duckdb" \
    "SELECT noun, particle, verb
     FROM collocation_frequency
     WHERE corpus_id = 'ted'
     ORDER BY raw_frequency DESC, noun, particle, verb
     LIMIT 1"
)
curl --fail --silent --get http://127.0.0.1:8000/api/collocations \
  --data-urlencode "term=$natsume_ted_noun" \
  --data-urlencode 'pos=noun' \
  --data-urlencode 'rankBy=raw' \
  --data-urlencode 'corpusId=ted' > data/ted-only-collocations.json
curl --fail --silent --get http://127.0.0.1:8000/api/examples \
  --data-urlencode "noun=$natsume_ted_noun" \
  --data-urlencode "particle=$natsume_ted_particle" \
  --data-urlencode "verb=$natsume_ted_verb" \
  --data-urlencode 'corpusId=ted' > data/ted-only-examples.json
jq -e '.particleGroups | any(.items | length > 0)' data/ted-only-collocations.json
jq -e '.examples | length > 0' data/ted-only-examples.json
```

Request all three corpora at `limitPerParticle=100`, save the body, and assert:

```bash
curl --fail --silent --get http://127.0.0.1:8000/api/collocations \
  --data-urlencode "term=$natsume_ted_noun" \
  --data-urlencode 'pos=noun' \
  --data-urlencode 'rankBy=raw' \
  --data-urlencode 'corpusId=jnlp' \
  --data-urlencode 'corpusId=ted' \
  --data-urlencode 'corpusId=wiki' \
  --data-urlencode 'limitPerParticle=100' > data/ted-all-corpora-response.json
test "$(wc -c < data/ted-all-corpora-response.json)" -lt 1048576

natsume_status=$(curl --silent --output data/ted-limit-error.json --write-out '%{http_code}' --get \
  http://127.0.0.1:8000/api/collocations \
  --data-urlencode "term=$natsume_ted_noun" \
  --data-urlencode 'pos=noun' \
  --data-urlencode 'rankBy=raw' \
  --data-urlencode 'corpusId=jnlp' \
  --data-urlencode 'corpusId=ted' \
  --data-urlencode 'corpusId=wiki' \
  --data-urlencode 'limitPerParticle=151')
test "$natsume_status" = 400
jq -e '.error.code == "invalid_parameter"' data/ted-limit-error.json

for natsume_corpora in 'ted' 'jnlp&corpusId=ted'; do
  natsume_status=$(curl --silent --output /dev/null --write-out '%{http_code}' \
    "http://127.0.0.1:8000/api/collocations?term=$(printf %s "$natsume_ted_noun" | jq -sRr @uri)&pos=noun&rankBy=raw&corpusId=$natsume_corpora&limitPerParticle=200")
  test "$natsume_status" = 200
done
```

Record the chosen noun/particle/verb and response byte count in the release record. Also confirm a request for 151 items with all three corpora returns the documented 400 envelope while one- or two-corpus requests may use up to 200.

- [ ] **Step 6: Publish, restart, and run live smoke**

Stop the candidate server, then:

```bash
natsume_candidate_artifact=$(tail -n 1 data/ted-artifact-path-20260814.txt)
nix run .#build-corpus -- publish "$natsume_candidate_artifact" deploy
nix run .#build-corpus -- current deploy
```

Restart the packaged server from `deploy/current`. Verify readiness reports the new instance ID; `/api/corpora` lists TED; and the curated TED search/example works. With that server still running, execute this live-browser smoke:

```bash
nix develop .#frontend --command bash -lc 'cd natsume-frontend && node --input-type=module' <<'JS'
import { chromium } from '@playwright/test';

const browser = await chromium.launch();
for (const width of [375, 1280]) {
  const page = await browser.newPage({ viewport: { width, height: 844 } });
  await page.goto('http://127.0.0.1:8000/');
  await page.getByText('TED Talks').first().waitFor();
  await page.getByText('No single license applies', { exact: false }).waitFor();
  await page.getByText('no license grant is asserted', { exact: false }).waitFor();
  await page.getByRole('link', { name: 'Contact' }).waitFor();
  const region = page.getByRole('region', { name: 'Particle collocations' });
  await region.scrollIntoViewIfNeeded();
  await page.evaluate(() => window.scrollBy(0, 600));
  await region.focus();
  await page.keyboard.press('ArrowRight');
  if ((await region.evaluate((element) => element.scrollLeft)) <= 0) {
    throw new Error(`particle region did not scroll at ${width}px`);
  }
  await page.close();
}
await browser.close();
JS
```

Expected: both widths pass while the page is vertically displaced; the footer names all three sources, mixed terms, unresolved TED permission, and contact.

- [ ] **Step 7: Commit only the release record**

```bash
git add docs/releases
git status --short
git commit -m "docs: record three-corpus production release"
```

Before committing, verify `git status --short` does not stage `data/`, `artifacts/`, a database, `AGENDA.md`, or `container.nix`.

---

### Task 10: Retire completed design scaffolding

**Files:**
- Delete: `docs/superpowers/specs/2026-08-14-ui-restoration-design.md`
- Delete: `docs/superpowers/specs/2026-08-14-ted-inclusion-design.md`
- Delete: `docs/superpowers/plans/2026-08-14-ui-restoration.md`
- Delete: `docs/superpowers/plans/2026-08-14-ted-inclusion.md`

**Interfaces:**
- Consumes: successful UI verification, selected three-corpus artifact, and the permanent release record.
- Produces: repository documentation whose durable facts live in README, source lock, recoverability record, notices, tests, and the release record rather than implementation scaffolding.

- [ ] **Step 1: Confirm every durable decision has a permanent owner**

Run:

```bash
rg -n "iwslt2017-ja-en-training-v1|natsume-source-content-v1|owner-accepted-unresolved-permission" README.md docs/corpus-sources.lock.json docs/corpus-recoverability.md docs/releases corpus-notices src tests
rg -n "Particle collocations|favicon.png|each_key_duplicate|No single license applies" natsume-frontend tests docs/releases
```

Expected: the TED source/hash/risk decisions and UI regression contracts are present outside `docs/superpowers/`.

- [ ] **Step 2: Remove the completed specs and plans**

Use `apply_patch` to delete exactly the four files listed above. Do not remove the `docs/superpowers/` directory recursively and do not touch unrelated documentation.

- [ ] **Step 3: Commit scaffolding retirement**

```bash
git add docs/superpowers/specs/2026-08-14-ui-restoration-design.md docs/superpowers/specs/2026-08-14-ted-inclusion-design.md docs/superpowers/plans/2026-08-14-ui-restoration.md docs/superpowers/plans/2026-08-14-ted-inclusion.md
git commit -m "docs: retire completed UI and TED scaffolding"
```
