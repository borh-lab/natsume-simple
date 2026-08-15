# Teaching and Hashing Simplification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the obsolete corpus-loader teaching fork and unused NLP setup, make the executable teaching examples exercise the production pipeline, and stream ordinary file checksums with Python's standard library without changing artifact semantics.

**Architecture:** Keep one production path for adaptation, segmentation, extraction, and persistence. `data.py` becomes the lightweight home of Japanese filtering and sentence preparation; teaching tests call the same adapters and extraction functions as production. Replace only ordinary whole-file SHA-256 reads with `hashlib.file_digest`; preserve every framed/canonical identity hash exactly.

**Tech Stack:** Python 3.12, pytest, spaCy/GiNZA captured token observations, DuckDB, Nix flakes.

## Global Constraints

- Preserve the public HTTP API, DuckDB schema, manifest structure, artifact identity inputs, rejection policy, and model-selection behavior.
- Do not change `source_content_sha256`, `source_manifest_sha256`, `canonical_article_ids_sha256`, or random artifact-instance ID generation.
- Do not introduce a hashing helper or generic teaching abstraction; use the standard primitive directly at the four ordinary-file call sites.
- Keep `data.py` importable without Polars, Pydantic, Torch, or wtpsplit.
- Characterization tests land before behavior-preserving deletion.
- Do not stage or modify owner files such as `AGENDA.md`.

---

## Task 1: Replace the legacy teaching fork with production examples

**Files:**

- Modify: `tests/test_builder_cli.py`
- Modify: `tests/test_teaching_examples.py`
- Modify: `tests/test_models.py`
- Modify: `src/natsume_simple/data.py`
- Modify: `src/natsume_simple/builder_cli.py`
- Modify: `src/natsume_simple/pattern_extraction.py`

- [ ] **Step 1: Move the segmentation contract test to the intended module and confirm RED**

  In `tests/test_builder_cli.py`, import `natsume_simple.data` and change `test_split_japanese_sentences_keeps_language_policy_outside_segmentation` to call `data.split_japanese_sentences(...)`. Retain its exact assertions:

  ```python
  observations = Counter()
  assert list(
      data.split_japanese_sentences(
          ("日本語の段落です。\nEnglish paragraph.",),
          splitter=Splitter(),
          observations=observations,
      )
  ) == ["日本語の文章です。"]
  assert observations == {"candidate": 2, "retained": 1, "dropped": 1}
  ```

  Run:

  ```bash
  nix develop .#test --command pytest tests/test_builder_cli.py::test_split_japanese_sentences_keeps_language_policy_outside_segmentation -vv
  ```

  Expected: FAIL because `natsume_simple.data` has no `split_japanese_sentences` yet.

- [ ] **Step 2: Rewrite the teaching examples against production boundaries and confirm RED**

  In `tests/test_teaching_examples.py`:

  - replace imports of `BaseCorpusLoader`, `GenericCorpusLoader`, and `process_sentence`;
  - import `SourceDocument`, `SentenceRecord`, `adapt_ted_iwslt_archive`, `extract_collocations`, `source_content_sha256`, and `split_japanese_sentences`;
  - make `test_source_adaptation_example` create a tiny ZIP member named `ja-en/train.tags.ja-en.ja` containing one `<doc>` with talk ID `lesson-1`, title `教材`, and text unit `教材を読む。`;
  - assert the adapter returns corpus `ted`, no rejections, and exactly one `SourceDocument` whose identity, metadata, text units, and `source_content_sha256(("教材を読む。",))` match;
  - make `test_segmentation_example` call `split_japanese_sentences` with the existing fixture splitter and assert `第一文。`, `第二文。`, `第三文。`;
  - make `test_occurrence_construction_example` pass `SentenceRecord(("lesson", "source-1"), 0, "ことを説明するならば")` and the existing captured parsed doc to `extract_collocations(..., extractor_id="teaching-extractor")`;
  - assert no rejections and one exact `CollocationOccurrence` for `こと / を / 説明する`, spans `(0, 2)`, `(2, 3)`, `(3, 10)`, and extractor ID `teaching-extractor`.

  Keep all seven names in `TEACHING_BOUNDARIES` and `teaching_boundaries.toml` unchanged.

  Run:

  ```bash
  nix develop .#test --command pytest tests/test_teaching_examples.py -vv
  ```

  Expected: FAIL because segmentation has not moved yet and the old teaching-only path is still present.

- [ ] **Step 3: Move sentence preparation into lightweight `data.py`**

  Move `split_japanese_sentences` from `builder_cli.py` to `data.py` without changing its paragraph splitting, Japanese filter, generator behavior, or observation counters. Add only the imports it needs:

  ```python
  from collections import Counter
  from collections.abc import Iterable
  ```

  Keep `is_japanese` beside it. Import the function normally from `data.py` in `builder_cli.py`; this must remain safe because `data.py` will no longer load any NLP model or heavy dependency at import time.

- [ ] **Step 4: Delete the legacy loader fork and its tests**

  Delete `CorpusEntry`, `BaseCorpusLoader`, and `GenericCorpusLoader` from `data.py`. Remove now-unused logging, `Iterator`, `Path`, Polars, Pydantic, Torch, and wtpsplit imports. Delete the tests in `tests/test_models.py` that exist only for those loaders, retaining the real `test_model_loading` gate.

  Delete `process_sentence` from `pattern_extraction.py` after the teaching occurrence test uses `extract_collocations` instead. Confirm there are no remaining consumers:

  ```bash
  rg -n '\b(CorpusEntry|BaseCorpusLoader|GenericCorpusLoader|process_sentence)\b' src tests
  ```

  Expected: no matches.

- [ ] **Step 5: Run focused GREEN checks**

  ```bash
  nix develop .#test --command pytest \
    tests/test_builder_cli.py::test_split_japanese_sentences_keeps_language_policy_outside_segmentation \
    tests/test_teaching_examples.py \
    tests/test_models.py \
    -m 'not nlp_model' -vv
  nix build .#checks.x86_64-linux.source-quality --print-build-logs
  ```

  Expected: all model-free tests pass; the marked model-loading test is deselected; source quality passes.

- [ ] **Step 6: Commit the production teaching path**

  ```bash
  git add src/natsume_simple/data.py src/natsume_simple/builder_cli.py \
    src/natsume_simple/pattern_extraction.py tests/test_builder_cli.py \
    tests/test_teaching_examples.py tests/test_models.py
  git commit -m "refactor: teach the production corpus pipeline"
  ```

---

## Task 2: Remove unused NLP setup and the redundant iterator wrapper

**Files:**

- Modify: `src/natsume_simple/pattern_extraction.py`

- [ ] **Step 1: Establish the characterization gate**

  Run the extraction tests before editing:

  ```bash
  nix develop .#test --command pytest \
    tests/test_pattern_extraction.py tests/test_teaching_examples.py \
    -m 'not nlp_model' -vv
  rg -n '\bload_nlp_model\b' src tests
  ```

  Expected: tests pass and the search reports only the function definition.

- [ ] **Step 2: Delete unused setup**

  Delete `load_nlp_model` and the module-level Torch import. Keep the module-level spaCy import because doctests and type references still use it.

- [ ] **Step 3: Remove only the redundant `chain(...)` layer**

  In `normalize_verb_span`, replace:

  ```python
  list(chain(takewhile(...)))
  ```

  with:

  ```python
  list(takewhile(...))
  ```

  Remove `chain` from the itertools import. Do not otherwise rewrite normalization or extraction logic.

- [ ] **Step 4: Run focused GREEN checks and commit**

  ```bash
  nix develop .#test --command pytest \
    tests/test_pattern_extraction.py tests/test_teaching_examples.py \
    -m 'not nlp_model' -vv
  nix build .#checks.x86_64-linux.source-quality --print-build-logs
  rg -n '\b(load_nlp_model|torch|chain)\b' src/natsume_simple/pattern_extraction.py
  ```

  Expected: tests and quality pass; the search has no matches.

  ```bash
  git add src/natsume_simple/pattern_extraction.py
  git commit -m "refactor: remove unused extraction setup"
  ```

---

## Task 3: Stream ordinary file hashes with `hashlib.file_digest`

**Files:**

- Modify: `tests/test_builder_cli.py`
- Modify: `tests/test_artifact_builder.py`
- Modify: `tests/test_artifact_registry.py`
- Modify: `src/natsume_simple/builder_cli.py`
- Modify: `src/natsume_simple/artifact_builder.py`
- Modify: `src/natsume_simple/artifact_validation.py`
- Modify: `src/natsume_simple/release_inputs.py`

- [ ] **Step 1: Add a RED test for streaming model-path hashes**

  Extend `test_path_sha256_covers_relative_names_and_contents` with `monkeypatch`. Compute the expected digest before patching, then replace `Path.read_bytes` with a function that fails the test. Assert both a single file hash and the directory hash:

  ```python
  expected_file = hashlib.sha256(b"first").hexdigest()
  monkeypatch.setattr(
      Path,
      "read_bytes",
      lambda path: pytest.fail(f"read_bytes used for {path}"),
  )
  assert builder_cli.path_sha256(model / "a") == expected_file
  assert builder_cli.path_sha256(model) == expected.hexdigest()
  ```

  Run:

  ```bash
  nix develop .#test --command pytest tests/test_builder_cli.py::test_path_sha256_covers_relative_names_and_contents -vv
  ```

  Expected: FAIL because both current paths call `Path.read_bytes`.

- [ ] **Step 2: Add RED tests for artifact build and validation**

  Add `test_build_artifact_streams_database_checksum` in `tests/test_artifact_builder.py`. Patch `Path.read_bytes` to fail for `.duckdb` files, build the normal fixture, and assert its manifest contains a 64-character `databaseSha256`.

  Add `test_publish_streams_database_checksum_validation` in `tests/test_artifact_registry.py`. Build the fixture first, then patch `Path.read_bytes` to fail for `.duckdb`, publish it, and assert `current_artifact(deploy)` resolves to the fixture.

  Run:

  ```bash
  nix develop .#test --command pytest \
    tests/test_artifact_builder.py::test_build_artifact_streams_database_checksum \
    tests/test_artifact_registry.py::test_publish_streams_database_checksum_validation \
    -vv
  ```

  Expected: both tests FAIL at the current whole-file database reads.

- [ ] **Step 3: Replace artifact checksum reads directly**

  In `artifact_builder.py` and `artifact_validation.py`, open `corpus.duckdb` in binary mode and call:

  ```python
  hashlib.file_digest(source, "sha256").hexdigest()
  ```

  Preserve the validator's existing bounded `database_unreadable` error handling.

- [ ] **Step 4: Replace release-input and model-path hashing directly**

  In `release_inputs.verify_file`, replace the manual chunk loop with `hashlib.file_digest(source, "sha256")`; preserve the size check and all current error reasons.

  In `builder_cli.path_sha256`:

  - hash a file with `hashlib.file_digest(source, "sha256")`;
  - preserve directory ordering and exact `relative-name + NUL + bytes` framing;
  - after updating the shared directory digest with the relative name and NUL, stream each file into that same digest with:

    ```python
    with item.open("rb") as source:
        hashlib.file_digest(source, lambda: digest)
    ```

  Do not add a wrapper and do not alter canonical JSON or source-content hashes.

- [ ] **Step 5: Run focused GREEN and regression checks**

  ```bash
  nix develop .#test --command pytest \
    tests/test_builder_cli.py::test_path_sha256_covers_relative_names_and_contents \
    tests/test_artifact_builder.py \
    tests/test_artifact_registry.py \
    tests/test_release_inputs.py \
    -vv
  rg -n 'read_bytes\(' src/natsume_simple
  ```

  Expected: tests pass and production code has no ordinary file-hash `read_bytes` calls.

- [ ] **Step 6: Commit the hashing simplification**

  ```bash
  git add src/natsume_simple/builder_cli.py src/natsume_simple/artifact_builder.py \
    src/natsume_simple/artifact_validation.py src/natsume_simple/release_inputs.py \
    tests/test_builder_cli.py tests/test_artifact_builder.py \
    tests/test_artifact_registry.py
  git commit -m "refactor: use standard streaming file hashes"
  ```

---

## Task 4: Verify the full system and retire execution scaffolding

**Files:**

- Modify: `docs/superpowers/specs/2026-08-15-teaching-and-hashing-simplification-design.md`
- Delete: `docs/superpowers/plans/2026-08-15-teaching-and-hashing-simplification.md`

- [ ] **Step 1: Run the complete default check graph**

  ```bash
  nix flake check --print-build-logs
  ```

  Expected: source quality, backend tests, frontend checks, Playwright, packaging, builder smoke, edge configuration, closure, server smoke, and the container build all pass.

- [ ] **Step 2: Run the release/scheduled NLP-model gate**

  ```bash
  nix build .#nlp-model-integration --print-build-logs
  ```

  Expected: GiNZA model loading plus the normalization/extraction behavior matrix pass.

- [ ] **Step 3: Audit scope and repository hygiene**

  ```bash
  git diff --stat 3d6541a..HEAD
  git status --short
  rg -n '\b(CorpusEntry|BaseCorpusLoader|GenericCorpusLoader|process_sentence|load_nlp_model)\b' src tests
  find . -type f -size +10M -not -path './.git/*' -printf '%p %s\n'
  ```

  Expected: only the approved Python/tests/docs changed; the removed symbols have no matches; no corpus database, archive, or model was added; `AGENDA.md` remains untouched and untracked.

- [ ] **Step 4: Record implementation evidence and retire the plan**

  Update the design spec status to `Implemented`, list the landed commits, and record the two verification commands and their outcomes. Delete this execution plan because it is temporary task scaffolding; retain the design spec as the durable decision record.

- [ ] **Step 5: Commit lifecycle documentation**

  ```bash
  git add docs/superpowers/specs/2026-08-15-teaching-and-hashing-simplification-design.md
  git add -u docs/superpowers/plans/2026-08-15-teaching-and-hashing-simplification.md
  git commit -m "docs: record teaching simplification delivery"
  ```
