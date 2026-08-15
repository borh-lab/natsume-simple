# Teaching and Hashing Simplification

Status: Proposed for implementation

## Purpose

The documented walkthrough should teach the production corpus pipeline rather than a test-only legacy fork. Plain file hashing should use the Python standard-library primitive instead of three independent implementations, and the extraction module should not expose an unused model-loading policy that conflicts with the real builder.

This is a behavior-preserving production refactor. Public HTTP contracts, artifact schema and manifest fields, source-content hash framing, canonical JSON hashes, model identity, corpus selection, and release policy do not change.

## Current Evidence

Production imports only `is_japanese` from `src/natsume_simple/data.py`. `CorpusEntry`, `BaseCorpusLoader`, and `GenericCorpusLoader` are referenced only by `tests/test_models.py` and `tests/test_teaching_examples.py`; no CLI or corpus adapter constructs them. The repository has no external consumers.

`process_sentence` in `pattern_extraction.py` is also test-only. It returns word tuples for the removed mutable word relation alongside NPV tuples, while the production pipeline constructs schema-v1 `CollocationOccurrence` values through `corpus_pipeline.extract_collocations`.

`load_nlp_model` has no caller. The real build loads `ja_ginza` only after input validation and explicitly configures deterministic CPU execution in `builder_cli._build`. Keeping a second automatic GPU/MPS and model-fallback policy makes an unsupported execution path appear live.

Plain file SHA-256 is implemented with `Path.read_bytes()` in artifact creation, artifact validation, and the file branch of `path_sha256`, while release-input verification carries its own chunk loop. The artifact database is approximately one gigabyte, so `read_bytes()` unnecessarily materializes the complete file. Python 3.12 already provides `hashlib.file_digest`.

## Teaching Surface

`data.py` becomes the small Japanese-text preparation module promised by the README. It owns:

- `is_japanese(line, min_length=200)` with its existing behavior and doctests;
- `split_japanese_sentences(text_units, *, splitter, observations=None)`, moved from `builder_cli.py` without changing paragraph splitting, empty filtering, the five-character Japanese policy, or observation-counter semantics.

The move makes `data.py` cheap to import: it no longer imports Polars, Pydantic, Torch, or wtpsplit. `builder_cli` imports the function normally and continues supplying the concrete SaT model and observation counter.

The following legacy teaching structures are removed:

- `CorpusEntry`;
- `BaseCorpusLoader`;
- `GenericCorpusLoader`;
- the generic-loader-only tests in `tests/test_models.py`;
- `process_sentence` and its legacy word-tuple behavior.

The executable boundary inventory remains seven entries. Its examples change as follows:

- source adaptation uses the real `adapt_ted_iwslt_archive` adapter on a tiny local ZIP fixture and asserts the resulting `SourceDocument` identity and text units;
- segmentation calls the production `split_japanese_sentences` function with a fixture splitter;
- occurrence construction calls `extract_collocations` with a captured parsed document and asserts the resulting `CollocationOccurrence` identity, lemmas, spans, and extractor ID.

Japanese filtering, normalization, aggregation, and query-semantics examples retain their current contracts. The count remains a deletion alarm rather than a quota; changing which function demonstrates a boundary is recorded in `teaching_boundaries.toml` only when the target name changes.

## Extraction Cleanup

`load_nlp_model` and its otherwise-unused Torch import are deleted. GiNZA normalization and matching behavior, doctests, captured token observations, and the model-dependent gate remain unchanged.

The redundant `chain(...)` around the single `takewhile(...)` iterator in `normalize_verb_span` may be removed in the same extraction-only commit because it changes neither values nor evaluation order. No further normalization control-flow rewrite is included.

## File Hashing

All plain file byte hashes use `hashlib.file_digest` directly at their current ownership sites:

- artifact database checksum creation;
- artifact database checksum validation;
- release-input verification;
- file and directory-content branches of `builder_cli.path_sha256`.

For a directory model hash, the existing digest is passed through `file_digest` for each file after the same ordered relative name and NUL separator have been added. This preserves the exact byte framing and therefore existing model hashes. No shared hashing module or exported helper is added.

The following distinct hashes are not consolidated or changed:

- `source_content_sha256` length-prefix framing;
- `source_manifest_sha256` canonical JSON;
- `canonical_article_ids_sha256` compact JSON;
- random artifact-instance identity.

## Test Contract

Tests must prove:

1. `split_japanese_sentences` retains paragraph ordering, removes empty splitter output, applies the existing Japanese policy, and records candidate/retained/dropped counts.
2. Every teaching-boundary target is callable and exercises a production function rather than `GenericCorpusLoader` or `process_sentence`.
3. The real adapter example preserves source identity and text units.
4. The real occurrence-construction example preserves noun, particle, verb, spans, source identity, sentence ordinal, and extractor identity.
5. Building and validating an artifact succeeds when `Path.read_bytes` is forbidden for `corpus.duckdb`.
6. `path_sha256` produces the existing file and directory digests when `Path.read_bytes` is forbidden.
7. Locked-source verification retains its size/checksum failures and atomic acquisition behavior.
8. Model-free and model-dependent normalization/extraction gates retain their existing expectations.

## Patch Boundaries

The work lands as three independently revertible changes:

1. replace the legacy teaching fork with production pipeline examples, then remove its last callers and implementations;
2. delete the unused model loader and redundant iterator wrapper;
3. replace plain file-read hashing with `hashlib.file_digest`.

Behavior changes discovered during the work are reported and fixed separately. Tests are not weakened to accommodate deletions; legacy tests disappear only with the test-only behavior they exclusively call.

## Deliberate Omissions

- No generic loader protocol or replacement CSV adapter: there is no present consumer.
- No shared hashing module: the standard-library primitive is already the common abstraction.
- No API router split: the application factory remains the cohesive owner of lifecycle, database access, errors, logging, and query limits.
- No generic pagination abstraction or frontend state refactor.
- No changes to release-specific policy checks, corpus inputs, dependencies, Nix outputs, or artifact files.

## Lifecycle

This document is retained as the decision record. After all focused and full Nix gates pass, its status changes to `Implemented`; the execution plan is retired.
