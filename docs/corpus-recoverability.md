# Corpus Recoverability Gate

**Date:** 2026-08-12
**Gate status:** Blocked
**Scope:** Read-only evidence for the legacy JNLP, TED, and Wikipedia corpora

This record implements Gate 3A from the corpus-builder design. It distinguishes
integrity, technical reacquisition, identity recovery, and permission to publish;
success in one category is not evidence for another. It is an engineering
inventory, not legal advice.

The machine-readable candidates are in
[`corpus-sources.lock.json`](./corpus-sources.lock.json). Entries marked
`blocked` are evidence candidates, not approved acquisition inputs.

## Legacy seed

The only inspected legacy database is the untracked `data/corpus.db`:

- size: 1,001,402,368 bytes;
- SHA-256: `7326a9fa1d3f5231d83e46e8543d20139bac0ee17aa7570434e5e62ec05c0eca`;
- canonical `duckdb_tables().sql` schema fingerprint:
  `9e4fe60f456062646b590118d8bfc46253ae8a8401a2be99a3858575154b8b09`;
- schema: `source`, `sentence`, `lemma`, `word`, `sentence_word`, and
  `collocation`;
- 226,328 sources, 584,185 sentences, 13,419,829 sentence-word rows, and
  924,763 collocations.

| Legacy corpus | Sources | Sentences | Collocations |
| ------------- | ------: | --------: | -----------: |
| TED           | 224,898 |   233,222 |      443,875 |
| Wikipedia     |     971 |   216,568 |      218,323 |
| 自然言語処理  |     459 |   134,395 |      262,565 |

All inspection used DuckDB read-only connections. No conversion has begun. The
required checksum-verified backup outside this repository and outside artifact
retention paths does not yet exist; the operator must name its durable target
before any conversion command may read the seed for export.

## JNLP LaTeX corpus

**Disposition:** Conditional technical pass; blocked on durable source
resolution and exact-snapshot license evidence.

The local archive is 12,314,348 bytes with SHA-256
`6f71776cf19c3b62a6678d622d52a0431931b4de01eb5da71bf716031f1212c6`.
Although its embedded readme says 2020-03-16, the archive contains the five
Vol.27 No.1 papers and metadata associated with the 2020-06-15 release. Identity
therefore uses the checksum and observed contents, not the stale embedded
version string.

The [ANLP source page](https://www.anlp.jp/resource/journal_latex/) publishes the
corpus under CC BY 4.0 and records both the 2020-06-15 release and the current
2026-06-15 release. Its single archive URL is overwritten, however; today it no
longer resolves the local 2020 bytes. A checksum detects substitution but cannot
reacquire missing bytes. Gate passage requires one of:

1. a durable content-addressed copy of the exact local archive plus
   contemporaneous license confirmation; or
2. an explicit decision to adopt, checksum, and durably retain a newer release.

The representative `V01/V01N01-01.tex` conversion ran twice with an empty
environment except the pinned Nix binaries:

- `nkf 2.1.5`;
- `pandoc 3.7.0.2`;
- normalized LaTeX SHA-256:
  `3bdde8203e1207cd4afba90581763672aeae90d0b88df93747a587da2b1437d1`;
- plain-text SHA-256:
  `a3ff9f8dce6067a4ced5fa62097aa9bc86fe6a951361c36b4d7be86ad514f6ce`.

Both executions were byte-identical. This proves the declared conversion path
for one representative paper, not the full 634-file archive.

## Japanese Wikipedia 2023-11-01

**Disposition:** Technical reacquisition pass; public attribution remains a
release requirement.

The exact input named by the legacy loader is available as 15 Parquet shards at
Hugging Face revision
[`b8e579a0c09383e0e254c9980d56833d16048707`](https://huggingface.co/datasets/wikimedia/wikipedia/commit/b8e579a0c09383e0e254c9980d56833d16048707).
The source lock records every file's size and SHA-256. Acquisition can download
those explicit immutable URLs and a local adapter can read `id`, `url`, `title`,
and `text` with a Parquet reader; it does not execute a dataset loader script.

The historical Wikimedia XML dump is no longer retained and the cleaning
implementation used to create these Parquet files is not fully pinned. The
checksummed Parquet bytes are therefore the source artifact; this project must
not claim that it can reconstruct them from the original dump.

The [Wikimedia dump license guide](https://dumps.wikimedia.org/legal.html) says
text is generally available under CC BY-SA 4.0 and GFDL, subject to the
controlling Terms of Use and content-specific exceptions. A public service must
retain article identity/URL and provide attribution, license, modification, and
takedown information. The upstream dataset metadata's older license labels are
recorded as source metadata, not treated as the controlling legal conclusion.

## TED / IWSLT

**Disposition:** Technical sources identified; Gate 3A and public publication
blocked.

The 2014–2016 WIT³ archive and 2017 Japanese-English training archive have
immutable revisions, sizes, and SHA-256 values in the source lock. A local
adapter can use `tarfile`, `zipfile`, and XML/text parsing without executing the
upstream Python loaders. For IWSLT 2017, stripping whitespace and tag lines from
`train.tags.ja-en.ja` produces 223,108 Japanese rows, matching upstream dataset
metadata. The 1.67 GB WIT³ archive has not yet been locally downloaded and its
representative extraction remains unverified.

Reacquisition cannot recreate the legacy source identities. The existing loader
falls back to `hash(example["translation"]["en"])`; Python hash randomization
makes those IDs process-specific. Reacquired sentence content can be compared,
but identity preservation requires conversion from the backed-up legacy seed or
a legacy-backed mapping. The new adapter should use stable upstream talk IDs and
record that as a deliberate identity correction.

Publication permission is unresolved. The 2017 archive names TED copyright and
CC BY-NC-ND 3.0; upstream WIT³ metadata conflicts between BY-NC and BY-NC-ND.
The [CC BY-NC-ND deed](https://creativecommons.org/licenses/by-nc-nd/4.0/)
does not permit distribution of adapted material. TED's current
[Terms of Use](https://www.ted.com/about/our-organization/our-policies-terms/ted-com-terms-of-use)
also say that educational use does not include external research datasets and
that dataset/analysis/ML uses require a separate written license. Whether an
earlier grant controls these archived inputs needs qualified review or written
TED permission. Until then the builder may use a synthetic TED fixture, but no
public artifact may include TED-derived data.

## Gate conclusion

Gate 3A is not passed:

- the legacy seed lacks the required durable verified backup;
- the exact local JNLP archive lacks a resolvable durable location and stronger
  exact-snapshot license evidence;
- WIT³ representative extraction has not been run;
- TED public derived-output permission is unresolved; and
- exact TED legacy identities require conversion or a backed-seed mapping.

Schema-v1 fixture work may continue. Production corpus conversion, replacement,
and public publication remain blocked until the applicable items above have
recorded evidence.
