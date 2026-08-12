# Corpus Recoverability Gate

**Date:** 2026-08-12
**Gate status:** Blocked
**Scope:** Evidence for the legacy JNLP, TED, and Wikipedia corpora

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

All inspection used DuckDB read-only connections. No conversion has begun. A
distinct read-only local safety copy now exists outside the repository at
`/persistent/home/bor/Backups/natsume-simple/corpus-7326a9fa1d3f5231d83e46e8543d20139bac0ee17aa7570434e5e62ec05c0eca.db`.
It is stored on the persistent `btrfs` filesystem backed by `/dev/nvme3n1p2`,
and its adjacent `SHA256SUMS` check passes. This protects against accidental
modification, deletion, and tmpfs/reboot loss, but it is on the same physical
machine and is not the durable off-machine backup required before conversion.
The operator must still name that durable target before any conversion command
may read the seed for export.

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

The full archive conversion ran twice with an empty environment except the
pinned `nkf 2.1.5` and `pandoc 3.7.0.2` binaries. Of 634 LaTeX members, 549
converted successfully, 85 failed deterministically, and the successful output
totalled 27,861,043 bytes. The canonical manifest of successful output hashes,
sizes, and failed member paths has SHA-256
`14a23fccdae0ec9c3008ca9ffddffff96d6297c2c7973fb1e7c5de18f61f34dc`.
The second run had zero output or disposition mismatches.

The metadata workbook contains 837 rows for the 634 archive members. Every one
of the 459 legacy JNLP source titles maps uniquely to a member and converts
successfully with the declared toolchain. The remaining members contain 90
additional successful conversions and all 85 failures; they were never part of
the legacy corpus. The planned release therefore uses an explicit ordered list
of the 459 legacy-equivalent member identities with SHA-256
`0578102b0b4d719eb1ade968499b2d2778986b9d89aa299d402d844c8673d0c1`.
It does not silently add the other 90 convertible papers.

As a content check, an 81-code-point body sentence from
`V01/V01N01-03.tex`, identified by SHA-256
`5b55c4cf3e73d8a32eae9d58732559d3a80220bed1eda1be1bda31825f7aa2b2`,
occurs verbatim in the converted plaintext. No corpus text is stored in this
record or the source lock.

## Japanese Wikipedia 2023-11-01

**Disposition:** Technical reacquisition pass; public attribution remains a
release requirement.

The exact input named by the legacy loader is available as 15 Parquet shards at
Hugging Face revision
[`b8e579a0c09383e0e254c9980d56833d16048707`](https://huggingface.co/datasets/wikimedia/wikipedia/commit/b8e579a0c09383e0e254c9980d56833d16048707).
The source lock records every file's size and SHA-256. Acquisition can download
those explicit immutable URLs and a local adapter can read `id`, `url`, `title`,
and `text` with a Parquet reader; it does not execute a dataset loader script.

The pinned first shard was downloaded and independently checked at 611,504,422
bytes with SHA-256
`4751c14478e712fd637bd83c2cf3537b0e299ea5115e9a78ddededf42f34c29d`.
Applying the legacy prefilter to its first 1,000 rows selects exactly 971 unique
titles, matching all 971 legacy Wikipedia sources with no missing or additional
title. All 216,568 legacy Wikipedia sentences occur verbatim in the pinned
article text for their matched source. The ordered upstream article-ID list has
SHA-256 `249dc639f646da4db3711571ea97231a22421c22a5a90ebedf86e7ee3b471991`;
the hash-only observation manifest has SHA-256
`2cb160d13631d65359b9490071ad195095018d6e8eecc878c4a950003b2c0164`.
This proves the source selection and legacy identity reconciliation without
retaining article text in the repository.

The historical Wikimedia XML dump is no longer retained and the cleaning
implementation used to create these Parquet files is not fully pinned. The
checksummed Parquet bytes are therefore the source artifact; this project must
not claim that it can reconstruct them from the original dump.

The [Wikimedia dump license guide](https://dumps.wikimedia.org/legal.html) says
text is generally available under CC BY-SA 4.0 and GFDL, subject to the
controlling Terms of Use and content-specific exceptions. A public service must
retain article identity/URL and provide attribution, license, modification, and
takedown information. Because the serving database contains extracted and
segmented Wikipedia text, this project conservatively treats the published
database and corpus-derived content as CC BY-SA 4.0 rather than MIT. The
repository's software remains MIT. The upstream dataset metadata's older
license labels are recorded as source metadata, not treated as the controlling
legal conclusion.

## TED / IWSLT

**Disposition:** Excluded from the planned public release; synthetic local
fixtures only.

TED accounts for 224,898 of 226,328 legacy source rows but only 233,222
sentences: the loader created approximately one pseudo-source per subtitle
segment. It also accounts for 443,875 of 924,763 legacy collocations. Carrying
those rows into schema v1 would preserve neither the intended document-level
source model nor efficient source joins. The planned public release therefore
contains JNLP and Wikipedia. Its corpus selector and per-million mean operate
over those two corpora; TED is not silently treated as a pending third public
corpus.

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
a legacy-backed mapping. If TED is licensed and deliberately reintroduced, its
adapter must model a talk as `SourceDocument`, subtitle segments as ordered text
units/sentences, and the stable upstream talk ID as `external_id`. That is an
explicit granularity and identity correction, not a legacy-preserving rebuild.

Publication permission is unresolved. The 2017 archive names TED copyright and
CC BY-NC-ND 3.0; upstream WIT³ metadata conflicts between BY-NC and BY-NC-ND.
The [CC BY-NC-ND deed](https://creativecommons.org/licenses/by-nc-nd/4.0/)
does not permit distribution of adapted material. TED's current
[Terms of Use](https://www.ted.com/about/our-organization/our-policies-terms/ted-com-terms-of-use)
also say that educational use does not include external research datasets and
that dataset/analysis/ML uses require a separate written license. Whether an
earlier grant controls these archived inputs needs qualified review or written
TED permission. Until then no public or locally distributed artifact may include
TED-derived data; tests use synthetic TED-shaped fixtures only.

The unverified 1.67 GB WIT³ representative extraction and a legacy identity
mapping are unfinished engineering, not owner inputs. They are removed from the
critical path by the two-corpus product decision. Both become mandatory before
any future decision to reintroduce TED.

## Gate conclusion

Gate 3A is not yet passed for the planned JNLP + Wikipedia public release.

### Owner or external decisions

- the legacy seed lacks the required durable verified backup;
- the exact local JNLP archive lacks a resolvable durable location and stronger
  exact-snapshot license evidence.

### Engineering disposition

No Gate 3A source-adapter engineering remains for the planned corpus identities.
The full JNLP conversion audit, the explicit 459-member selection, and the
Wikipedia source/legacy reconciliation have recorded evidence. Production
builder implementation, NLP extraction, schema validation, and query
reconciliation remain Spec 3 work; they cannot begin against production inputs
until the owner or external decisions above are closed.

### Conditional TED work

TED is not a dependency of the planned release. Reintroducing it would require
qualified license review or written permission, the 1.67 GB WIT³ extraction,
and talk-level identity reconciliation.

Schema-v1 fixture work may continue. Production corpus conversion, replacement,
and public publication remain blocked until every applicable JNLP/Wikipedia item
above has recorded evidence.
