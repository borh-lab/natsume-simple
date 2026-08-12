# Corpus Recoverability Gate

**Date:** 2026-08-12
**Gate status:** Passed
**Scope:** Evidence for the legacy JNLP, TED, and Wikipedia corpora

This record implements Gate 3A from the corpus-builder design. It distinguishes
integrity, technical reacquisition, identity recovery, and permission to publish;
success in one category is not evidence for another. It is an engineering
inventory, not legal advice.

The machine-readable source records are in
[`corpus-sources.lock.json`](./corpus-sources.lock.json). Entries marked
`ready` are approved acquisition inputs for the next build.

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

All inspection used DuckDB read-only connections. The legacy database is
evidence for reconciliation, not a seed of record or a prerequisite for the new
builder.

## JNLP LaTeX corpus

**Disposition:** Technical and publication-source pass.

The local archive is 12,314,348 bytes with SHA-256
`6f71776cf19c3b62a6678d622d52a0431931b4de01eb5da71bf716031f1212c6`.
Although its embedded readme says 2020-03-16, the archive contains the five
Vol.27 No.1 papers and metadata associated with the 2020-06-15 release. Identity
therefore uses the checksum and observed contents, not the stale embedded
version string.

The [ANLP source page](https://www.anlp.jp/resource/journal_latex/) publishes the
corpus under CC BY 4.0 and records both the 2020-06-15 release and the current
2026-06-15 release. The current official archive is 19,166,717 bytes with
SHA-256 `8610f8c391634de11a816950008d63675c52e940c6c0c7df29d7ded005547fdb`.
Its URL is overwritten when ANLP publishes a new release. That is accepted: an
operator deliberately updates the source lock, and the new source checksum and
tool versions become identity inputs for a new artifact.

The legacy archive was converted twice with only the pinned `nkf 2.1.5` and
`pandoc 3.7.0.2` binaries. All 459 legacy JNLP sources map uniquely and convert;
the repeated run had no output or disposition differences. The current 2026
archive was then checked through the same pipeline using
`V12/V12N01-04.tex`; it produced a 78,774-byte plaintext file. This proves the
pipeline across the legacy and current archive formats without freezing future
Pandoc output or the paper set.

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

Gate 3A passes for the planned JNLP + Wikipedia public release. Both corpora have
data-only acquisition paths, locally demonstrated adapters, recorded source and
license evidence, and explicit product dispositions. Exact reproduction of an
older source/tool combination is not required: source bytes, adapter policy, and
tool versions are recorded per artifact so an intentional change creates a new
build rather than invalidating the pipeline.

### Conditional TED work

TED is not a dependency of the planned release. Reintroducing it would require
qualified license review or written permission, the 1.67 GB WIT³ extraction,
and talk-level identity reconciliation.

Spec 3 builder implementation and schema-v1 fixture work may proceed. Public
publication still requires the specified content-license and attribution files.
