# TED Corpus Inclusion Design

**Date:** 2026-08-14  
**Status:** Revised after review; awaiting approval

## Purpose and decision

Include TED-derived Japanese subtitles in both local and publicly hosted corpus
artifacts. This supersedes the current JNLP + Wikipedia product disposition.
The owner explicitly accepts proceeding while TED redistribution permission is
unresolved. Engineering records that decision and the source evidence; it does
not describe the content as permission-cleared.

Generated databases and source archives remain release assets outside Git. Git
contains only acquisition metadata, adapters, tests, notices, and release
records.

## Inputs

Use the immutable IWSLT 2017 Japanese-English candidate already recorded in
`docs/corpus-sources.lock.json`, at revision
  `c18a4f81a47ae6fa079fe9d32db288ddde38451d`, SHA-256
  `a923cdaa5632e55a94e799d31468c58fd2eab290a34f8b950a562f6ff20046b6`.

Acquisition downloads this data file directly and verifies size and SHA-256.
It does not call `datasets.load_dataset`, execute remote dataset code, or place
the archive in a Nix check or server closure.

The locked 1.67-GB WIT³ archive at revision
`39479634113d3d716305d294dfa5581fa29df496` and SHA-256
`4f3d977f743e330d566024420ea53ca4db50649191fd350f8e9a60d070b13dff`
remains provenance and format evidence, not a production input.

## Format and overlap spike result

Both locked archives were downloaded and their recorded sizes and SHA-256 values
verified on 2026-08-14. The latest WIT³ Japanese XML snapshot contains 2,001
talks and 548,678 `<seekvideo>` fragments. IWSLT 2017 training contains 1,863
`<doc>`-bounded talks and 223,108 line-ordered Japanese rows. Their talk-ID
overlap is 1,802; WIT³ has 199 IDs absent from training and training has 61 IDs
absent from WIT³. None of the overlapping talks has an identical segment list or
identical concatenated text because the releases use materially different
caption segmentation.

The difference is also syntactic, not merely duplicative: WIT³'s 548,678
caption-timed fragments are roughly 2.5 times the 223,108
translation-aligned, sentence-oriented IWSLT training rows. The current splitter
treats text-unit boundaries as hard paragraph boundaries and does not merge
across them, so using the WIT³ fragments would freeze truncated caption syntax
before dependency parsing. IWSLT training rows are the appropriate text-unit
granularity for the existing pipeline.

The historical WIT³ dataset loader did not expose talk or segment IDs as fields
and, due to its nested-XML condition, contributed only titles and descriptions.
IWSLT 2017 supplied the 223,108 subtitle rows that form nearly all of the legacy
TED corpus. Combining the latest raw WIT³ transcript with IWSLT training would
therefore double-count differently segmented content rather than restore
existing data.

The production selection policy is consequently
`iwslt2017-ja-en-training-v1`: IWSLT 2017 training is the sole TED text input.
WIT³ remains inspected provenance and format evidence; it is not merged into
`ted` and is not acquired by the release-input command.

## Adapter contract

The adapter reads the exact `ja-en/train.tags.ja-en.ja` member directly from the
local ZIP using Python archive and text primitives. It rejects a missing or
duplicate named member and never extracts archive paths to disk.

For each `<doc>` in `train.tags.ja-en.ja`:

- a TED talk is one `SourceDocument`;
- `<talkid>` is its `external_id`;
- the Japanese title, speaker/author, source URL, and publisher `TED Conference
  LLC` are retained; `year` remains absent because the training metadata does
  not carry the talk year;
- non-tag Japanese rows in file order become ordered text units; and
- a talk without subtitle text is a counted, talk-level rejection subject to the
  configured limits.

A missing talk ID, a duplicate talk ID, or conflicting metadata is an identity
violation and aborts adaptation immediately. Identity violations are never
admitted as bounded rejection counts.

This intentionally corrects the legacy loader, which usually fell back to a
process-randomized hash per translated segment and produced approximately one
source per subtitle row. Legacy pseudo-source IDs are not preserved.

The adapter returns the same `AdaptationResult` used by JNLP and Wikipedia. The
existing Japanese sentence filter, SaT segmentation, GiNZA extraction,
immutable artifact builder, and schema remain unchanged. `AdaptationResult`
rejections remain talk-denominated because the existing enforcement denominator
is accepted talks plus rejected talks. Metadata and tag lines are archive syntax,
not rejected subtitle segments; blank lines are ignored. The adapter does not
apply a second Japanese-language policy. Input inspection and `identityInputs`
record accepted-talk and subtitle-text-unit counts, while the existing
post-segmentation `is_japanese(..., min_length=5)` policy remains the sole text
filter.

All three adapters use one source-content hash, recorded in `identityInputs` as
`sourceContentHash: "natsume-source-content-v1"`. It is SHA-256 over that ASCII
domain prefix followed by a zero byte, then for each ordered text unit its
unsigned eight-byte big-endian UTF-8 byte length followed by its UTF-8 bytes.
JNLP and Wikipedia each supply their existing single text unit; TED supplies its
ordered subtitle rows. This avoids delimiter ambiguity, makes every manifest
source hash recomputable under one declared scheme, and intentionally gives the
new artifact new JNLP and Wikipedia content hashes without changing old
immutable artifacts.

## Builder and release flow

The release-input loader and acquisition command expand from two sources to the
three selected files: JNLP, Wikipedia, and IWSLT 2017. The IWSLT 2017 lock entry
uses `status: "ready"`, `plannedPublic: true`, and `servingCorpusId: "ted"`;
WIT³ remains `status: "excluded"` and non-selected. The planned artifact lists
`["jnlp", "ted", "wiki"]`. Input inspection validates the TED archive and
reports accepted talks, subtitle text units, and bounded talk-level rejection
reasons before loading NLP models.

The build command accepts one verified local IWSLT archive as an explicit path,
and that path requires `--source-lock` just as Wikipedia input does. When
supplied, it adds one corpus record with ID `ted` and label `TED Talks`, and
records the file, `iwslt2017-ja-en-training-v1`, and observed counts in
`identityInputs`. Local builds may still omit TED by omitting the path.

Production release checking expects exactly `jnlp`, `ted`, and `wiki`. New code
in `release_check` asserts that the manifest's `identityInputs.sourceFiles`
entry for TED exactly matches the selected lock entry's corpus ID, filename,
size, and SHA-256. It does not add a frozen talk-ID manifest: unlike Wikipedia,
the complete named TED member is consumed rather than selecting a subset from a
larger source. Release checking also requires the declared
`natsume-source-content-v1` scheme, updated notices, and the existing structural
checks. Publication still selects an external immutable artifact through
`deploy/current`; no database is added to Git or a Nix closure.

With three selected corpora, the existing API response budget applies. The
default `limitPerParticle=100` remains valid and no API shape changes. The maximum
with all three selected is 150 under the existing 450 item-corpus budget; the API
ceiling remains 200 for one- and two-corpus selections.

## Licensing and attribution

The source lock changes IWSLT 2017 from `excluded` to an owner-accepted public
input and retains all conflicting evidence:

- the 2017 archive names TED copyright and CC BY-NC-ND 3.0;
- WIT³ metadata conflicts between BY-NC and BY-NC-ND; and
- current TED terms state that dataset and analysis uses require separate written
  permission.

The artifact is a collection under mixed terms, not a combined work offered under
one project-wide content license. The source lock changes `gateStatus` to
`owner-accepted-unresolved-permission` and replaces the single artifact-wide SPDX
expression with `singleLicenseAsserted: false` plus per-corpus conclusions: JNLP
is CC BY 4.0, Wikipedia is subject to CC BY-SA 4.0 and applicable Wikimedia
terms, and TED has `spdxExpression: null` with `status: "no-grant-asserted"`.
The TED entry records that inclusion is an owner decision pending permission.

`LICENSE-CONTENT.txt`, `ATTRIBUTION.md`, the public footer, and the release record
must name TED, link the source and applicable evidence, describe subtitle
segmentation/extraction as modifications, and state both that no single license
is asserted for the collection and that TED inclusion reflects an owner decision
while permission remains unresolved. Existing JNLP and Wikipedia notices remain
intact. This record is not legal clearance.

## Validation

Before a production build:

1. Verify the IWSLT archive size and SHA-256 value.
2. Run representative extraction and assert talk-level grouping, file-order
   segments, metadata, and content-hash framing. Run a deterministic sample of
   talk IDs through the pinned SaT model and Japanese filter, recording input
   text units, candidate sentences, retained sentences, and filter drops; use
   this rather than treating the legacy 233,222-sentence total as a fact about
   the new pipeline.
3. Run a complete adapter-only inspection and record talk-level rejection counts
   plus accepted-talk and subtitle-text-unit counts.
4. Confirm repeated adaptation produces identical source identities, ordered
   text units, and content hashes.

After the build:

- artifact validation and release checking pass for three corpora;
- `/api/corpora` exposes `ted` with nonzero sentence and collocation counts;
- a curated TED collocation and example are retrievable through the public API;
- selection and per-million ranking work for TED alone and all three corpora;
- the response-size contract remains below one MiB;
- the footer and artifact notices expose TED attribution and permission status;
- the release record includes actual source, sentence, occurrence,
  `lemma_frequency`, DuckDB-file-size, and sentence-filter candidate/retained/drop
  totals; and
- on the named release host, the same fixed direct-application request family as
  the 20260813 release, updated to select all three corpora, receives 500/500
  successful responses from ten concurrent clients after one warm request per
  URL, with no 5xx and p95 below one second. The procedure and result remain
  release evidence rather than a permanent benchmark framework.

Before the full build, use the adapter count, representative SaT sample, and
measured 20260813 rates (468,527 sentences, 623,798 occurrences, a 176,173,056
byte DuckDB file, 4:02:47 elapsed, and 5.4 GiB peak RSS) to record projected
sentence, occurrence, file-size, wall-time, and peak-memory ranges. The legacy
233,222 TED sentences and 443,875 occurrences are explicitly labelled as
cross-pipeline assumptions, not new-pipeline acceptance values. They currently
suggest about 702,000 sentences, 1.07 million occurrences, a roughly 300 MB
DuckDB file under linear scaling, 6–7 hours, and 8–10 GiB for the three-corpus
build. The sample refines the sentence multiplier; occurrence density, rather
than sentence count alone, drives the memory estimate. The operator confirms
the host has the required memory, the projected artifact is a practical external
release asset, and full restart cost is acceptable.
Progress continues through existing per-document segmentation logs;
checkpointing, streaming infrastructure, and a new orchestration service are not
introduced.

Adding TED also amends `docs/corpus-recoverability.md`: it replaces the excluded
disposition and two-corpus Gate 3A conclusion, records the owner's unresolved-risk
decision, and explicitly reverses the earlier legacy-identity-mapping requirement
because randomized pseudo-source IDs are neither reproducible nor the intended
talk-level model.

## Deliberate omissions

- Do not reproduce randomized legacy TED identities.
- Do not restore `trust_remote_code=True` or depend on mutable dataset names.
- Do not commit source archives or generated DuckDB files.
- Add resumable checkpoints only if a measured failed production build makes
  full restart cost unacceptable after the pre-build estimate has made that cost
  explicit.
