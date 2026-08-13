# TED Corpus Inclusion Design

**Date:** 2026-08-14  
**Status:** Approved for implementation planning

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

Use the two immutable candidates already recorded in
`docs/corpus-sources.lock.json`:

- WIT³/IWSLT 2014–2016 archive at revision
  `39479634113d3d716305d294dfa5581fa29df496`, SHA-256
  `4f3d977f743e330d566024420ea53ca4db50649191fd350f8e9a60d070b13dff`;
- IWSLT 2017 Japanese-English archive at revision
  `c18a4f81a47ae6fa079fe9d32db288ddde38451d`, SHA-256
  `a923cdaa5632e55a94e799d31468c58fd2eab290a34f8b950a562f6ff20046b6`.

Acquisition downloads these data files directly and verifies size and SHA-256.
It does not call `datasets.load_dataset`, execute remote dataset code, or place
the archives in a Nix check or server closure.

## Adapter contract

The adapter reads local archives using Python archive and XML/text primitives.
Archive extraction rejects absolute paths and parent traversal before writing.

For each release:

- a TED talk is one `SourceDocument`;
- the stable upstream talk/document ID is its `external_id`, namespaced when the
  two archive formats use overlapping identifier domains;
- the talk title, year, speaker/author, publisher, and source URL are retained
  when present;
- Japanese subtitle segments are ordered by their upstream segment number and
  become ordered text units; and
- exact duplicate talk/segment observations may collapse, while conflicting
  observations for one identity reject the input rather than silently choose.

This intentionally corrects the legacy loader, which usually fell back to a
process-randomized hash per translated segment and produced approximately one
source per subtitle row. Legacy pseudo-source IDs are not preserved.

The adapter returns the same `AdaptationResult` used by JNLP and Wikipedia. The
existing Japanese sentence filter, SaT segmentation, GiNZA extraction,
rejection-limit enforcement, immutable artifact builder, and schema remain
unchanged.

## Builder and release flow

The release-input loader and acquisition command expand from two sources to the
four selected files: JNLP, Wikipedia, WIT³, and IWSLT 2017. Input inspection
validates both TED archives and reports accepted talks, subtitle text units, and
bounded rejection reasons before loading NLP models.

The build command accepts the two verified local TED archives as explicit paths.
When supplied, it adds one corpus record with ID `ted` and label `TED Talks`, and
records both files, the adapter policy, and observed counts in
`identityInputs`. Local builds may still omit TED by omitting both paths; passing
only one TED archive is rejected so `ted` never ambiguously names a partial
release.

Production release checking expects exactly `jnlp`, `ted`, and `wiki`, confirms
the locked TED input identities, requires the updated content notices, and
performs the existing structural and endpoint checks. Publication still selects
an external immutable artifact through `deploy/current`; no database is added to
Git or a Nix closure.

With three selected corpora, the existing API response budget applies. The
default `limitPerParticle=100` remains valid and no API shape changes.

## Licensing and attribution

The source lock changes TED from `excluded` to an owner-accepted public input and
retains all conflicting evidence:

- the 2017 archive names TED copyright and CC BY-NC-ND 3.0;
- WIT³ metadata conflicts between BY-NC and BY-NC-ND; and
- current TED terms state that dataset and analysis uses require separate written
  permission.

`LICENSE-CONTENT.txt`, `ATTRIBUTION.md`, the public footer, and the release record
must name TED, link the source and applicable evidence, describe subtitle
segmentation/extraction as modifications, and state that inclusion reflects an
owner decision while permission remains unresolved. Existing JNLP and Wikipedia
notices remain intact. This record is not legal clearance.

## Validation

Before a production build:

1. Verify both archive sizes and SHA-256 values.
2. Run representative extraction for every archive format and assert talk-level
   grouping plus segment order.
3. Run a complete adapter-only inspection and record talk, text-unit, and
   rejection counts.
4. Confirm repeated adaptation produces identical source identities and ordered
   text units.

After the build:

- artifact validation and release checking pass for three corpora;
- `/api/corpora` exposes `ted` with nonzero sentence and collocation counts;
- a curated TED collocation and example are retrievable through the public API;
- selection and per-million ranking work for TED alone and all three corpora;
- the response-size contract remains below one MiB; and
- the footer and artifact notices expose TED attribution and permission status.

The full extraction is expected to be a long-running release operation. Progress
continues through existing per-document segmentation logs; checkpointing,
streaming infrastructure, and a new orchestration service are not introduced.

## Deliberate omissions

- Do not reproduce randomized legacy TED identities.
- Do not restore `trust_remote_code=True` or depend on mutable dataset names.
- Do not commit source archives or generated DuckDB files.
- Add resumable checkpoints only if a measured failed production build makes
  full restart cost unacceptable.

