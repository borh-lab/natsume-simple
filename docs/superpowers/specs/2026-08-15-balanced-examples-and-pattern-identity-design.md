# Balanced Examples and Pattern Identity

Status: Implemented

Implementation evidence: the balanced and composable example-page API contracts, grammatical-pattern browser scenarios, and the Nix `source-quality`, `backend`, `frontend`, `playwright`, `package-frontend`, and `server-smoke` checks pass on the implemented tree.

## Purpose

Example pages should expose evidence from every selected corpus instead of exhausting the alphabetically first corpus before showing the next one. The compact result identity should also describe the grammatical pattern represented by the current search rather than ending with the ambiguous label `Noun` or `Verb`.

## Current Behavior

`GET /api/examples` orders matching occurrences by corpus ID, source ID, sentence ID, spans, and extractor ID. This is a total order, but its corpus-first prefix means an initial five-example page may contain only the first corpus even when other selected corpora have matching evidence.

The endpoint is offset-paginated over an immutable artifact. The frontend initially requests five occurrences and appends later pages of at most twenty using `offset = examples.length`. These request and response shapes are already sufficient; the ordering policy is the defect.

The accepted-result identity currently renders, for example, `648 matches · “こと” · Noun`. The final token names the selected input role but does not show the noun–particle–verb structure being searched.

## Example Ordering

The server owns balancing. The frontend continues to display the response order without regrouping or sampling.

The examples query assigns a one-based row number within each corpus:

```sql
ROW_NUMBER() OVER (
    PARTITION BY src.corpus_id
    ORDER BY src.id, s.id,
             o.n_begin, o.n_end,
             o.p_begin, o.p_end,
             o.v_begin, o.v_end,
             o.extractor_id
) AS corpus_row
```

The outer query orders by `corpus_row, corpus_id`, then applies the existing `LIMIT limit + 1 OFFSET offset`. The window order is total because the occurrence relation's uniqueness constraint covers sentence identity, grammatical triple, spans, and extractor identity; the request fixes the grammatical triple. `corpus_row` is therefore unique within one corpus, and `(corpus_row, corpus_id)` is a total order across selected corpora.

This produces deterministic round-robin pages in canonical corpus-ID order. With three eligible corpora, consecutive results are one from each corpus while all three have evidence. When a corpus is exhausted or has no match, the remaining corpora fill subsequent positions; page capacity is not reserved for unavailable evidence.

Balancing changes presentation order only. It does not deduplicate occurrences, change corpus totals, weight ranking statistics, or imply that the selected corpora contain equal amounts of evidence.

Offset retains its existing meaning over the final ordered result. Concatenating pages yields the same sequence as one larger request, with no duplicates or omissions. The artifact remains immutable, so no cursor or snapshot identity is required.

The existing `limit`, `offset`, `hasMore`, response schema, query timeout, request identity, and frontend append behavior do not change.

## Result Identity

`SearchSummary.svelte` renders the accepted input as a grammatical pattern:

- noun search: `648 matches · “こと”–particle–verb`;
- verb search: `648 matches · noun–particle–“集める”`.

The count label uses `match` for exactly one result and `matches` otherwise.

The searched term is bold in the visible pattern. The placeholders stay lowercase and neutral because they describe roles rather than controls or corpus identity. The `title` retains the compact plain-text pattern. The accessible label expands it semantically rather than asking a screen reader to infer grammar from punctuation:

- noun search: `648 matches; searched noun “こと”; pattern noun, particle, verb`;
- verb search: `648 matches; searched verb “集める”; pattern noun, particle, verb`.

The identity continues to describe only the accepted response. Editing draft controls leaves it unchanged and shows the existing `Not applied` status until a matching response is accepted.

## Failure and Pagination Semantics

- An empty initial page returns `examples = []` and `hasMore = false`.
- An offset at or past exhaustion returns an empty page with `hasMore = false`.
- A corpus with no matching occurrences contributes no placeholder row.
- Later-page request errors and response-identity mismatches retain their existing frontend behavior.
- Repeated requests against one artifact and selection return the same order.

## Test Contract

API tests must prove:

1. Three corpora interleave in canonical corpus-ID order while all have matching evidence.
2. A corpus with fewer matches drops out and the remaining corpora fill the page.
3. Concatenating several offset pages equals one request for the combined prefix, without duplicate occurrence identities or omissions.
4. Repeating a request returns the identical order.
5. Existing empty, exhausted, invalid-offset, response-size, timeout, and selected-corpus contracts remain green.
6. The existing non-empty later-page service benchmark remains below its one-second p95 gate.

Frontend tests must prove:

1. The initial disclosure preserves the server's balanced corpus order.
2. `Load more examples` appends the next server page without client regrouping.
3. Noun and verb searches render the exact patterns above with the searched term emphasized.
4. Draft edits preserve the accepted pattern identity until submission succeeds.

## Deliberate Omissions

- No per-corpus cursor or offset: one immutable total order already supports stable pagination.
- No client-side balancing: it would make different page sizes produce different visible sequences and could not recover examples excluded by the server page.
- No proportional sampling: equal exposure across available corpora is the present product requirement.
- No schema or index change: the query already sorts matching occurrences, and the existing one-second p95 trigger owns future physical-design work.
- No extracted ordering module or pagination abstraction: the SQL order is one cohesive endpoint invariant with no second implementation or caller.

## Architecture Review

The change is protocol design rather than a module-deepening candidate: the service already owns selection, total ordering, pagination, and `hasMore`, so corpus balancing belongs in the same query. The minimum sufficient implementation uses DuckDB's `ROW_NUMBER` primitive and changes no public request or response surface.

The window must rank every matching occurrence before the outer limit, while the previous query could use `TOP_N`. This cost was measured against the deployed artifact's largest result (`必要–が–ある`, 4,020 occurrences across three corpora). At offset 4,000 with a 20-row page, twelve in-process runs, excluding the first two as warm-up, produced a 9.00ms warm median for the corpus-first query and 9.82ms for the balanced query, with both maxima near 10.2ms. This does not earn an index, cache, materialized relation, or new benchmark harness. The existing service benchmark remains the release gate and the one-second p95 threshold remains the revisit trigger.

The test-only three-corpus fixture data belongs beside the API contract test rather than in the shared two-corpus browser fixture, because changing that shared fixture would rewrite unrelated response and color assertions.

## Relationship to Existing Decisions

This document supersedes only the corpus-first example ordering in `2026-08-14-incremental-results-design.md`. That document's total-order requirement, offset-pagination contract, no-total-count decision, and immutable-artifact premise remain in force.

## Lifecycle

This document is the retained decision record. After implementation, its status changes to `Implemented`; the execution plan is retired after all gates pass.
