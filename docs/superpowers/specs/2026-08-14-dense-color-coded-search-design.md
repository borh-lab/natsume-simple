# Dense, Color-Coded Search Interface

Status: Approved for implementation

## Purpose

The search interface should read like a dense spreadsheet rather than a stack of buttons. Grammatical roles must use one semantic color system across the search-type control and highlighted example sentences. Corpus identity must use a separate color system so an ambiguous source title still communicates where an example came from.

## Constraints

- Preserve existing ranking, bar-scale modes, corpus contributions, raw-frequency tooltips, pagination, example loading, highlighting spans, keyboard operation, and dark mode.
- Keep the native form submission and search-controller contracts unchanged.
- Do not add a theme system, runtime palette configuration, component library, or new dependency.
- Corpus identity must not rely on color alone for assistive technology.
- The three-corpus artifact limit remains the source of the corpus palette size.

## Color Ownership

Grammatical roles keep the existing semantic palette:

| Role | Light | Dark |
| --- | --- | --- |
| Noun | blue-600 | blue-400 |
| Particle | red-600 | red-400 |
| Verb | green-600 | green-400 |

The sentence-highlighting module exports these role classes as the single owner. The search-type control consumes the same classes; it does not duplicate equivalent color literals.

Corpus slots use a disjoint palette:

| Slot | Bar/border | Light title | Dark title |
| --- | --- | --- | --- |
| 0 | violet-600 | violet-700 | violet-300 |
| 1 | orange-600 | orange-700 | orange-300 |
| 2 | cyan-600 | cyan-700 | cyan-300 |

The existing corpus-order-to-slot mapping remains authoritative. Bars, example borders, and source titles all consume the same slot. Each visible source title also receives an accessible name of `<corpus label>: <source title>`, so corpus identity is not conveyed only by color.

## Header and Search Type

At medium and wider viewports, the header is a three-column grid with equal flexible side columns:

1. brand aligned left;
2. search/type controls aligned to the viewport center;
3. theme toggle aligned right.

Below the medium breakpoint, brand and theme occupy the first row and the controls occupy a centered, full-width second row.

The native search-direction select is replaced by one compact radio group with two choices:

- `Noun → Particle → Verb` for noun search;
- `Noun ← Particle ← Verb` for verb search.

The three words use the shared grammatical-role colors. Native radio inputs retain keyboard and form semantics; the labels provide the segmented visual surface. The selected choice has a clear light/dark background and focus-visible outline.

## Dense Result Rows

Each collapsed collocation summary targets a 30–32 pixel row:

- one-pixel horizontal separator instead of a rounded card border;
- `px-1 py-1` or less;
- small chevron;
- thinner frequency bar;
- no vertical gap between adjacent rows;
- subtle hover and open backgrounds in both themes.

Expanded examples remain directly beneath their summary at full column width. The example list uses dividers and compact padding instead of separated rounded cards. Paging, retry, identity-error, and exhausted controls remain full-width and visually distinct, but their padding is reduced to match the spreadsheet density.

Opening a row must not indent its examples or move the next result outside the same column flow.

## Component Changes

- `sentence.ts` owns and exports grammatical-role highlight classes.
- `presentation/search.ts` owns corpus bar colors plus slot-indexed title and border classes.
- `SearchControls.svelte` renders the radio-based type selector using the shared role classes.
- `+page.svelte` owns the responsive centered-header grid.
- `ParticleColumn.svelte` passes the existing corpus slot/label information through the collocation component chain.
- `CollocationItem.svelte` renders the compact summary surface.
- `SentenceExamples.svelte` renders corpus-colored titles/borders and compact example rows.

No new store, context, public API field, palette registry, or generic styling component is introduced.

## Test Contract

Browser coverage must fail if any of these observable properties regress:

- the desktop search-control center differs materially from the viewport center;
- mobile controls no longer occupy a centered second row;
- noun, particle, and verb labels in the selector differ from their corresponding sentence-highlight colors;
- source titles from different corpora share one computed color;
- an example title's corpus slot disagrees with its left border and collocation-bar slot;
- corpus colors reuse the grammatical blue/red/green role colors;
- a collapsed result summary exceeds 34 pixels in the fixture viewport;
- adjacent summaries regain vertical gaps or rounded card treatment;
- open/collapsed and focus-visible states stop being distinguishable in light or dark mode;
- source-title accessible names omit the corpus label.

Existing frontend unit, Svelte diagnostic, build, Playwright, and server-smoke gates remain required.

## Deliberate Omissions

- Do not make palettes operator-configurable. Revisit only if a present artifact requires more than three corpora or branding requires runtime theming.
- Do not build a reusable segmented-control component. Revisit when a second segmented control exists.
- Do not synchronize row heights across particle columns; rows represent independently ranked collocations and do not share row identity.
