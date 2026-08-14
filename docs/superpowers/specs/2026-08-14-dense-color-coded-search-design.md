# Dense, Color-Coded Search Interface

Status: Approved for implementation

## Purpose

The search interface should read like a dense spreadsheet rather than a stack of buttons. Grammatical roles must use one semantic color system across the search-type control and highlighted example sentences. Corpus identity must use a separate color system so an ambiguous source title still communicates where an example came from.

## Constraints

- Preserve existing ranking, bar-scale modes, corpus contributions, raw-frequency tooltips, pagination, example loading, highlighting spans, keyboard operation, and dark mode.
- Keep the native form submission and search-controller contracts unchanged.
- Do not add a theme system, runtime palette configuration, component library, or new dependency.
- Corpus identity must not rely on color alone for any user.
- The three-corpus artifact limit remains the source of the corpus palette size.

## Color Ownership

`src/lib/presentation/colors.ts` is the single owner of the two semantic palettes. This is
an earned shared seam rather than a general theme registry: grammatical-role colors have
two present consumers (sentence highlights and the search-type control), and corpus colors
have three (frequency bars, example borders, and example source labels).

Keeping these values in `sentence.ts` would make a text-segmentation module own UI styling;
keeping them in `presentation/search.ts` would make sentence highlighting depend on
collocation bar math. The focused color module avoids both dependency inversions. It exports
only the fixed role map and the fixed corpus-slot records used by this interface.

Grammatical roles keep the existing semantic palette:

| Role | Light | Dark |
| --- | --- | --- |
| Noun | blue-600 | blue-400 |
| Particle | red-600 | red-400 |
| Verb | green-600 | green-400 |

`sentence.ts` and `SearchControls.svelte` consume the same exported role map; neither
duplicates equivalent color literals.

Corpus slots use a disjoint palette:

| Slot | SVG bar | Border | Light title | Dark title |
| --- | --- | --- | --- | --- |
| 0 | `#7c3aed` | violet-600 | violet-700 | violet-300 |
| 1 | `#ea580c` | orange-600 | orange-700 | orange-300 |
| 2 | `#0891b2` | cyan-600 | cyan-700 | cyan-300 |

The existing corpus-order-to-slot mapping remains authoritative. Bars, example borders,
and source labels all consume the same slot. An example starts with the compact visible text
`<corpus label> · <source title>:`. The label and title share the corpus title color, and
the row carries the matching left border. The explicit corpus label means identity is not
conveyed by color alone and is also present in the accessible text without a separate ARIA
override or badge component.

## Header and Search Type

At medium and wider viewports, the header is a three-column grid with equal flexible side columns:

1. brand aligned left;
2. search/type controls aligned to the viewport center;
3. theme toggle aligned right.

Below the medium breakpoint, brand and theme occupy the first row and the controls occupy a centered, full-width second row.

The native search-direction select is replaced by one compact radio group with two choices:

- `Noun → Particle → Verb` for noun search;
- `Noun ← Particle ← Verb` for verb search.

The three words use the shared grammatical-role colors. Native radio inputs retain keyboard
and form semantics; the labels provide the segmented visual surface. The selected choice has
a clear light/dark background and focus-visible outline. The role colors explain grammatical
structure, while the separate violet/orange/cyan palette means corpus contribution only.

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

- `presentation/colors.ts` owns the fixed grammatical-role map and corpus-slot records.
- `sentence.ts` consumes the role map while retaining sole ownership of span validation and
  segmentation.
- `presentation/search.ts` retains bar calculations and consumes corpus-slot SVG colors.
- `SearchControls.svelte` renders the radio-based type selector using the shared role classes.
- `+page.svelte` owns the responsive centered-header grid.
- `ParticleColumn.svelte` passes the existing corpora and slot map through the collocation
  component chain; no context or second corpus-identity map is introduced.
- `CollocationItem.svelte` renders the compact summary surface.
- `SentenceExamples.svelte` resolves each example's own `corpusId` against those values and
  renders the visible corpus label, colored title/border, and compact example row.

No new store, context, public API field, runtime palette registry, or generic styling component is introduced.

### Architecture disposition

- **Type:** Decomplect, followed by a small implementation refactor. Styling values move
  out of sentence segmentation and bar mathematics; neither behavior gains a new protocol.
- **Evidence:** grammatical-role classes currently live privately in `sentence.ts`, corpus
  SVG colors live in `presentation/search.ts`, and the requested selector and example rows
  create second and third consumers. This would be disconfirmed if either requested consumer
  were removed, in which case the corresponding constants should remain local.
- **Hazards:** the module contains immutable values only. It owns no state, time, identity,
  trust boundary, configuration, or runtime selection. Existing role colors and corpus slot
  ordering remain behavior-preservation constraints.
- **Characterization:** unit coverage pins the role map, the three corpus slots, and palette
  disjointness; browser coverage proves the maps reach the selector, highlights, bars, and
  example labels rather than merely testing exported constants.

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
- visible and accessible source text omits the corpus label.

Existing frontend unit, Svelte diagnostic, build, Playwright, and server-smoke gates remain required.

## Deliberate Omissions

- Do not make palettes operator-configurable. Revisit only if a present artifact requires more than three corpora or branding requires runtime theming.
- Do not build a reusable segmented-control component. Revisit when a second segmented control exists.
- Do not synchronize row heights across particle columns; rows represent independently ranked collocations and do not share row identity.
