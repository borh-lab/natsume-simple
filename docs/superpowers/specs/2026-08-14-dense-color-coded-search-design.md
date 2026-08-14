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
have five presentation uses (item bars, particle-mass bars, corpus-filter swatches, example
borders, and example source labels).

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

| Slot | SVG/swatch/border | Light title | Dark title |
| --- | --- | --- | --- |
| 0 | `#7c3aed` | violet-700 | violet-300 |
| 1 | `#ea580c` | orange-700 | orange-300 |
| 2 | `#0891b2` | cyan-700 | cyan-300 |

The existing corpus-order-to-slot mapping remains authoritative. Bars, example borders,
filter swatches, and source labels all consume the same slot. SVG fills, swatches, and
example left borders use the slot's one hex value; the border does not shadow that value
with an independently maintained Tailwind color token. Light and dark title classes remain
whole static literals so Tailwind discovers them without generated class names.

When more than one corpus is selected, an example starts with the compact visible text
`<corpus label> · <source title>:`. The label and title share the corpus title color, and
the row carries the matching left border. The explicit corpus label means identity is not
conveyed by color alone and is also present in the accessible text without a separate ARIA
override or badge component. With one selected corpus, only `<source title>:` is rendered:
the corpus is already unambiguous, and repeating a long corpus label in every 320-pixel row
would work against the density requirement.

The TypeScript accessor does not use modulo arithmetic. Slots 0–2 return the three corpus
records; an out-of-range slot returns one neutral gray fallback record rather than aliasing
an existing corpus color. The backend rejects artifacts outside the one-to-three-corpus
contract at build and startup, while the local fallback keeps malformed fixtures or future
contract drift distinguishable in the UI. Unit coverage pins this behavior with a fourth
corpus and proves that it does not reuse any of the three valid slot colors.
All out-of-range slots intentionally share the same neutral fallback; they are a visible
contract-failure state, not an expanded identity palette.

## Header and Search Type

At the `xl` breakpoint (1280 CSS pixels) and wider, the header is a three-column grid with
equal flexible side columns:

1. brand aligned left;
2. search/type controls aligned to the viewport center;
3. theme toggle aligned right.

Below `xl`, brand and theme occupy the first row and the controls occupy a centered,
full-width second row. The second-row layout is intentional at tablet and ordinary laptop
widths: the complete direction control, search input, and submit action do not fit safely
between two equal side tracks.

The native search-direction select is replaced by one compact radio group with two choices:

- `Noun → Particle → Verb` for noun search;
- `Noun ← Particle ← Verb` for verb search.

The three words use the shared grammatical-role colors. The choices live in a
`<fieldset>` whose visually hidden legend is `Search direction`. Each radio has a descriptive
accessible name (`Noun-particle collocations` or `Verb-particle collocations`) rather than
requiring assistive technology to interpret arrow glyphs. Native radio inputs retain keyboard
and form semantics; the labels provide the segmented visual surface. Each transparent radio
is a full-size absolute overlay inside its label, so the visible segment remains a real pointer
and Playwright hit target. Focus is drawn on an unclipped child surface.

Selected, hover, focus, and submit-button chrome within the search widget uses neutral
gray/slate colors, not grammatical blue/red/green. Corpus-checkbox accents, the results-region
focus ring, compact-row hover/open/focus chrome, and example paging actions are neutral for
the same reason. This is a scoped rule for surfaces that display semantic role or corpus
colors; explicitly labelled loading and error messages may retain their established status
colors. Compact rows use an inset ring or zero-offset outline so focus does not paint across
an adjacent row's separator.

## Dense Result Rows

Each collapsed collocation summary is at most 32 pixels high in the fixture viewport:

- one-pixel horizontal separator instead of a rounded card border;
- `text-sm leading-5` with `px-1 py-1` or less;
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
- `CorpusOptions.svelte` consumes the same slot accessor for corpus-filter swatches and uses
  neutral checkbox chrome.
- `SearchControls.svelte` renders the radio-based type selector using the shared role classes.
- `+page.svelte` owns the responsive centered-header grid.
- `ParticleOverview.svelte` retains the bar-scale control and scrolling region while changing
  that region's focus chrome to neutral.
- `ParticleColumn.svelte` consumes the shared slot accessor for the particle-mass bar and
  passes the existing corpora and slot map through the collocation component chain; no
  context or second corpus-identity map is introduced.
- `CollocationItem.svelte` renders the compact summary surface.
- `SentenceExamples.svelte` resolves each example's own `corpusId` against those values and
  renders the visible corpus label, colored title/border, and compact example row.

No new store, context, public API field, runtime palette registry, or generic styling component is introduced.

### Architecture disposition

- **Type:** Decomplect, followed by a small implementation refactor. Styling values move
  out of sentence segmentation and bar mathematics; neither behavior gains a new protocol.
- **Evidence:** grammatical-role classes currently live privately in `sentence.ts`, while
  corpus colors are already consumed by item bars, particle-mass bars, and filter swatches
  from `presentation/search.ts`. The requested selector, example borders, and source labels
  add three consumers. This would be disconfirmed if the requested cross-component consumers
  were removed, in which case the corresponding constants should remain local.
- **Hazards:** the module contains immutable values only. It owns no state, time, identity,
  trust boundary, configuration, or runtime selection. Existing role colors and corpus slot
  ordering remain behavior-preservation constraints.
- **Characterization:** unit coverage pins the role map, the three corpus slots, and palette
  disjointness; browser coverage proves the maps reach the selector, highlights, bars, and
  example labels rather than merely testing exported constants.

## Test Contract

Unit coverage must pin:

- the unchanged noun/particle/verb class map;
- the three corpus hex colors and their disjointness from role colors;
- the neutral, non-aliasing result for an out-of-range fourth corpus slot.

Browser coverage must fail if any of these observable properties regress:

- the desktop search-control center differs materially from the viewport center;
- mobile controls no longer occupy a centered second row;
- the search-direction group is not named `Search direction`, or either radio loses its
  descriptive accessible name;
- noun, particle, and verb labels in the selector differ from their corresponding sentence-highlight colors;
- source titles from different corpora share one computed color;
- an example title's corpus slot disagrees with its left border and collocation-bar slot;
- a corpus-filter swatch disagrees with that corpus's item-bar or particle-mass-bar color;
- corpus colors reuse the grammatical blue/red/green role colors;
- a collapsed result summary exceeds 34 pixels in the fixture viewport;
- adjacent summaries regain vertical gaps or rounded card treatment;
- selector or compact-row selection/open/focus chrome reuses a grammatical-role color;
- the computed open-row background equals the closed-row background in either theme;
- a keyboard-focused summary has `outline-style: none` in either theme;
- visible and accessible source text omits the corpus label when multiple corpora are selected.

For the desktop centering assertion, the horizontal midpoint of `header-controls` must be
within four CSS pixels of the viewport midpoint at the fixture desktop viewport. The row
height assertion permits a two-pixel browser-rendering tolerance and therefore fails above
34 pixels even though the design target is at most 32 pixels.

The existing Playwright tests that call `selectOption` on `Search direction` change
mechanically to select the corresponding radio. Their protected behavior remains: changing
direction marks the controls dirty without fetching, and the submitted `pos` reaches
`/api/collocations` when the user updates the results.

Existing frontend unit, Svelte diagnostic, build, Playwright, and server-smoke gates remain required.

## Deliberate Omissions

- Do not make palettes operator-configurable. Revisit only if a present artifact requires more than three corpora or branding requires runtime theming.
- Do not build a reusable segmented-control component. Revisit when a second segmented control exists.
- Do not synchronize row heights across particle columns; rows represent independently ranked collocations and do not share row identity.
