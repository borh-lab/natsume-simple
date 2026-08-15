# Compact Search Header and Results Toolbar

Status: Approved for implementation

## Purpose

Vertical and horizontal space both belong primarily to the collocation spreadsheet. The current interface spends separate rows on corpus selection, result identity, and bar scale, while its query-direction control repeats grammatical words without making the selected search mode obvious. The interface should retain one primary search header and at most one persistent toolbar before the results.

## Current Evidence

Rendered at 1440 pixels, the current page uses three rows below the header for corpora, result identity, and bar scale. At 390 pixels, corpus controls wrap and these controls occupy four rows. The mobile header's equal-width tracks also force `Natsume Simple` onto two lines even though the theme toggle needs only one small track. The direction selector repeats `Noun → Particle → Verb` twice, and its selected state relies mainly on a subtle background.

The current disclosure marker is a text `▶`. In the production font stack it renders as a bright orange emoji-style square, visually competing with the orange corpus slot. Disclosure state is interface chrome, not corpus identity, so this collision belongs to the same visual cleanup.

The existing controller intentionally separates draft controls from the accepted result. Editing the term or direction sets `draftDiffersFromResult`; it does not mutate the visible result. Corpus changes submit immediately. Bar scale is presentation-only state currently owned by `ParticleOverview`.

The existing test suite uses the current visible wording as selectors: `Update results` appears at eleven submit sites, `Search direction` at seven mode-control sites, and the long result summary at the shared search helper plus four focused assertions. These are migration owners, not evidence that the old wording must survive.

## Design

### Search header

The header keeps the brand at the start, the query form centered on wide screens, and the theme toggle at the end. At `lg` and above it uses equal outer tracks with a flexible center track so the form can use available width without losing viewport centering. Below `lg`, the first row uses `minmax(0, 1fr) auto`: the brand receives all space not required by the theme button and remains on one line. The query form occupies the second row at full available width.

The query form contains:

- a visibly labelled `Search by` segmented radio group with short `Noun` and `Verb` options;
- the search input, which grows to consume remaining width; and
- a compact submit button labelled `Go`, `Update`, or `Searching…`.

The arrows and repeated `Particle` labels are removed. The selected mode uses checked radio semantics, bold text, a neutral background, and a two-pixel inset ring that does not change control dimensions. Hover and keyboard focus are separately visible and remain neutral. Grammatical role colors stay reserved for annotated sentence spans. The short visible labels do not weaken accessible names: the radios remain `Noun-particle collocations` and `Verb-particle collocations` to assistive technology.

The form is width-driven rather than content-width-driven: the input has `min-width: 0`, grows into the flexible center track, and yields only to the intrinsic widths of the selector and submit button. The submit button reserves enough width for `Update`, so editing a draft does not shift the centered form. The narrow layout therefore remains one query row rather than wrapping each control onto its own row.

### Results toolbar

One compact toolbar sits immediately above the particle spreadsheet. It owns:

- corpus selection;
- bar scale;
- accepted-result identity; and
- the optional draft-status indicator.

At the `xl` breakpoint (1280 CSS pixels) these appear in one non-wrapping row with compact separators. Corpus controls and bar scale stay at the start. Result identity uses remaining space and reads, for example, `648 matches · “時間” · Noun`. When any draft control differs from that accepted result, a compact neutral `Not applied` badge appears. The header submit button simultaneously reads `Update`.

The result identity reflects only the accepted response and its submitted input. It does not change as the draft is edited. A long term may truncate visually, but the complete identity remains available as an accessible label and title.

Below `xl`, persistent horizontal scrolling and multi-row toolbar wrapping are both prohibited. Corpus and scale controls move behind an `Options` button. When the selection is filtered, its visible label becomes `Options · N/M`, where `N` is selected corpora and `M` is available corpora; the default all-selected state does not repeat a low-signal count. The anchored overlay renders the same corpus and scale control DOM used by the wide layout; responsive layout changes its positioning rather than creating another stateful implementation. The persistent toolbar row retains the Options button, compact accepted-result identity, and optional status badge. The overlay is anchored to the toolbar start, layered above results, and constrained to the viewport width.

The overlay:

- uses a real button with `aria-expanded` and `aria-controls`;
- closes on Escape, outside activation, or focus leaving the options widget;
- remains open when request-driven disabling blurs a focused checkbox to `null` or `document.body`; only a real focus destination outside the widget counts as focus departure;
- does not resize or displace the results when opened; and
- preserves the existing corpus swatches, labels, last-corpus guard, and bar-scale semantics.

The controlled panel remains in the DOM while closed and is hidden from layout and accessibility exposure. This keeps `aria-controls` resolvable and ensures the inline and overlay layouts use one control tree.

`Not applied` and `Previous result` use neutral border, background, and text colors rather than amber/orange. Their literal text carries the state, so neither collides with corpus or grammatical-role palettes. They are mutually exclusive with the existing precedence: `Not applied` when draft inputs differ from the accepted result; otherwise `Previous result` when the accepted result is stale because an equivalent request is loading or failed.

Error and empty-result messages remain full-width content below the toolbar because compressing them would obscure actionable information.

The toolbar appears once corpus metadata exists. Before any accepted result, it exposes corpus controls but omits result identity and bar scale. An accepted empty result receives the normal toolbar and identity. A failure with a previous result retains the complete toolbar and adds the existing full-width error; an initial failure with no corpus metadata renders only the error.

## Component Ownership

`+page.svelte` owns the header and the single results toolbar because it already composes search state, corpus state, accepted results, and the overview. It owns the `BarScale` value and passes it into `ParticleOverview`.

The toolbar is implemented directly at this orchestration boundary. Its small local `optionsOpen` place and outside/Escape/focus-close behavior do not justify a wrapper component whose interface would merely repeat the page's controller inputs. The options button and control panel share one wrapper, and one corpus/scale control DOM is positioned inline at `xl` or as an overlay below it.

`SearchControls.svelte` owns only draft query input, query-mode selection, autocomplete, and submission. Its public props do not grow.

`SearchSummary.svelte` becomes a compact inline representation of accepted-result identity and draft status. It does not own layout rows or configuration state.

`CorpusOptions.svelte` remains the single corpus-control implementation. The page places that same component inline or in the narrow-screen overlay; there are not separate desktop and mobile control implementations.

`ParticleOverview.svelte` receives `barScale` as a prop and no longer renders the scale selector. Its spreadsheet, scrolling, and particle-column behavior remain unchanged.

`CollocationItem.svelte` replaces the text disclosure glyph with a small neutral `currentColor` SVG chevron. Rotation still communicates the native `details` state. It never uses a corpus or grammatical-role color.

No store, generic toolbar framework, popover dependency, controller API, or network API surface is introduced. The only component-interface change is the required `barScale` prop on `ParticleOverview`.

## State and Failure Semantics

- Editing term, direction, or selected corpora changes draft state. Existing results remain visible, retain their accepted identity, and show `Not applied` whenever those values differ.
- Submission keeps the accepted identity and badge until a matching response is accepted. Loading uses the existing stale-result behavior.
- A successful response updates result identity and removes `Not applied`.
- Request failure preserves the previous accepted result and its identity while the existing full-width error remains visible.
- Corpus toggles continue to submit immediately. During the request they briefly differ from the accepted result; success reconciles them, while failure correctly leaves `Not applied` visible until the visitor retries or restores the accepted selection.
- Bar-scale changes remain local presentation changes and never trigger a request or dirty badge. Moving the value to the page intentionally makes the chosen scale persist across accepted searches for the lifetime of the page; a reload restores the default `Within particle` scale.

## Responsive Contract

- The current brand's intrinsic content occupies about 222 pixels. Replacing the 347-pixel direction control with `Search by`, `Noun`, and `Verb` gives the proposed query form a conservative width below 500 pixels. At 1024 pixels, equal outer tracks can each hold the brand while leaving at least 500 pixels for the center, which earns the `lg` header breakpoint.
- Existing corpus labels occupy about 425 pixels; scale controls, result identity, status, separators, and gaps bring the conservative inline-toolbar budget above the 992-pixel content width at 1024 but below the 1248-pixel content width at 1280. This earns the separate `xl` toolbar breakpoint.
- At `lg` (1024 CSS pixels) and above, the header query form is centered within the viewport to the existing four-pixel tolerance in both `Go` and `Update` states.
- Below `lg`, the brand remains on one line and the query form uses the full second header row without horizontal overflow or wrapping.
- At `xl` and above, the results toolbar is at most 40 pixels high with inline configuration.
- Below `xl`, corpus and scale controls are absent from persistent layout and available through the Options overlay.
- The toolbar itself never scrolls horizontally.
- Opening the overlay does not change the spreadsheet's vertical position.

## Test Contract

Browser coverage must prove:

1. Noun and Verb remain accessible radios in a group named `Search by`.
2. Their visible text is `Noun` and `Verb`, while their accessible names remain `Noun-particle collocations` and `Verb-particle collocations`.
3. The checked segment has bold text, a distinct neutral background, and a visible two-pixel inset border in light and dark themes.
4. Keyboard focus remains visible independently of selection.
5. At 390 pixels the search input remains at least 96 pixels wide; at 1024 and 1280 the form center differs from the viewport center by at most four pixels. The button has the same measured width in `Go` and `Update` states and uses the exact visible labels `Go`, `Update`, and `Searching…`.
6. Editing the draft leaves the accepted result identity unchanged and adds `Not applied`; accepting the response updates identity and removes it. `Previous result` appears only when stale without a differing draft.
7. At 1280 pixels the toolbar is no more than 40 pixels high and all persistent children share that row. Playwright's project default becomes an explicit 1440 × 900 viewport, while boundary tests pin 1023/1024 for the header and 1279/1280 for toolbar layout.
8. At 390 pixels, the toolbar does not overflow horizontally, corpus and scale controls are accessible through `Options`, and the overlay does not move the spreadsheet.
9. The overlay opens and closes by pointer, Escape, and real focus departure, reports `aria-expanded` correctly, and stays open when a corpus toggle disables its focused checkbox during the resulting request.
10. Existing query-direction, autocomplete, corpus filtering, last-corpus protection, bar-scale, dark-mode, spreadsheet, and pagination behaviors remain protected through migrated selectors and explicit viewports.
11. Bar scale persists across a successful new search, never produces a request, and resets only on page reload.
12. Grammatical-role color ownership remains pinned by `ROLE_TEXT_CLASSES`, the `sentenceSegments` unit tests, and computed colors on expanded sentence spans. The retired selector-to-sentence equality assertion is not recreated because the selector is intentionally neutral.
13. The disclosure chevron and both status badges use neutral computed colors that differ from all corpus swatches and expanded noun, particle, and verb spans.
14. Idle, initial-error, accepted-empty, and previous-result error states render the toolbar contents defined above.

## Test Migration

The browser suite is migrated deliberately rather than described as textually unchanged:

- all `Update results` submit locators become `Update`;
- every `Search direction` group locator becomes `Search by`, while radio locators retain the descriptive accessible names;
- the shared search helper and focused result-summary assertions adopt `N matches · “term” · Noun|Verb`;
- `Controls changed — update results to apply them.` becomes `Not applied`;
- mobile bar-scale assertions open Options before locating the control;
- the old 1024-pixel second-row header assertion becomes explicit 1023/1024 boundary coverage;
- the three `[data-role]` selector palette assertions are retired or relocated to expanded sentence spans; and
- `playwright.config.ts` sets the ordinary test viewport to 1440 × 900 rather than landing accidentally on the `xl` boundary; responsive tests continue to set their own explicit sizes.

## Deliberate Omissions

- No generic responsive-toolbar abstraction: there is one toolbar and no second consumer.
- No new popover package: the interaction is small and uses existing browser primitives and Svelte state.
- No URL persistence for configuration: no present consumer requires shareable view state.
- No shortened corpus labels on desktop: the wide row has room, while narrow layouts use the overlay.

## Decision Log

### 2026-08-15: Keep configuration outside the primary header

Putting corpus and scale controls into the centered header would make query centering depend on corpus-label width and overload the primary action area. One dedicated results toolbar uses the available width without weakening hierarchy.

Revisit only if a future global navigation system replaces the current header.

### 2026-08-15: Use an overlay rather than horizontal scrolling or wrapping on narrow screens

Horizontal toolbar scrolling hides configuration and creates a second horizontal interaction beside the spreadsheet. Wrapping consumes the vertical space this change exists to recover. An overlay preserves one persistent row.

Revisit if configuration becomes a primary per-query workflow that must remain continuously visible.

### 2026-08-15: Keep accepted-result identity visible

The result identity is the necessary boundary between editable draft controls and the data currently displayed. It is compressed, not removed. The `Not applied` badge describes that relationship without a full sentence or additional row.

Revisit if results update continuously as controls change and there is no longer a draft/accepted distinction.

### 2026-08-15: Persist bar scale across searches

Bar scale describes how the visitor wants to compare bars, not the identity of a search response. Resetting it whenever a new response remounts the overview makes repeated comparisons needlessly revert. Page-local ownership keeps the preference without adding storage, URL state, or a store.

Revisit if different search modes acquire incompatible scale choices.

### 2026-08-15: Keep disclosure chrome neutral

The font-rendered `▶` currently appears as a bright orange square and collides with the orange corpus slot. A small `currentColor` SVG is deterministic across platforms and keeps expand/collapse state outside both semantic palettes.

Revisit only if the interface adopts a shared icon set with the same neutral-state contract.

## Lifecycle

This document is the retained decision record. After implementation, its status changes to `Implemented`. Any execution plan created from it is retired after the checks pass.
