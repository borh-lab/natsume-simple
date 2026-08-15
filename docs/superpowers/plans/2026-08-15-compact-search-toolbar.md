# Compact Search Toolbar Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the vertically wasteful search/options layout with a compact, accessible query header and one responsive results toolbar while preserving the accepted-result boundary.

**Architecture:** `+page.svelte` remains the orchestration boundary and owns the single toolbar, overlay state, and page-local bar scale. Existing leaf components keep their present responsibilities: `SearchControls` edits/submits the draft, `SearchSummary` renders accepted-result identity, `CorpusOptions` renders one corpus-control tree, and `ParticleOverview` renders only results. Browser tests exercise real layout, focus, and request transitions.

**Tech Stack:** Svelte 5 runes, SvelteKit, Tailwind CSS 4, TypeScript 6, Playwright, Vitest, Nix flakes.

## Global Constraints

- Keep the brand, query form, and theme switch in the header; configuration remains in one results toolbar.
- At `lg` (1024px) the query form is viewport-centered within four pixels; below `lg` it occupies one full second row.
- At `xl` (1280px) configuration is inline in a toolbar no taller than 40px; below `xl` it is available through one non-scrolling overlay.
- The checked query mode uses bold text, a neutral background, and a visible two-pixel inset border; role and corpus colors are not used for chrome.
- Visible query-mode labels are `Noun` and `Verb`; accessible names remain `Noun-particle collocations` and `Verb-particle collocations` in a group named `Search by`.
- Submit labels are exactly `Go`, `Update`, and `Searching…`, with a stable measured button width.
- Accepted-result identity is compact and immutable while draft controls change; draft differences show `Not applied`.
- Corpus toggles still submit immediately; request-induced focus loss must not close the narrow options overlay.
- Bar scale remains local presentation state, persists across accepted searches, and resets on reload.
- Do not add a store, controller method, popover dependency, generic toolbar component, or network API.
- Preserve untracked owner files `AGENDA.md` and `container.nix`; never stage them.

---

### Task 1: Compact query-mode and submit controls

**Files:**
- Modify: `natsume-frontend/src/lib/components/SearchControls.svelte`
- Modify: `natsume-frontend/tests/test.ts`

**Interfaces:**
- Consumes: existing bindable `term` and `pos`, `loading`, `dirty`, `onsubmit`, and `findSuggestions` props.
- Produces: the unchanged component interface with a `Search by` radio group and stable-width `Go | Update | Searching…` submit control.

- [ ] **Step 1: Write failing browser assertions for the compact selector**

Add a focused Playwright test that obtains `getByRole('group', { name: 'Search by' })`, asserts the two descriptive radio names, checks `Noun`, and verifies its visible span has `fontWeight >= 600`, a background different from the unchecked span, and a two-pixel inset ring/box-shadow in both themes. Assert the control has no `[data-role]` descendants.

- [ ] **Step 2: Write failing assertions for compact button labels and stable width**

At a 1024px viewport, record the submit button width in the accepted `Go` state, edit the term, assert visible text `Update`, and require the width delta to be at most one pixel. Submit and assert `Searching…` while the intercepted collocations request is pending.

- [ ] **Step 3: Run the focused tests and verify RED**

Run:

```bash
nix develop .#frontend --command bash -lc 'cd natsume-frontend && npm run test:integration -- --grep "compact search mode|stable submit width"'
```

Expected: failures because the group is still named `Search direction`, the long arrow labels and role colors remain, and the draft label is `Update results`.

- [ ] **Step 4: Implement the minimal selector and button changes**

In `SearchControls.svelte`:

- remove the `ROLE_TEXT_CLASSES` import and all `[data-role]` spans;
- use `<legend class="mr-1 text-sm font-medium">Search by</legend>` and keep the fieldset's accessible group semantics;
- keep each radio as a transparent full-size overlay so Playwright `.check()` remains actionable;
- render only `Noun` and `Verb` in their presentation spans;
- use neutral checked styles with `peer-checked:font-bold`, a light/dark neutral fill, and `peer-checked:shadow-[inset_0_0_0_2px_currentColor]` so selection does not change dimensions;
- keep focus-visible styling on the segment without an overflow-clipping ancestor;
- change the form to `flex w-full min-w-0 flex-nowrap items-center gap-2`, give the search wrapper/input `min-w-0 flex-1`, and reserve the submit width with `min-w-[5.5rem]`; and
- render `{loading ? 'Searching…' : dirty ? 'Update' : 'Go'}`.

- [ ] **Step 5: Migrate existing query-control selectors**

Change all eleven `Update results` button locators to `Update`, all seven `Search direction` group locators to `Search by`, and keep descriptive radio locators unchanged. Replace selector-role palette assertions with assertions against expanded sentence spans; do not recreate role colors in query chrome.

- [ ] **Step 6: Run the focused and existing interaction tests GREEN**

Run:

```bash
nix develop .#frontend --command bash -lc 'cd natsume-frontend && npm run test:integration -- --grep "compact search mode|stable submit width|query direction|suggestions"'
```

Expected: all selected tests pass, including clickable radios and autocomplete focus transfer.

- [ ] **Step 7: Commit Task 1**

```bash
git add natsume-frontend/src/lib/components/SearchControls.svelte natsume-frontend/tests/test.ts
git commit -m "feat: compact search mode controls"
```

### Task 2: Center the responsive search header

**Files:**
- Modify: `natsume-frontend/src/routes/+page.svelte`
- Modify: `natsume-frontend/playwright.config.ts`
- Modify: `natsume-frontend/tests/test.ts`

**Interfaces:**
- Consumes: the unchanged `SearchControls` component from Task 1.
- Produces: `header-controls` centered at and above 1024px, with a non-wrapping brand and full-width query row below 1024px.

- [ ] **Step 1: Add failing 1023/1024 header geometry tests**

Set explicit viewports of 1023px and 1024px. At 1023px assert brand and theme switch share row one, query controls occupy row two, the brand heading stays on one line, the document has no horizontal overflow, and the input is at least 96px wide. At 1024px assert the query form's center differs from the viewport center by no more than four pixels in both `Go` and `Update` states.

- [ ] **Step 2: Pin the normal Playwright viewport**

Set `viewport: { width: 1440, height: 900 }` under `use` in `playwright.config.ts`, so ordinary tests do not land on the `xl` boundary.

- [ ] **Step 3: Run the header tests and verify RED**

Run:

```bash
nix develop .#frontend --command bash -lc 'cd natsume-frontend && npm run test:integration -- --grep "responsive search header"'
```

Expected: the 1024px form remains on row two because the current breakpoint is `xl`, and narrow brand/header geometry fails.

- [ ] **Step 4: Implement the header grid**

Use the complete page-header classes:

```svelte
<div class="grid w-full grid-cols-[minmax(0,1fr)_auto] items-center gap-3 p-4 lg:grid-cols-[1fr_minmax(0,auto)_1fr]">
```

Keep the brand in column one with `min-w-0 flex-nowrap`; make the heading `whitespace-nowrap`. Place the query controls across both columns on row two below `lg`, then at `lg` use column two/row one with `w-full min-w-0 justify-self-center`. Keep the theme switch in column two below `lg` and column three at `lg`.

- [ ] **Step 5: Run header tests GREEN and check Svelte types**

Run:

```bash
nix develop .#frontend --command bash -lc 'cd natsume-frontend && npm run test:integration -- --grep "responsive search header" && npm run check'
```

Expected: geometry tests and `svelte-check` pass.

- [ ] **Step 6: Commit Task 2**

```bash
git add natsume-frontend/src/routes/+page.svelte natsume-frontend/playwright.config.ts natsume-frontend/tests/test.ts
git commit -m "feat: center responsive search header"
```

### Task 3: Build the single responsive results toolbar

**Files:**
- Modify: `natsume-frontend/src/routes/+page.svelte`
- Modify: `natsume-frontend/src/lib/components/SearchSummary.svelte`
- Modify: `natsume-frontend/src/lib/components/ParticleOverview.svelte`
- Modify: `natsume-frontend/tests/test.ts`

**Interfaces:**
- Consumes: `CorpusOptions`, controller corpus/result state, and `BarScale` from `$lib/presentation/search`.
- Produces: page-local `barScale: BarScale`, `ParticleOverview` prop `barScale: BarScale`, and a single toolbar whose same options panel is inline at `xl` and overlaid below it.

- [ ] **Step 1: Add failing accepted-identity and status tests**

Update the shared search helper to expect `N matches · “term” · Noun|Verb`. Add a test that edits term and direction, verifies the accepted identity remains unchanged, verifies a neutral `Not applied` badge, submits, then verifies the identity changes and the badge disappears. Cover `Previous result` only for an equivalent loading/error state.

- [ ] **Step 2: Add failing toolbar state and scale-ownership tests**

Assert that before the first accepted response the toolbar exposes corpus options but has no result identity or Bar scale. After a result, change scale to `Across particles`, submit a new term, and assert the value persists without a scale-triggered request. Reload and assert `Within particle`. Cover accepted-empty, previous-result error, and initial corpus-metadata error contents.

- [ ] **Step 3: Add failing responsive toolbar and overlay tests**

At 1280px assert the toolbar is at most 40px high, all persistent children share its vertical band, and it does not overflow. At 1279px and 390px assert corpus/scale controls are hidden from persistent layout, `Options` is visible, opening it leaves the spreadsheet top unchanged, the panel is controlled by `aria-controls`, Escape/outside/real focus departure close it, and the toolbar never horizontally scrolls. Toggle one corpus while the overlay is open and require it to remain open across the resulting request. Assert `Options` becomes `Options · 2/3` only when filtered.

- [ ] **Step 4: Run focused toolbar tests and verify RED**

Run:

```bash
nix develop .#frontend --command bash -lc 'cd natsume-frontend && npm run test:integration -- --grep "accepted result identity|results toolbar|options overlay|bar scale persists"'
```

Expected: failures because configuration and summary occupy separate rows, scale state is owned by `ParticleOverview`, and no overlay exists.

- [ ] **Step 5: Make SearchSummary compact and neutral**

Render one inline `aria-live="polite"` container. Use accepted response/input only for `${count.toLocaleString()} matches · “${term}” · ${Noun|Verb}` and expose the full string via `title`/accessible label while allowing visual truncation. Render exactly one neutral pill: `Not applied` when `draftDiffers`, otherwise `Previous result` when `stale`. Remove the amber colors and block spacing.

- [ ] **Step 6: Move bar-scale state and toolbar composition to the page**

In `+page.svelte`, import `type BarScale`, add `let barScale = $state<BarScale>('particle')`, `let optionsOpen = $state(false)`, and retain the options panel in the DOM. Compose directly:

- a relatively positioned toolbar with `data-testid="results-toolbar"`;
- one wrapper/button/panel pair with `aria-expanded` and `aria-controls="results-options"`;
- the same panel containing `CorpusOptions` and the Bar scale `<select>`;
- inline panel positioning at `xl`, absolute overlay positioning below `xl`, and hidden/inert semantics while closed;
- `SearchSummary` as the flexible, truncating identity; and
- neutral status/spacing and compact separators.

Close on Escape, outside pointer activation, and focusout only when `relatedTarget` is a real node outside the wrapper. Ignore `null`/`document.body` focus destinations so request-driven checkbox disabling cannot close the panel. Keep the panel open through a corpus toggle.

Only show Bar scale once `controller.result` exists. Show the toolbar once `controller.corpora.length > 0`; keep actionable errors below it. Pass `{barScale}` to `ParticleOverview`.

- [ ] **Step 7: Make ParticleOverview a pure results view**

Add `barScale: BarScale` to its props, remove local `$state`, remove the Bar scale control and outer `space-y-2` wrapper, and retain the keyboard-focusable spreadsheet region unchanged.

- [ ] **Step 8: Migrate summary, mobile scale, and geometry selectors**

Replace the long summary strings and old dirty sentence with the new identity/badge. Mobile tests must open Options before locating Bar scale. Replace the old 1024 second-row assertion with Task 2's boundary checks. Give affected tests explicit viewport sizes.

- [ ] **Step 9: Run focused toolbar tests GREEN**

Run:

```bash
nix develop .#frontend --command bash -lc 'cd natsume-frontend && npm run test:integration -- --grep "accepted result identity|results toolbar|options overlay|bar scale persists"'
```

Expected: all focused tests pass, including request-induced blur and non-displacing overlay behavior.

- [ ] **Step 10: Commit Task 3**

```bash
git add natsume-frontend/src/routes/+page.svelte natsume-frontend/src/lib/components/SearchSummary.svelte natsume-frontend/src/lib/components/ParticleOverview.svelte natsume-frontend/tests/test.ts
git commit -m "feat: add compact results toolbar"
```

### Task 4: Neutralize disclosure chrome and close visual gaps

**Files:**
- Modify: `natsume-frontend/src/lib/components/CollocationItem.svelte`
- Modify: `natsume-frontend/tests/test.ts`

**Interfaces:**
- Consumes: native `details[open]` state and existing semantic color tokens exposed by corpus swatches and sentence spans.
- Produces: a deterministic `currentColor` SVG chevron whose collapsed/open state remains visible and whose color is outside semantic palettes.

- [ ] **Step 1: Add a failing visual-semantic browser assertion**

Expand an example and collect computed corpus swatch colors plus noun/particle/verb sentence-span colors. Assert the disclosure icon and `Not applied`/`Previous result` badge colors are neutral and differ from every collected semantic color. Reach summary focus via keyboard Tab and assert its focus outline is visible. Assert open and closed summary backgrounds differ in both themes.

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
nix develop .#frontend --command bash -lc 'cd natsume-frontend && npm run test:integration -- --grep "neutral disclosure and status chrome"'
```

Expected: the current text glyph resolves to emoji/orange rendering and status text remains amber.

- [ ] **Step 3: Replace the glyph with a neutral SVG**

Keep the existing wrapper `<div>` and remove no structural parent. Replace `▶` with a small inline SVG using `viewBox="0 0 10 10"`, `fill="currentColor"`, and a triangle path. Preserve the `disclosure-chevron` class and native `details[open]` rotation. Do not use a corpus or role class.

- [ ] **Step 4: Run the focused visual test GREEN**

Run:

```bash
nix develop .#frontend --command bash -lc 'cd natsume-frontend && npm run test:integration -- --grep "neutral disclosure and status chrome"'
```

Expected: neutral palette, keyboard focus, and open-state assertions pass.

- [ ] **Step 5: Commit Task 4**

```bash
git add natsume-frontend/src/lib/components/CollocationItem.svelte natsume-frontend/tests/test.ts
git commit -m "fix: neutralize disclosure chrome"
```

### Task 5: Verify, document, and retire the execution plan

**Files:**
- Modify: `docs/superpowers/specs/2026-08-15-compact-search-toolbar-design.md`
- Delete after verification: `docs/superpowers/plans/2026-08-15-compact-search-toolbar.md`

**Interfaces:**
- Consumes: Tasks 1–4 and their browser coverage.
- Produces: an implemented retained design record and a fully gated frontend.

- [ ] **Step 1: Run frontend format, lint, type, unit, build, and browser gates**

Run:

```bash
nix develop .#frontend --command bash -lc 'cd natsume-frontend && npm run format && npm run lint && npm run check && npm run test:unit -- --run && npm run build && npm run test:integration'
```

Expected: all commands pass with no warnings or failures.

- [ ] **Step 2: Run hermetic Nix checks**

Run:

```bash
nix build .#checks.x86_64-linux.frontend .#checks.x86_64-linux.playwright .#checks.x86_64-linux.package-frontend .#checks.x86_64-linux.server-smoke
```

Expected: all four derivations build successfully.

- [ ] **Step 3: Perform the final sufficiency review**

Confirm the diff adds no component, store, controller method, dependency, URL state, duplicated corpus controls, or network API. Confirm `AGENDA.md` and `container.nix` remain untracked and unstaged.

- [ ] **Step 4: Mark the retained design record implemented**

Change the design status from `Approved for implementation` to `Implemented`. Add a short evidence line naming the passing Playwright and Nix gates; do not copy the implementation plan into the spec.

- [ ] **Step 5: Retire this execution plan**

Delete `docs/superpowers/plans/2026-08-15-compact-search-toolbar.md` after every gate passes. The retained design record already owns the rationale, breakpoints, and revisit triggers.

- [ ] **Step 6: Commit verification documentation**

```bash
git add docs/superpowers/specs/2026-08-15-compact-search-toolbar-design.md
git add -u docs/superpowers/plans/2026-08-15-compact-search-toolbar.md
git commit -m "docs: record compact toolbar implementation"
```

- [ ] **Step 7: Review the final branch before merge**

Run:

```bash
git status --short
git diff --check main...HEAD
git diff --stat main...HEAD
git log --oneline main..HEAD
```

Expected: only `AGENDA.md` and `container.nix` are untracked, the diff is whitespace-clean, and commits are limited to this feature.
