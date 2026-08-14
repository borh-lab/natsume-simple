# Dense, Color-Coded Search Interface Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the search header genuinely centered and accessible, give grammatical roles and corpora distinct stable colors, identify each example's corpus, and compress collocation results into a spreadsheet-density layout.

**Architecture:** Add one immutable `presentation/colors.ts` owner for fixed role and corpus presentation values. Existing search math, sentence segmentation, and Svelte components consume those values directly; no store, context, API field, runtime theme registry, or generic component is added. Unit tests pin palette invariants, while Playwright proves the values reach controls, bars, source labels, and dense rows.

**Tech Stack:** Svelte 5, TypeScript 6, Tailwind CSS 4, Vitest 4, Playwright 1.61, Nix flakes.

## Global Constraints

- Preserve ranking, bar-scale modes, corpus contributions, raw-frequency tooltips, pagination, example loading, highlighting spans, keyboard operation, autocomplete timing, and dark mode.
- Keep native form submission, `SearchController`, and public API contracts unchanged.
- Keep grammatical roles at blue-600/400, red-600/400, and green-600/400.
- Use corpus colors `#7c3aed`, `#ea580c`, and `#0891b2`; invalid later slots use neutral gray and never alias a valid slot.
- Show corpus identity as text whenever more than one corpus is selected.
- Use neutral search/data-display chrome; labelled loading and error messages retain established status colors.
- Keep collapsed summaries at or below 34 measured CSS pixels, targeting at most 32.
- Add no dependency, runtime configuration, public surface, generic segmented control, or row-height synchronization.
- Preserve the owner's untracked `AGENDA.md` and `container.nix`.

---

### Task 1: Own semantic colors in one presentation module

**Files:**
- Create: `natsume-frontend/src/lib/presentation/colors.ts`
- Create: `natsume-frontend/src/lib/presentation/colors.test.ts`
- Modify: `natsume-frontend/src/lib/sentence.ts:10-16`
- Modify: `natsume-frontend/src/lib/presentation/search.ts:1-6`
- Modify: `natsume-frontend/src/lib/components/CorpusOptions.svelte:1-35`
- Modify: `natsume-frontend/src/lib/components/ParticleColumn.svelte:1-125`
- Modify: `natsume-frontend/src/lib/components/CollocationItem.svelte:1-78`

**Interfaces:**
- Produces: `ROLE_TEXT_CLASSES: Readonly<Record<'noun' | 'particle' | 'verb', string>>`.
- Produces: `corpusStyleForSlot(slot: number): { color: string; titleClass: string }`.
- Preserves: `corpusColorMap(corpora): Record<string, number>`.
- Removes: `CORPUS_COLORS` and modulo-based palette indexing.

- [ ] **Step 1: Write the failing palette contract**

Create `colors.test.ts`:

```ts
import { describe, expect, it } from 'vitest';
import { CORPUS_STYLES, ROLE_TEXT_CLASSES, corpusStyleForSlot } from './colors';

describe('semantic presentation colors', () => {
	it('pins grammatical roles and keeps corpus slots disjoint', () => {
		expect(ROLE_TEXT_CLASSES).toEqual({
			noun: 'font-bold text-blue-600 dark:text-blue-400',
			particle: 'font-bold text-red-600 dark:text-red-400',
			verb: 'font-bold text-green-600 dark:text-green-400'
		});
		expect(CORPUS_STYLES).toEqual([
			{ color: '#7c3aed', titleClass: 'text-violet-700 dark:text-violet-300' },
			{ color: '#ea580c', titleClass: 'text-orange-700 dark:text-orange-300' },
			{ color: '#0891b2', titleClass: 'text-cyan-700 dark:text-cyan-300' }
		]);
		const roleHex = new Set(['#2563eb', '#dc2626', '#16a34a']);
		expect(CORPUS_STYLES.every(({ color }) => !roleHex.has(color))).toBe(true);
	});

	it('uses a neutral fourth slot instead of aliasing a valid corpus', () => {
		const fallback = corpusStyleForSlot(3);
		expect(fallback).toEqual({
			color: '#6b7280',
			titleClass: 'text-gray-700 dark:text-gray-300'
		});
		expect(CORPUS_STYLES.map(({ color }) => color)).not.toContain(fallback.color);
	});
});
```

- [ ] **Step 2: Run the focused unit test and verify RED**

```bash
nix develop .#frontend --command npm --prefix natsume-frontend run test:unit -- --run src/lib/presentation/colors.test.ts
```

Expected: FAIL because `presentation/colors.ts` does not exist.

- [ ] **Step 3: Implement the fixed palette owner**

Create `colors.ts`:

```ts
export const ROLE_TEXT_CLASSES = {
	noun: 'font-bold text-blue-600 dark:text-blue-400',
	particle: 'font-bold text-red-600 dark:text-red-400',
	verb: 'font-bold text-green-600 dark:text-green-400'
} as const;

export const CORPUS_STYLES = [
	{ color: '#7c3aed', titleClass: 'text-violet-700 dark:text-violet-300' },
	{ color: '#ea580c', titleClass: 'text-orange-700 dark:text-orange-300' },
	{ color: '#0891b2', titleClass: 'text-cyan-700 dark:text-cyan-300' }
] as const;

const FALLBACK_CORPUS_STYLE = {
	color: '#6b7280',
	titleClass: 'text-gray-700 dark:text-gray-300'
} as const;

export function corpusStyleForSlot(slot: number) {
	return CORPUS_STYLES[slot] ?? FALLBACK_CORPUS_STYLE;
}
```

In `sentence.ts`, import `ROLE_TEXT_CLASSES`, delete private `highlightClasses`, and use:

```ts
className: ROLE_TEXT_CLASSES[span.type]
```

Delete `CORPUS_COLORS` from `presentation/search.ts`. In `CorpusOptions.svelte`, `ParticleColumn.svelte`, and `CollocationItem.svelte`, import `corpusStyleForSlot` and replace each modulo lookup with:

```ts
corpusStyleForSlot(colorSlots[corpusId]).color
```

Add `data-testid="corpus-swatch" data-corpus-id={corpus.id}` to the filter swatch, and `data-corpus-id={segment.corpusId}` to item-bar `<rect>` and particle-mass `<span>`.

- [ ] **Step 4: Run focused presentation tests and verify GREEN**

```bash
nix develop .#frontend --command npm --prefix natsume-frontend run test:unit -- --run src/lib/presentation/colors.test.ts src/lib/presentation/search.test.ts src/lib/sentence.test.ts
nix develop .#frontend --command npm --prefix natsume-frontend run check
```

Expected: focused tests PASS; Svelte reports zero errors and warnings.

- [ ] **Step 5: Commit**

```bash
git add natsume-frontend/src/lib/presentation/colors.ts natsume-frontend/src/lib/presentation/colors.test.ts natsume-frontend/src/lib/sentence.ts natsume-frontend/src/lib/presentation/search.ts natsume-frontend/src/lib/components/CorpusOptions.svelte natsume-frontend/src/lib/components/ParticleColumn.svelte natsume-frontend/src/lib/components/CollocationItem.svelte
git commit -m "refactor: centralize semantic interface colors"
```

---

### Task 2: Center an accessible search-direction control

**Files:**
- Modify: `natsume-frontend/src/lib/components/SearchControls.svelte:1-159`
- Modify: `natsume-frontend/src/routes/+page.svelte:17-49`
- Modify: `natsume-frontend/tests/test.ts:385-475,555-566`

**Interfaces:**
- Consumes: `ROLE_TEXT_CLASSES`.
- Preserves: bound `term` and `pos`, form submission, autocomplete focus containment, and `onsubmit` timing.
- Produces: native radios named `Noun-particle collocations` and `Verb-particle collocations` in a fieldset named `Search direction`.

- [ ] **Step 1: Add the failing header/accessibility browser contract**

Add to `tests/test.ts`:

```ts
test('centers an accessible search-direction control in the responsive header', async ({ page }) => {
	await page.setViewportSize({ width: 1280, height: 844 });
	await page.goto('/');
	const controls = page.getByTestId('header-controls');
	const group = page.getByRole('group', { name: 'Search direction' });
	await expect(group.getByRole('radio', { name: 'Noun-particle collocations' })).toBeChecked();
	await expect(group.getByRole('radio', { name: 'Verb-particle collocations' })).toBeVisible();
	const desktopBox = await controls.boundingBox();
	expect(desktopBox).not.toBeNull();
	expect(Math.abs((desktopBox?.x ?? 0) + (desktopBox?.width ?? 0) / 2 - 640)).toBeLessThanOrEqual(4);
	const roleColors = await group.locator('[data-role]').evaluateAll((elements) =>
		elements.map((element) => getComputedStyle(element).color)
	);
	const selectedSurface = group
		.getByRole('radio', { name: 'Noun-particle collocations' })
		.locator('xpath=following-sibling::span');
	const selectedBackground = await selectedSurface.evaluate(
		(element) => getComputedStyle(element).backgroundColor
	);
	expect(roleColors).not.toContain(selectedBackground);
	await group.getByRole('radio', { name: 'Noun-particle collocations' }).focus();
	expect(await selectedSurface.evaluate((element) => getComputedStyle(element).outlineStyle)).not.toBe('none');
	expect(roleColors).not.toContain(
		await page.getByRole('button', { name: 'Go' }).evaluate(
			(element) => getComputedStyle(element).backgroundColor
		)
	);

	await page.setViewportSize({ width: 390, height: 844 });
	const brandBox = await page.getByTestId('brand').boundingBox();
	const mobileBox = await controls.boundingBox();
	expect(brandBox).not.toBeNull();
	expect(mobileBox).not.toBeNull();
	expect(mobileBox?.y ?? 0).toBeGreaterThanOrEqual((brandBox?.y ?? 0) + (brandBox?.height ?? 0));
	expect(Math.abs((mobileBox?.x ?? 0) + (mobileBox?.width ?? 0) / 2 - 195)).toBeLessThanOrEqual(4);
});
```

- [ ] **Step 2: Run it and verify RED**

```bash
nix develop .#frontend --command bash -lc 'cd natsume-frontend && NATSUME_FIXTURE_COMMAND="cd .. && nix develop .#test --command python -m tests.fixture_server" npm run test:integration -- --grep "centers an accessible"'
```

Expected: FAIL because the current select is not a radio group and controls are right-aligned.

- [ ] **Step 3: Replace only the direction select**

Import `ROLE_TEXT_CLASSES`; retain the existing script/effects. Replace the select with a `<fieldset>` and two labels. Each label contains a `class="peer sr-only"` native radio with `name="search-position"`, `bind:group={pos}`, and one of these values/names:

```svelte
<input type="radio" value="noun" aria-label="Noun-particle collocations" />
<input type="radio" value="verb" aria-label="Verb-particle collocations" />
```

Use this complete visual content for the noun label; reverse only the arrows for verb:

```svelte
<span class="flex h-full items-center gap-1 px-2 text-sm peer-checked:bg-gray-200 peer-focus-visible:outline peer-focus-visible:outline-2 peer-focus-visible:outline-offset-0 peer-focus-visible:outline-gray-900 dark:peer-checked:bg-gray-700 dark:peer-focus-visible:outline-gray-100">
	<span class={ROLE_TEXT_CLASSES.noun} data-role="noun">Noun</span>
	<span aria-hidden="true">→</span>
	<span class={ROLE_TEXT_CLASSES.particle} data-role="particle">Particle</span>
	<span aria-hidden="true">→</span>
	<span class={ROLE_TEXT_CLASSES.verb} data-role="verb">Verb</span>
</span>
```

The fieldset legend is `<legend class="sr-only">Search direction</legend>`. Give the fieldset `class="flex h-10 overflow-hidden rounded border border-gray-400 dark:border-gray-600"`; separate the second label with a neutral left border. Change submit chrome to:

```svelte
class="h-10 rounded bg-gray-900 px-4 font-bold text-white hover:bg-gray-700 disabled:opacity-60 dark:bg-gray-100 dark:text-gray-900 dark:hover:bg-gray-300"
```

- [ ] **Step 4: Center controls with a responsive header grid**

Replace the header wrapper with `grid-cols-[1fr_1fr] md:grid-cols-[1fr_auto_1fr]`. Keep brand at row 1 left; put `header-controls` at `col-span-2 row-start-2` centered on mobile and `md:col-span-1 md:col-start-2 md:row-start-1 md:justify-self-center` on desktop. Move `ThemeSwitch` out of `header-controls` into `col-start-2 row-start-1 justify-self-end md:col-start-3`. Retain all existing `SearchControls` props verbatim.

- [ ] **Step 5: Migrate existing direction tests mechanically**

Replace each `selectOption('verb')` with:

```ts
await page
	.getByRole('group', { name: 'Search direction' })
	.getByRole('radio', { name: 'Verb-particle collocations' })
	.check();
```

In the autocomplete focus test, focus `getByRole('radio', { name: 'Noun-particle collocations' })`. In the mobile test, assert the named group. Scope theme-button lookups to `page`, not `header-controls`.

- [ ] **Step 6: Run affected browser coverage**

```bash
nix develop .#frontend --command bash -lc 'cd natsume-frontend && NATSUME_FIXTURE_COMMAND="cd .. && nix develop .#test --command python -m tests.fixture_server" npm run test:integration -- --grep "centers an accessible|supports both query directions|keeps displayed results|opens suggestions|keeps autocomplete|reuses the primary controls"'
```

Expected: selected tests PASS; changing a radio remains draft-only until submission.

- [ ] **Step 7: Commit**

```bash
git add natsume-frontend/src/lib/components/SearchControls.svelte natsume-frontend/src/routes/+page.svelte natsume-frontend/tests/test.ts
git commit -m "feat: center accessible search direction controls"
```

---

### Task 3: Identify every example's corpus

**Files:**
- Modify: `natsume-frontend/src/lib/components/CollocationItem.svelte:1-80`
- Modify: `natsume-frontend/src/lib/components/SentenceExamples.svelte:1-128`
- Modify: `natsume-frontend/tests/test.ts:34-94`

**Interfaces:**
- Consumes: `corpusStyleForSlot`, existing `corpora`, existing `colorSlots`, and `Example.corpusId`.
- Produces: `<corpus label> · <source title>:` only for multi-corpus selections.
- Preserves: request identity, page sizes, accumulation, retry, span highlighting, and full-column width.

- [ ] **Step 1: Add failing corpus-identity assertions**

After opening the first disclosure in the main Playwright flow, add:

```ts
const alphaExample = disclosure.locator('[data-testid="example-row"][data-corpus-id="alpha"]').first();
const betaExample = disclosure.locator('[data-testid="example-row"][data-corpus-id="beta"]').first();
await expect(alphaExample.getByTestId('example-source')).toContainText('Alpha · Alpha one:');
await expect(betaExample.getByTestId('example-source')).toContainText('Beta · Beta one:');
await expect(alphaExample.getByTestId('example-source')).toHaveClass(/text-violet-700/);
await expect(betaExample.getByTestId('example-source')).toHaveClass(/text-orange-700/);

const alphaColors = await Promise.all([
	alphaExample.evaluate((element) => getComputedStyle(element).borderLeftColor),
	page.locator('[data-testid="bar-segment"][data-corpus-id="alpha"]').first().evaluate((element) => getComputedStyle(element).fill),
	page.locator('[data-testid="particle-mass"] [data-corpus-id="alpha"]').first().evaluate((element) => getComputedStyle(element).backgroundColor),
	page.locator('[data-testid="corpus-swatch"][data-corpus-id="alpha"]').evaluate((element) => getComputedStyle(element).backgroundColor)
]);
expect(new Set(alphaColors)).toEqual(new Set([alphaColors[0]]));
expect(
	await betaExample.getByTestId('example-source').evaluate((element) => getComputedStyle(element).color)
).not.toBe(
	await alphaExample.getByTestId('example-source').evaluate((element) => getComputedStyle(element).color)
);

for (const [role, className] of [
	['noun', '.text-blue-600'],
	['particle', '.text-red-600'],
	['verb', '.text-green-600']
] as const) {
	const selectorColor = await page
		.getByRole('group', { name: 'Search direction' })
		.locator(`[data-role="${role}"]`)
		.first()
		.evaluate((element) => getComputedStyle(element).color);
	const sentenceColor = await disclosure
		.locator(className)
		.first()
		.evaluate((element) => getComputedStyle(element).color);
	expect(selectorColor).toBe(sentenceColor);
	expect(alphaColors).not.toContain(selectorColor);
}
```

- [ ] **Step 2: Run the main flow and verify RED**

```bash
nix develop .#frontend --command bash -lc 'cd natsume-frontend && NATSUME_FIXTURE_COMMAND="cd .. && nix develop .#test --command python -m tests.fixture_server" npm run test:integration -- --grep "searches, filters, rescales"'
```

Expected: FAIL because examples lack corpus labels and styling.

- [ ] **Step 3: Pass existing corpus values through**

Pass `corpora` and `colorSlots` from `CollocationItem` to `SentenceExamples`. Add `Corpus` to the latter's type imports, accept both props, import `corpusStyleForSlot`, and derive:

```ts
const corpusLabels = $derived(
	Object.fromEntries(corpora.map((corpus) => [corpus.id, corpus.label]))
);
```

- [ ] **Step 4: Render explicit corpus identity**

Inside the existing example loop:

```svelte
{@const corpusStyle = corpusStyleForSlot(colorSlots[example.corpusId])}
<li
	class="border-l-2 py-1 pl-1"
	style:border-left-color={corpusStyle.color}
	data-testid="example-row"
	data-corpus-id={example.corpusId}
>
	<strong class={corpusStyle.titleClass} data-testid="example-source">
		{#if selectedCorpusIds.length > 1}{corpusLabels[example.corpusId] ?? example.corpusId} · {/if}{example.sourceTitle}:
	</strong>
	{#each sentenceSegments( example.text, [{ ...example.nounSpan, type: 'noun' }, { ...example.particleSpan, type: 'particle' }, { ...example.verbSpan, type: 'verb' }] ) as segment, index (index)}
		{#if segment.className}<span class={segment.className}>{segment.text}</span
			>{:else}{segment.text}{/if}
	{/each}
</li>
```

Change the list from `space-y-2` to `divide-y dark:divide-gray-700`. Do not change `load`, effects, state transitions, limits, or accumulation.

- [ ] **Step 5: Pin the single-corpus rule**

After the existing main flow unchecks alpha, open the newly rendered `集める` disclosure and assert:

```ts
const singleCorpusDisclosure = page.locator('details').filter({ hasText: '集める' }).first();
await singleCorpusDisclosure.locator('summary').click();
await expect(singleCorpusDisclosure.getByTestId('example-source').first()).toContainText('Beta one:');
await expect(singleCorpusDisclosure.getByTestId('example-source').first()).not.toContainText('Beta ·');
```

- [ ] **Step 6: Run example flows**

```bash
nix develop .#frontend --command bash -lc 'cd natsume-frontend && NATSUME_FIXTURE_COMMAND="cd .. && nix develop .#test --command python -m tests.fixture_server" npm run test:integration -- --grep "searches, filters, rescales|loads more examples|rejects a later example page|keeps accepted examples"'
```

Expected: selected tests PASS, including duplicate occurrences and later pages.

- [ ] **Step 7: Commit**

```bash
git add natsume-frontend/src/lib/components/CollocationItem.svelte natsume-frontend/src/lib/components/SentenceExamples.svelte natsume-frontend/tests/test.ts
git commit -m "feat: identify example corpora consistently"
```

---

### Task 4: Compress collocations into spreadsheet rows

**Files:**
- Modify: `natsume-frontend/src/lib/components/ParticleOverview.svelte:20-44`
- Modify: `natsume-frontend/src/lib/components/CollocationItem.svelte:30-82`
- Modify: `natsume-frontend/src/lib/components/SentenceExamples.svelte:69-128`
- Modify: `natsume-frontend/tests/test.ts:330-385`

**Interfaces:**
- Preserves: `details/summary` keyboard behavior, bar geometry, full-width examples, and all loading/error/exhausted branches.
- Produces: adjacent separator rows at no more than 34 measured pixels, neutral state chrome, and compact example actions.

- [ ] **Step 1: Add failing density/state assertions**

Before opening the first row in `distinguishes expandable rows in light and dark mode`, add:

```ts
const firstBox = await firstSummary.boundingBox();
const secondBox = await secondSummary.boundingBox();
expect(firstBox).not.toBeNull();
expect(secondBox).not.toBeNull();
expect(firstBox?.height ?? Infinity).toBeLessThanOrEqual(34);
expect(Math.abs((secondBox?.y ?? 0) - ((firstBox?.y ?? 0) + (firstBox?.height ?? 0)))).toBeLessThanOrEqual(1);
expect(await firstSummary.evaluate((element) => getComputedStyle(element).borderRadius)).toBe('0px');
const roleColors = await page
	.getByRole('group', { name: 'Search direction' })
	.locator('[data-role]')
	.evaluateAll((elements) => elements.map((element) => getComputedStyle(element).color));
await firstSummary.focus();
expect(await firstSummary.evaluate((element) => getComputedStyle(element).outlineStyle)).not.toBe('none');
expect(await firstSummary.evaluate((element) => getComputedStyle(element).outlineOffset)).toBe('0px');
expect(roleColors).not.toContain(
	await firstSummary.evaluate((element) => getComputedStyle(element).outlineColor)
);
```

After focusing the second summary, retain the non-`none` outline check and add:

```ts
expect(await secondSummary.evaluate((element) => getComputedStyle(element).outlineOffset)).toBe('0px');
expect(roleColors).not.toContain(
	await secondSummary.evaluate((element) => getComputedStyle(element).outlineColor)
);
```

Immediately after the existing test computes `openBackground`, add:

```ts
expect(roleColors).not.toContain(openBackground);
```

- [ ] **Step 2: Run it and verify RED**

```bash
nix develop .#frontend --command bash -lc 'cd natsume-frontend && NATSUME_FIXTURE_COMMAND="cd .. && nix develop .#test --command python -m tests.fixture_server" npm run test:integration -- --grep "distinguishes expandable rows"'
```

Expected: FAIL because summaries are rounded padded cards with offset outlines.

- [ ] **Step 3: Implement the compact summary**

Remove the outer `py-1`. Give `details` `border-b border-gray-200 dark:border-gray-700`. Replace summary chrome with:

```svelte
class="grid min-h-0 w-full cursor-pointer grid-cols-[auto_minmax(5rem,2fr)_minmax(0,3fr)] items-center gap-1 px-1 py-1 text-sm leading-5 hover:bg-gray-100 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-0 focus-visible:outline-gray-700 group-open:bg-gray-200 dark:hover:bg-gray-800 dark:focus-visible:outline-gray-200 dark:group-open:bg-gray-700"
```

Reduce the SVG from `h-4` to `h-2.5`; preserve percentages, tooltips, raw frequencies, and aspect ratio.

- [ ] **Step 4: Compact examples and neutral actions**

Remove `mt-2`, rounded cards, and vertical gaps from `SentenceExamples`. Keep the root at
`class="w-full text-sm"`, the example count at `mb-0.5 px-1 text-xs text-gray-500`, the list
at `divide-y divide-gray-200 dark:divide-gray-700`, and rows at `border-l-2 py-1 pl-1`.
Use these exact compact state surfaces:

```svelte
<!-- Initial loading -->
<p class="px-1 py-1">Loading examples…</p>

<!-- Loading a later page -->
<div class="mt-1 border border-gray-300 bg-gray-50 px-2 py-1 text-gray-900 dark:border-gray-600 dark:bg-gray-800 dark:text-gray-100">
	Loading more examples…
</div>

<div class="mt-1 border border-red-200 bg-red-50 px-2 py-1 dark:border-red-800 dark:bg-red-950">
	{#if examples.length === 0}
		<p class="mb-0.5 text-red-700 dark:text-red-300">Examples could not be loaded.</p>
	{/if}
	<button
		type="button"
		class="font-medium underline"
		onclick={() => load(examples.length ? 20 : 5)}>Try again</button
	>
</div>

<!-- Identity error -->
<p class="mt-1 bg-amber-50 px-2 py-1 text-amber-800 dark:bg-amber-950 dark:text-amber-200">
	Data changed — update the search before loading more.
</p>

<!-- Exhausted -->
<div class="mt-1 bg-gray-100 px-2 py-1 text-gray-600 dark:bg-gray-800 dark:text-gray-300">
	All examples shown
</div>
```

Use neutral paging action classes:

```svelte
class="mt-1 w-full border border-gray-300 bg-gray-50 px-2 py-1 text-gray-900 hover:bg-gray-100 dark:border-gray-600 dark:bg-gray-800 dark:text-gray-100 dark:hover:bg-gray-700"
```

Keep red request errors and amber identity errors because their text states the meaning. Change `ParticleOverview` focus from blue/offset-2 to `focus-visible:outline-gray-700 focus-visible:outline-offset-0 dark:focus-visible:outline-gray-200`.

- [ ] **Step 5: Run affected visual flows**

```bash
nix develop .#frontend --command bash -lc 'cd natsume-frontend && NATSUME_FIXTURE_COMMAND="cd .. && nix develop .#test --command python -m tests.fixture_server" npm run test:integration -- --grep "distinguishes expandable rows|renders a keyboard-scrollable particle spreadsheet|searches, filters, rescales|loads more examples"'
```

Expected: selected tests PASS in both themes; scrolling and paging remain unchanged.

- [ ] **Step 6: Commit**

```bash
git add natsume-frontend/src/lib/components/ParticleOverview.svelte natsume-frontend/src/lib/components/CollocationItem.svelte natsume-frontend/src/lib/components/SentenceExamples.svelte natsume-frontend/tests/test.ts
git commit -m "style: compact collocations into spreadsheet rows"
```

---

### Task 5: Run complete gates and mark the design implemented

**Files:**
- Modify: `docs/superpowers/specs/2026-08-14-dense-color-coded-search-design.md:3`

**Interfaces:**
- Verifies: diagnostics, lint, unit tests, production build, Playwright flows, packaged frontend, and server delivery.
- Produces: retained design record with `Status: Implemented`.

- [ ] **Step 1: Run hermetic gates**

```bash
nix build .#checks.x86_64-linux.frontend -L
nix build .#checks.x86_64-linux.playwright -L
nix build .#checks.x86_64-linux.package-frontend .#checks.x86_64-linux.server-smoke -L
```

Expected: every requested derivation builds and every embedded test passes.

- [ ] **Step 2: Check tracked-file hygiene**

```bash
git diff --check
git status --short
git ls-files data '*.duckdb' '*.zip' | sed -n '1,20p'
```

Expected: no whitespace errors; only intended changes plus pre-existing untracked owner files; no corpus database/archive is tracked.

- [ ] **Step 3: Mark the design implemented**

Change line 3 to:

```markdown
Status: Implemented
```

- [ ] **Step 4: Commit documentation status**

```bash
git add docs/superpowers/specs/2026-08-14-dense-color-coded-search-design.md
git commit -m "docs: record dense interface implementation"
```

- [ ] **Step 5: Review the complete range**

```bash
git log --oneline --decorate -7
git diff --stat 93c4493..HEAD
git status --short
```

Expected: four focused implementation commits plus documentation; `AGENDA.md` and `container.nix` remain untouched.
