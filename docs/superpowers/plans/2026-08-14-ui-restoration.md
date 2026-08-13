# UI Restoration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore the compact spreadsheet-like particle browser, correct autocomplete and dark-mode behavior, render duplicate-sentence examples safely, and return the custom header icon.

**Architecture:** Keep the existing Svelte component boundaries and server-driven ranking. Make browser behavior explicit in the existing Playwright suite, then apply local markup and CSS changes: document-level theme colors, a two-group header, widget-level autocomplete focus, one native horizontal results scroller, and full-width example disclosures. No new store, scroll controller, or component abstraction is introduced.

**Tech Stack:** Svelte 5, SvelteKit, TypeScript 6, Tailwind CSS 4, Playwright 1.61, Vitest 4, Nix flakes.

## Global Constraints

- Use `/favicon.png` as the existing custom brand mark; do not generate a replacement asset.
- The page owns vertical scrolling; `ParticleOverview` owns horizontal scrolling and has no bounded height or sticky heading.
- Particle columns are exactly `20rem` wide and do not shrink or wrap at 375px or 1280px.
- Native scrolling is the only scrolling implementation: no synchronized scroller, arrows, resize listener, or scroll-position state.
- Autocomplete is open only while focus is within the search form and suggestions exist.
- Example lists remain unkeyed because they are assigned once and are never reordered or spliced.
- Do not change API types, ranking, corpus selection, or the serialized `ParticleColumn` collocation key in this tranche.
- Run frontend commands through `nix develop .#frontend`; do not create a repository-root virtual environment.

---

### Task 1: Document theme and two-group header

**Files:**
- Modify: `natsume-frontend/src/tailwind.css`
- Modify: `natsume-frontend/src/routes/+page.svelte`
- Test: `natsume-frontend/tests/test.ts`

**Interfaces:**
- Consumes: existing `ThemeSwitch`, `SearchControls`, and `/favicon.png`.
- Produces: an `html`/`body` theme boundary and a header with `data-testid="brand"` and `data-testid="header-controls"` groups for stable browser assertions.

- [ ] **Step 1: Install the locked frontend dependencies**

Run:

```bash
nix develop .#frontend --command bash -lc 'cd natsume-frontend && npm ci'
```

Expected: npm exits 0 without changing `package-lock.json`.

- [ ] **Step 2: Write failing browser assertions for grouping and computed theme colors**

Replace the direct-child tag assertion in `supports both query directions and theme control` with:

```ts
const brand = page.getByTestId('brand');
const controls = page.getByTestId('header-controls');
await expect(brand.getByRole('img', { name: 'Natsume Simple' })).toHaveAttribute(
	'src',
	'/favicon.png'
);
await expect(brand.getByRole('heading', { name: 'Natsume Simple' })).toBeVisible();
await expect(controls.getByRole('combobox', { name: 'Search term' })).toBeVisible();
await expect(controls.getByRole('button', { name: 'Toggle dark mode' })).toBeVisible();

const light = await page.evaluate(() => ({
	htmlBackground: getComputedStyle(document.documentElement).backgroundColor,
	bodyBackground: getComputedStyle(document.body).backgroundColor,
	bodyColor: getComputedStyle(document.body).color
}));
await page.getByRole('button', { name: 'Toggle dark mode' }).click();
await expect(page.locator('html')).toHaveClass(/dark/);
const dark = await page.evaluate(() => ({
	htmlBackground: getComputedStyle(document.documentElement).backgroundColor,
	bodyBackground: getComputedStyle(document.body).backgroundColor,
	bodyColor: getComputedStyle(document.body).color
}));
expect(dark).not.toEqual(light);
expect(dark.htmlBackground).not.toBe('rgba(0, 0, 0, 0)');
expect(dark.bodyBackground).not.toBe('rgba(0, 0, 0, 0)');
await page.locator('main').evaluate((element) => element.replaceChildren());
const shortPage = await page.evaluate(() => ({
	bodyBackground: getComputedStyle(document.body).backgroundColor,
	bodyColor: getComputedStyle(document.body).color,
	bodyHeight: document.body.getBoundingClientRect().height,
	viewportHeight: window.innerHeight
}));
expect(shortPage.bodyBackground).toBe(dark.bodyBackground);
expect(shortPage.bodyColor).toBe(dark.bodyColor);
expect(shortPage.bodyHeight).toBeGreaterThanOrEqual(shortPage.viewportHeight);
```

- [ ] **Step 3: Run the focused test and confirm the regression**

Run:

```bash
nix develop .#frontend --command bash -lc 'cd natsume-frontend && npm run test:integration -- --grep "supports both query directions"'
```

Expected: FAIL because the test IDs and image do not exist, and the computed page colors do not change.

- [ ] **Step 4: Add the document theme boundary and header groups**

Extend `tailwind.css` with:

```css
@layer base {
	html {
		@apply min-h-full bg-white text-gray-900 dark:bg-gray-950 dark:text-gray-100;
	}

	body {
		@apply min-h-screen bg-inherit text-inherit;
	}
}
```

Change the header content in `+page.svelte` to this shape:

```svelte
<div class="mx-auto flex max-w-screen-2xl flex-wrap items-center gap-3 p-4">
	<div class="flex items-center gap-3" data-testid="brand">
		<img class="h-8 w-8" src="/favicon.png" alt="Natsume Simple" />
		<h1 class="text-2xl font-bold" tabindex="-1">Natsume Simple</h1>
	</div>
	<div class="ml-auto flex flex-wrap items-center justify-end gap-2" data-testid="header-controls">
		<SearchControls
			bind:term={controller.term}
			bind:pos={controller.pos}
			loading={controller.status === 'loading'}
			onsubmit={() => controller.submit()}
			findSuggestions={async (query, pos) => (await client.getSuggestions(query, pos)).suggestions}
		/>
		<ThemeSwitch />
	</div>
</div>
```

Keep the existing `SearchControls` props unchanged. The control group may wrap as one group; the theme switch remains after the form in DOM order.

- [ ] **Step 5: Run the focused test**

Run the command from Step 3.

Expected: PASS.

- [ ] **Step 6: Commit the theme and header repair**

```bash
git add natsume-frontend/src/tailwind.css natsume-frontend/src/routes/+page.svelte natsume-frontend/tests/test.ts
git commit -m "fix: restore page theme and header brand"
```

---

### Task 2: Focus-controlled autocomplete

**Files:**
- Modify: `natsume-frontend/src/lib/components/SearchControls.svelte`
- Test: `natsume-frontend/tests/test.ts`

**Interfaces:**
- Consumes: existing `findSuggestions(query, pos)` callback and bound `term`/`pos` props.
- Produces: `focusedWithin: boolean`; listbox visibility is `focusedWithin && suggestions.length > 0` and closes on Escape, submit, selection, or form focusout.

- [ ] **Step 1: Add the failing browser behavior test**

Add:

```ts
test('opens suggestions only while focus remains in the search widget', async ({ page }) => {
	await page.goto('/');
	const search = page.getByRole('combobox', { name: 'Search term' });
	await expect(search).toHaveValue('時間');
	await page.waitForTimeout(350);
	await expect(search).toHaveAttribute('aria-expanded', 'false');

	await search.focus();
	await expect(search).toHaveAttribute('aria-expanded', 'true');
	const suggestion = page.getByRole('option').first().getByRole('button');
	const label = await suggestion.textContent();
	await suggestion.click();
	await expect(search).toHaveValue(label?.trim() ?? '');
	await expect(search).toHaveAttribute('aria-expanded', 'false');

	await search.fill('時間');
	await page.waitForTimeout(350);
	await expect(search).toHaveAttribute('aria-expanded', 'true');
	await page.getByRole('heading', { name: 'Natsume Simple' }).focus();
	await expect(search).toHaveAttribute('aria-expanded', 'false');
});
```

The brand heading's `tabindex="-1"` from Task 1 gives the final focus-leave assertion a deterministic target without adding a tab stop.

- [ ] **Step 2: Confirm the asynchronous-open failure**

Run:

```bash
nix develop .#frontend --command bash -lc 'cd natsume-frontend && npm run test:integration -- --grep "opens suggestions only"'
```

Expected: FAIL because the initial debounced response sets `aria-expanded="true"` before focus.

- [ ] **Step 3: Implement widget-level focus state**

In `SearchControls.svelte`:

```ts
let focusedWithin = $state(false);

function focusout(event: FocusEvent & { currentTarget: HTMLFormElement }) {
	const next = event.relatedTarget;
	if (!(next instanceof Node) || !event.currentTarget.contains(next)) {
		focusedWithin = false;
		open = false;
	}
}
```

Change the debounce completion to:

```ts
suggestions = await findSuggestions(query, position);
open = focusedWithin && suggestions.length > 0;
active = -1;
```

Add these form handlers:

```svelte
onfocusin={() => (focusedWithin = true)}
onfocusout={focusout}
```

Keep the input handler as:

```svelte
onfocus={() => (open = suggestions.length > 0)}
```

Bound the listbox without changing its anchoring:

```svelte
class="absolute z-20 mt-1 max-h-64 min-w-full overflow-y-auto rounded border bg-white shadow dark:border-gray-600 dark:bg-gray-800"
```

- [ ] **Step 4: Run the focused browser test**

Run the command from Step 2.

Expected: PASS, including click selection after focus moves onto the suggestion button.

- [ ] **Step 5: Run Svelte static checking**

```bash
nix develop .#frontend --command bash -lc 'cd natsume-frontend && npm run check'
```

Expected: PASS; in particular, the typed `focusout` handler is accepted without `any` or suppression comments.

- [ ] **Step 6: Commit the autocomplete repair**

```bash
git add natsume-frontend/src/lib/components/SearchControls.svelte natsume-frontend/src/routes/+page.svelte natsume-frontend/tests/test.ts
git commit -m "fix: bind autocomplete visibility to widget focus"
```

---

### Task 3: Native horizontal spreadsheet region

**Files:**
- Modify: `natsume-frontend/src/lib/components/ParticleOverview.svelte`
- Modify: `natsume-frontend/src/lib/components/ParticleColumn.svelte`
- Test: `natsume-frontend/tests/test.ts`

**Interfaces:**
- Consumes: `result.particleGroups` in API order.
- Produces: one `data-testid="particle-overview"` region with accessible name `Particle collocations`; direct child columns expose `data-testid="particle-column"` and fixed 320px computed width.

- [ ] **Step 1: Add a reusable browser assertion for both specified viewports**

Add this helper and test:

```ts
async function expectSpreadsheet(page: import('@playwright/test').Page, width: number) {
	await page.setViewportSize({ width, height: 844 });
	await page.goto('/');
	const region = page.getByRole('region', { name: 'Particle collocations' });
	await expect(region).toBeVisible();
	const dimensions = await region.evaluate((element) => ({
		clientWidth: element.clientWidth,
		scrollWidth: element.scrollWidth
	}));
	expect(dimensions.scrollWidth).toBeGreaterThan(dimensions.clientWidth);
	const columns = region.getByTestId('particle-column');
	expect(await columns.count()).toBeGreaterThan(1);
	for (const column of await columns.all()) {
		expect(await column.evaluate((element) => getComputedStyle(element).width)).toBe('320px');
	}
	await region.focus();
	await page.keyboard.press('ArrowRight');
	expect(await region.evaluate((element) => element.scrollLeft)).toBeGreaterThan(0);
	await page.keyboard.press('End');
	expect(await region.evaluate((element) => element.scrollLeft)).toBeGreaterThan(0);
}

test('renders a keyboard-scrollable particle spreadsheet', async ({ page }) => {
	await expectSpreadsheet(page, 375);
	await expectSpreadsheet(page, 1280);
});
```

- [ ] **Step 2: Confirm the card grid fails the contract**

Run:

```bash
nix develop .#frontend --command bash -lc 'cd natsume-frontend && npm run test:integration -- --grep "particle spreadsheet"'
```

Expected: FAIL because there is no labelled region and columns are a wrapping responsive grid.

- [ ] **Step 3: Replace the grid with a focusable horizontal region**

Use this outer shape in `ParticleOverview.svelte`:

```svelte
<div
	class="flex overflow-x-auto border-y focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-blue-600 dark:border-gray-700"
	role="region"
	aria-label="Particle collocations"
	tabindex="0"
	data-testid="particle-overview"
>
	{#each result.particleGroups as group (group.particle)}
		<ParticleColumn
			{client}
			{group}
			{corpora}
			selectedCorpusIds={result.selectedCorpusIds}
			rankBy={result.rankBy}
			{pos}
		/>
	{/each}
</div>
```

Do not add `overflow-y`, `max-height`, sticky positioning, or scroll event handlers.

- [ ] **Step 4: Turn each card into a fixed-width spreadsheet column**

Change the `ParticleColumn` root to:

```svelte
<section
	class="w-80 shrink-0 border-r px-3 py-2 last:border-r-0 dark:border-gray-700"
	data-testid="particle-column"
>
```

Keep the heading normal-flow, retain the server count, and do not alter the collocation key in this task.

- [ ] **Step 5: Run the focused browser test and static checks**

```bash
nix develop .#frontend --command bash -lc 'cd natsume-frontend && npm run test:integration -- --grep "particle spreadsheet" && npm run check'
```

Expected: PASS at 375px and 1280px, with keyboard movement changing `scrollLeft`.

- [ ] **Step 6: Commit the spreadsheet layout**

```bash
git add natsume-frontend/src/lib/components/ParticleOverview.svelte natsume-frontend/src/lib/components/ParticleColumn.svelte natsume-frontend/tests/test.ts
git commit -m "fix: restore horizontal particle spreadsheet"
```

---

### Task 4: Full-width duplicate-safe examples

**Files:**
- Modify: `tests/database_fixture.py`
- Modify: `natsume-frontend/src/lib/components/CollocationItem.svelte`
- Modify: `natsume-frontend/src/lib/components/SentenceExamples.svelte`
- Test: `natsume-frontend/tests/test.ts`

**Interfaces:**
- Consumes: `/api/examples` occurrence rows, including multiple occurrences from one sentence.
- Produces: a full-width `<details>` whose summary owns both bar and label; the example list is unkeyed and accepts duplicate `sentenceId` values.

- [ ] **Step 1: Make the fixture represent duplicate occurrences from one sentence**

Change fixture sentence 1 to `情報を集める。情報を集める。`, add a second occurrence for sentence 1 with spans `(7, 9), (9, 10), (10, 13)`, and increase alpha's `collocation_count` from 6 to 7. This creates two `/api/examples` rows with the same sentence ID and different spans without adding a synthetic public identifier.

- [ ] **Step 2: Strengthen the existing expansion browser test**

After clicking `集める`, assert:

```ts
const disclosure = page.locator('details').filter({ hasText: '集める' }).first();
await expect(disclosure.locator('li').filter({ hasText: '情報を集める。情報を集める。' })).toHaveCount(2);
await expect(page.getByText('Loading examples…')).toHaveCount(0);
expect(await page.evaluate(() => document.body.scrollWidth)).toBeGreaterThan(0);

const disclosureBox = await disclosure.boundingBox();
const examplesBox = await disclosure.locator('[data-testid="sentence-examples"]').boundingBox();
expect(disclosureBox).not.toBeNull();
expect(examplesBox).not.toBeNull();
expect(Math.abs((examplesBox?.x ?? 0) - (disclosureBox?.x ?? 0))).toBeLessThanOrEqual(1);
expect(Math.abs((examplesBox?.width ?? 0) - (disclosureBox?.width ?? 0))).toBeLessThanOrEqual(2);
```

Also register and assert page errors explicitly:

```ts
const pageErrors: string[] = [];
page.on('pageerror', (error) => pageErrors.push(error.message));
expect(pageErrors.filter((message) => message.includes('each_key_duplicate'))).toEqual([]);
```

- [ ] **Step 3: Confirm the duplicate-key regression**

Run:

```bash
nix develop .#frontend --command bash -lc 'cd natsume-frontend && npm run test:integration -- --grep "safely expands examples"'
```

Expected: FAIL with `each_key_duplicate` or with the disclosure remaining in its loading state.

- [ ] **Step 4: Put the bar inside the disclosure summary**

Replace the sibling layout in `CollocationItem.svelte` with one `SentenceExamples` call and pass the existing `segments` and `colors` as props. In `SentenceExamples.svelte`, import the existing `StackSegment` type from `$lib/presentation/search` and extend the props with:

```ts
segments: StackSegment[];
colors: string[];
```

Render this complete structural shape, retaining the existing status branches and
safe `sentenceSegments` loop inside the marked body:

```svelte
<details
	class="w-full min-w-0 py-1"
	ontoggle={(event) => {
		if (event.currentTarget.open) load();
	}}
>
	<summary class="flex cursor-pointer items-center gap-2 font-medium">
		<svg width="64" height="20" aria-hidden="true" class="shrink-0 rounded">
			{#each segments as segment, index (segment.corpusId)}
				<rect
					x={`${segment.offset}%`}
					width={`${segment.percentage}%`}
					height="20"
					fill={colors[index % colors.length]}
				/>
			{/each}
		</svg>
		<span>{label}</span>
	</summary>
	<div class="mt-2 w-full text-sm" aria-live="polite" data-testid="sentence-examples">
		{#if status === 'loading'}
			<p>Loading examples…</p>
		{:else if status === 'empty'}
			<p>No examples found.</p>
		{:else if status === 'error'}
			<p class="text-red-700 dark:text-red-300">Examples could not be loaded.</p>
		{:else}
			<ul class="space-y-2">
				{#each examples as example}
					<li class="rounded bg-gray-100 p-2 dark:bg-gray-800">
						<strong>{example.sourceTitle}:</strong>
						{#each sentenceSegments( example.text, [{ ...example.nounSpan, type: 'noun' }, { ...example.particleSpan, type: 'particle' }, { ...example.verbSpan, type: 'verb' }] ) as segment, index (index)}
							{#if segment.className}<span class={segment.className}>{segment.text}</span
								>{:else}{segment.text}{/if}
						{/each}
					</li>
				{/each}
			</ul>
		{/if}
	</div>
</details>
```

Remove `ml-4`, and change `{#each examples as example (example.sentenceId)}` to `{#each examples as example}`. Keep the existing safe `sentenceSegments` rendering.

- [ ] **Step 5: Run the focused browser test and unit checks**

```bash
nix develop .#frontend --command bash -lc 'cd natsume-frontend && npm run test:integration -- --grep "safely expands examples" && npm run test:unit -- --run && npm run check'
```

Expected: PASS; both occurrences render and no Svelte page error occurs.

- [ ] **Step 6: Commit the example repair**

```bash
git add tests/database_fixture.py natsume-frontend/src/lib/components/CollocationItem.svelte natsume-frontend/src/lib/components/SentenceExamples.svelte natsume-frontend/tests/test.ts
git commit -m "fix: render examples at full column width"
```

---

### Task 5: Complete frontend verification

**Files:**
- Modify only if a verification failure identifies a defect in a Task 1–4 file.

**Interfaces:**
- Consumes: all UI changes from Tasks 1–4.
- Produces: Nix-built frontend, unit/static checks, and Chromium regression evidence.

- [ ] **Step 1: Run formatting and lint checks without mutation**

```bash
nix develop .#frontend --command bash -lc 'cd natsume-frontend && npm run lint && npm run check'
```

Expected: PASS. If Prettier reports a diff, run `npm run format`, inspect the exact frontend paths changed, and commit only those paths.

- [ ] **Step 2: Run all frontend tests and production build**

```bash
nix develop .#frontend --command bash -lc 'cd natsume-frontend && npm run test:unit -- --run && npm run test:integration && npm run build'
```

Expected: all Vitest and Playwright tests pass; Vite production build exits 0.

- [ ] **Step 3: Run the hermetic Nix checks**

```bash
nix build .#checks.x86_64-linux.frontend
nix build .#checks.x86_64-linux.playwright
```

Expected: both derivations build successfully.

- [ ] **Step 4: Inspect the deep-scroll smoke at both widths**

Use the Playwright assertions from Task 3 to scroll the page until the bottom scrollbar is off screen, focus the region, press ArrowRight and End, and confirm `scrollLeft` increases at both 375px and 1280px. Record failure as the evidence-trigger for a future scroll affordance; do not add one preemptively.

- [ ] **Step 5: Commit only if verification required a correction**

```bash
git status --short
git add natsume-frontend/src/tailwind.css natsume-frontend/src/routes/+page.svelte natsume-frontend/src/lib/components/SearchControls.svelte natsume-frontend/src/lib/components/ParticleOverview.svelte natsume-frontend/src/lib/components/ParticleColumn.svelte natsume-frontend/src/lib/components/CollocationItem.svelte natsume-frontend/src/lib/components/SentenceExamples.svelte natsume-frontend/tests/test.ts tests/database_fixture.py
git commit -m "test: complete UI restoration coverage"
```

If the worktree has no plan-owned changes, do not create an empty commit. Leave `AGENDA.md` and `container.nix` untouched.
