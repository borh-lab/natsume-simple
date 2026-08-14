import { expect, test } from '@playwright/test';

async function expectSpreadsheet(page: import('@playwright/test').Page, width: number) {
	await page.setViewportSize({ width, height: 844 });
	await page.goto('/');
	await expect(page.getByText(/results for “時間”/)).toBeVisible();
	await page.getByRole('combobox', { name: 'Search term' }).fill('情報');
	await page.getByRole('button', { name: 'Update results' }).click();
	const region = page.getByRole('region', { name: 'Particle collocations' });
	await expect(region).toBeVisible();
	const columns = region.getByTestId('particle-column');
	await expect.poll(() => columns.count()).toBeGreaterThan(1);
	const dimensions = await region.evaluate((element) => ({
		clientWidth: element.clientWidth,
		scrollWidth: element.scrollWidth
	}));
	const shouldOverflow = width < 2560;
	if (shouldOverflow) expect(dimensions.scrollWidth).toBeGreaterThan(dimensions.clientWidth);
	else expect(dimensions.scrollWidth).toBeLessThanOrEqual(dimensions.clientWidth + 1);
	for (const column of await columns.all()) {
		const columnWidth = await column.evaluate((element) => element.getBoundingClientRect().width);
		expect(columnWidth).toBeGreaterThanOrEqual(320);
		if (!shouldOverflow) expect(columnWidth).toBeGreaterThan(320);
	}
	if (shouldOverflow) {
		await region.focus();
		await page.keyboard.press('ArrowRight');
		await expect.poll(() => region.evaluate((element) => element.scrollLeft)).toBeGreaterThan(0);
		await page.keyboard.press('End');
		await expect.poll(() => region.evaluate((element) => element.scrollLeft)).toBeGreaterThan(0);
	}
}

test('searches, filters, rescales, and safely expands examples', async ({ page }) => {
	const pageErrors: string[] = [];
	page.on('pageerror', (error) => pageErrors.push(error.message));
	await page.goto('/');
	await expect(page.getByRole('button', { name: 'Go' })).toBeEnabled();

	const search = page.getByRole('combobox', { name: 'Search term' });
	await search.fill('情報');
	await Promise.all([
		page.waitForResponse((response) => {
			const url = new URL(response.url());
			return url.pathname === '/api/collocations' && url.searchParams.get('term') === '情報';
		}),
		page.getByRole('button', { name: 'Update results' }).click()
	]);
	await expect(page.getByRole('heading', { name: 'を', exact: true })).toBeVisible();

	const collocationRequests: string[] = [];
	page.on('request', (request) => {
		if (new URL(request.url()).pathname === '/api/collocations')
			collocationRequests.push(request.url());
	});
	const collocation = page.locator('summary').filter({ hasText: '集める' });
	await expect(collocation.locator('svg')).toBeVisible();
	await expect(collocation.getByText('集める', { exact: true })).toBeVisible();
	await collocation.click();
	const disclosure = page.locator('details').filter({ hasText: '集める' }).first();
	await expect(disclosure.locator(':scope > [data-testid="sentence-examples"]')).toBeVisible();
	await expect(
		disclosure.locator('li').filter({ hasText: '情報を集める。情報を集める。' })
	).toHaveCount(2);
	await expect(page.getByText('Loading examples…')).toHaveCount(0);
	await expect(page.locator('img[src="x"]')).toHaveCount(0);
	const disclosureBox = await disclosure.boundingBox();
	const examplesBox = await disclosure.locator('[data-testid="sentence-examples"]').boundingBox();
	expect(disclosureBox).not.toBeNull();
	expect(examplesBox).not.toBeNull();
	expect(Math.abs((examplesBox?.x ?? 0) - (disclosureBox?.x ?? 0))).toBeLessThanOrEqual(1);
	expect(Math.abs((examplesBox?.width ?? 0) - (disclosureBox?.width ?? 0))).toBeLessThanOrEqual(2);
	expect(pageErrors.filter((message) => message.includes('each_key_duplicate'))).toEqual([]);

	const requestsBeforeScaleChange = collocationRequests.length;
	const scale = page.getByRole('combobox', { name: 'Bar scale', exact: true });
	await expect(scale).toHaveValue('particle');
	await scale.selectOption('global');
	await expect(scale).toHaveValue('global');
	await expect(disclosure).toHaveAttribute('open', '');
	expect(collocationRequests).toHaveLength(requestsBeforeScaleChange);
	await expect(page.getByTestId('particle-mass').first()).toHaveAttribute(
		'aria-label',
		/% of selected frequency/
	);
	await expect(page.getByTestId('bar-segment').first().locator('title')).toContainText(
		'occurrences'
	);

	await page.locator('#corpus-alpha').uncheck();
	await expect(page.locator('#corpus-alpha')).not.toBeChecked();
	await expect(page.locator('summary').filter({ hasText: '集める' })).toBeVisible();
});

test('loads more examples without hiding the accepted page', async ({ page }) => {
	let initialCount = 0;
	let identity = { selectedCorpusIds: [] as string[], databaseBuildId: '' };
	await page.route('**/api/examples**', async (route) => {
		const url = new URL(route.request().url());
		if (url.searchParams.get('offset') !== '0') {
			expect(url.searchParams.get('offset')).toBe(String(initialCount));
			expect(url.searchParams.get('limit')).toBe('20');
			await route.fulfill({
				json: {
					examples: [
						{
							corpusId: 'alpha',
							sourceId: 99,
							sourceTitle: 'Additional source',
							sentenceId: 99,
							text: '追加の情報を集める。',
							nounSpan: { start: 3, end: 5 },
							particleSpan: { start: 5, end: 6 },
							verbSpan: { start: 6, end: 9 }
						}
					],
					hasMore: false,
					selectedCorpusIds: identity.selectedCorpusIds,
					databaseBuildId: identity.databaseBuildId
				}
			});
			return;
		}
		const response = await route.fetch();
		const body = await response.json();
		initialCount = body.examples.length;
		identity = {
			selectedCorpusIds: body.selectedCorpusIds,
			databaseBuildId: body.databaseBuildId
		};
		body.hasMore = true;
		await route.fulfill({ response, json: body });
	});

	await page.goto('/');
	await expect(page.getByRole('button', { name: 'Go' })).toBeEnabled();
	await page.getByRole('combobox', { name: 'Search term' }).fill('情報');
	await page.getByRole('button', { name: 'Update results' }).click();
	const disclosure = page.locator('details').filter({ hasText: '集める' }).first();
	await disclosure.locator('summary').click();
	await expect(disclosure.getByRole('button', { name: 'Load more examples' })).toBeVisible();
	await expect(disclosure.getByText(`${initialCount} examples shown`)).toBeVisible();
	await disclosure.getByRole('button', { name: 'Load more examples' }).click();
	await expect(disclosure.getByText('追加の情報を集める。')).toBeVisible();
	await expect(disclosure.getByText('All examples shown')).toBeVisible();
});

test('rejects a later example page from a different artifact', async ({ page }) => {
	let initialCount = 0;
	let selectedCorpusIds: string[] = [];
	await page.route('**/api/examples**', async (route) => {
		const url = new URL(route.request().url());
		if (url.searchParams.get('offset') !== '0') {
			await route.fulfill({
				json: {
					examples: [],
					hasMore: false,
					selectedCorpusIds,
					databaseBuildId: 'different-build'
				}
			});
			return;
		}
		const response = await route.fetch();
		const body = await response.json();
		initialCount = body.examples.length;
		selectedCorpusIds = body.selectedCorpusIds;
		body.hasMore = true;
		await route.fulfill({ response, json: body });
	});

	await page.goto('/');
	await expect(page.getByRole('button', { name: 'Go' })).toBeEnabled();
	await page.getByRole('combobox', { name: 'Search term' }).fill('情報');
	await page.getByRole('button', { name: 'Update results' }).click();
	const disclosure = page.locator('details').filter({ hasText: '集める' }).first();
	await disclosure.locator('summary').click();
	await disclosure.getByRole('button', { name: 'Load more examples' }).click();
	await expect(
		disclosure.getByText('Data changed — update the search before loading more.')
	).toBeVisible();
	expect(await disclosure.locator('li').count()).toBe(initialCount);
});

test('keeps accepted examples visible when a later page fails', async ({ page }) => {
	let initialCount = 0;
	await page.route('**/api/examples**', async (route) => {
		const url = new URL(route.request().url());
		if (url.searchParams.get('offset') !== '0') {
			await route.fulfill({
				status: 500,
				contentType: 'application/json',
				body: JSON.stringify({
					error: { code: 'query_failed', message: 'failed', requestId: 'test-request' }
				})
			});
			return;
		}
		const response = await route.fetch();
		const body = await response.json();
		initialCount = body.examples.length;
		body.hasMore = true;
		await route.fulfill({ response, json: body });
	});

	await page.goto('/');
	await expect(page.getByRole('button', { name: 'Go' })).toBeEnabled();
	await page.getByRole('combobox', { name: 'Search term' }).fill('情報');
	await page.getByRole('button', { name: 'Update results' }).click();
	const disclosure = page.locator('details').filter({ hasText: '集める' }).first();
	await disclosure.locator('summary').click();
	await disclosure.getByRole('button', { name: 'Load more examples' }).click();
	await expect(disclosure.getByRole('button', { name: 'Try again' })).toBeVisible();
	expect(await disclosure.locator('li').count()).toBe(initialCount);
});

test('loads more collocations in only the selected particle column', async ({ page }) => {
	let expectedTotal = 0;
	let initialWoCount = 0;
	let initialResponse: {
		particleGroups: Array<{
			particle: string;
			totalMatchingCollocations: number;
			returnedCount: number;
			items: unknown[];
			corpusDistribution: unknown[];
		}>;
		selectedCorpusIds: string[];
		databaseBuildId: string;
	} | null = null;

	await page.route('**/api/collocations**', async (route) => {
		const url = new URL(route.request().url());
		if (url.searchParams.get('term') !== '情報') {
			await route.continue();
			return;
		}
		if (url.searchParams.get('particle') === 'を') {
			if (!initialResponse) throw new Error('targeted page preceded initial response');
			expect(url.searchParams.get('offsetPerParticle')).toBe(String(initialWoCount));
			const initialGroup = initialResponse.particleGroups.find((group) => group.particle === 'を');
			if (!initialGroup) throw new Error('fixture response omitted を');
			await route.fulfill({
				json: {
					particleGroups: [
						{
							...initialGroup,
							returnedCount: 1,
							items: [
								{
									noun: '情報',
									particle: 'を',
									verb: '追加する',
									totalRawFrequency: 1,
									meanFrequencyPerMillion: 1,
									contributions: [
										{ corpusId: initialResponse.selectedCorpusIds[0], rawFrequency: 1 }
									]
								}
							]
						}
					],
					selectedCorpusIds: initialResponse.selectedCorpusIds,
					databaseBuildId: initialResponse.databaseBuildId
				}
			});
			return;
		}

		const response = await route.fetch();
		const body = await response.json();
		initialResponse = body;
		const wo = body.particleGroups.find((group: { particle: string }) => group.particle === 'を');
		if (!wo) throw new Error('fixture response omitted を');
		initialWoCount = wo.items.length;
		wo.totalMatchingCollocations = initialWoCount + 1;
		expectedTotal = body.particleGroups.reduce(
			(sum: number, group: { totalMatchingCollocations: number }) =>
				sum + group.totalMatchingCollocations,
			0
		);
		await route.fulfill({ response, json: body });
	});

	await page.goto('/');
	await expect(page.getByRole('button', { name: 'Go' })).toBeEnabled();
	const search = page.getByRole('combobox', { name: 'Search term' });
	await search.fill('情報');
	await Promise.all([
		page.waitForResponse((response) => {
			const url = new URL(response.url());
			return url.pathname === '/api/collocations' && url.searchParams.get('term') === '情報';
		}),
		page.getByRole('button', { name: 'Update results' }).click()
	]);
	const woColumn = page.getByTestId('particle-column').filter({
		has: page.getByRole('heading', { name: 'を', exact: true })
	});
	const gaColumn = page.getByTestId('particle-column').filter({
		has: page.getByRole('heading', { name: 'が', exact: true })
	});
	await expect(
		page.getByText(`${expectedTotal} matching results for “情報” · Noun–particle search`, {
			exact: true
		})
	).toBeVisible();
	await expect(
		woColumn.getByText(`Showing ${initialWoCount} of ${initialWoCount + 1}`)
	).toBeVisible();
	const otherCount = await gaColumn.locator('summary').count();
	const overview = page.getByRole('region', { name: 'Particle collocations' });
	await overview.evaluate((element) => (element.scrollLeft = 100));
	const scrollLeft = await overview.evaluate((element) => element.scrollLeft);

	await woColumn.getByRole('button', { name: 'Load 1 more' }).click();

	await expect(woColumn.getByText('追加する', { exact: true })).toBeVisible();
	expect(await gaColumn.locator('summary').count()).toBe(otherCount);
	await expect(
		page.getByText(`${expectedTotal} matching results for “情報” · Noun–particle search`, {
			exact: true
		})
	).toBeVisible();
	await expect(page.getByRole('combobox', { name: 'Bar scale' })).toHaveValue('particle');
	expect(await overview.evaluate((element) => element.scrollLeft)).toBe(scrollLeft);
});

test('distinguishes expandable rows in light and dark mode', async ({ page }) => {
	await page.goto('/');
	await expect(page.getByRole('button', { name: 'Go' })).toBeEnabled();
	await page.getByRole('combobox', { name: 'Search term' }).fill('情報');
	await page.getByRole('button', { name: 'Update results' }).click();
	const column = page.getByTestId('particle-column').filter({
		has: page.getByRole('heading', { name: 'を', exact: true })
	});
	const details = column.locator('details');
	await expect.poll(() => details.count()).toBeGreaterThan(1);
	const firstSummary = details.nth(0).locator('summary');
	const secondSummary = details.nth(1).locator('summary');
	const chevron = firstSummary.locator('[aria-hidden="true"]');
	await expect(chevron).toBeVisible();
	const collapsedTransform = await chevron.evaluate(
		(element) => getComputedStyle(element).transform
	);
	const collapsedBackground = await firstSummary.evaluate(
		(element) => getComputedStyle(element).backgroundColor
	);
	const secondLabel = await secondSummary.locator('span').last().innerText();
	expect(
		await details.nth(0).evaluate((element) => {
			const next = element.parentElement?.nextElementSibling?.querySelector('summary');
			return next?.querySelector('span:last-child')?.textContent?.trim();
		})
	).toBe(secondLabel.trim());

	await firstSummary.click();
	await expect(details.nth(0)).toHaveAttribute('open', '');
	const openBackground = await firstSummary.evaluate(
		(element) => getComputedStyle(element).backgroundColor
	);
	expect(openBackground).not.toBe(collapsedBackground);
	await expect
		.poll(() => chevron.evaluate((element) => getComputedStyle(element).transform))
		.not.toBe(collapsedTransform);

	await page.getByRole('button', { name: 'Toggle dark mode' }).click();
	const darkOpenBackground = await firstSummary.evaluate(
		(element) => getComputedStyle(element).backgroundColor
	);
	const darkCollapsedBackground = await secondSummary.evaluate(
		(element) => getComputedStyle(element).backgroundColor
	);
	expect(darkOpenBackground).not.toBe(darkCollapsedBackground);
	await secondSummary.focus();
	expect(
		await secondSummary.evaluate((element) => getComputedStyle(element).outlineStyle)
	).not.toBe('none');
});

test('supports both query directions and theme control', async ({ page }) => {
	await page.goto('/');
	await expect(page.getByRole('button', { name: 'Go' })).toBeEnabled();
	const brand = page.getByTestId('brand');
	const controls = page.getByTestId('header-controls');
	await expect(brand.getByRole('img', { name: 'Natsume Simple' })).toHaveAttribute(
		'src',
		'/favicon.png'
	);
	await expect(brand.getByRole('heading', { name: 'Natsume Simple' })).toBeVisible();
	await expect(controls.getByRole('combobox', { name: 'Search term' })).toBeVisible();
	await expect(controls.getByRole('button', { name: 'Toggle dark mode' })).toBeVisible();
	const direction = page.getByLabel('Search direction');
	const search = page.getByRole('combobox', { name: 'Search term' });
	await direction.selectOption('verb');
	await search.fill('集める');
	await Promise.all([
		page.waitForResponse((response) => {
			const url = new URL(response.url());
			return (
				url.pathname === '/api/collocations' &&
				url.searchParams.get('pos') === 'verb' &&
				url.searchParams.get('term') === '集める'
			);
		}),
		page.getByRole('button', { name: 'Update results' }).click()
	]);
	await expect(page.locator('summary').filter({ hasText: '情報' })).toBeVisible();

	const light = await page.evaluate(() => ({
		htmlBackground: getComputedStyle(document.documentElement).backgroundColor,
		bodyBackground: getComputedStyle(document.body).backgroundColor,
		bodyColor: getComputedStyle(document.body).color
	}));
	await controls.getByRole('button', { name: 'Toggle dark mode' }).click();
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
});

test('keeps displayed results tied to the submitted search while controls are edited', async ({
	page
}) => {
	await page.goto('/');
	const search = page.getByRole('combobox', { name: 'Search term' });
	await search.fill('情報');
	await page.getByRole('button', { name: 'Update results' }).click();
	await expect(page.getByText(/results for “情報” · Noun–particle search/)).toBeVisible();
	await expect(page.locator('summary').filter({ hasText: '集める' })).toBeVisible();

	await page.getByLabel('Search direction').selectOption('verb');
	await search.fill('集める');

	await expect(page.getByText(/results for “情報” · Noun–particle search/)).toBeVisible();
	await expect(page.locator('summary').filter({ hasText: '集める' })).toBeVisible();
	await expect(page.getByText('Controls changed — update results to apply them.')).toBeVisible();
	await expect(page.getByRole('button', { name: 'Update results' })).toBeVisible();
});

test('opens suggestions only while focus remains in the search widget', async ({ page }) => {
	await page.goto('/');
	const search = page.getByRole('combobox', { name: 'Search term' });
	await expect(search).toHaveValue('時間');
	await search.fill('情報');
	await page.getByRole('heading', { name: 'Natsume Simple' }).focus();
	await page.waitForTimeout(350);
	await expect(search).toHaveAttribute('aria-expanded', 'false');

	await page.getByLabel('Search direction').focus();
	await expect(search).toHaveAttribute('aria-expanded', 'true');
	const suggestion = page.locator('[role="option"] button').first();
	const label = await suggestion.textContent();
	await suggestion.click();
	await expect(search).toHaveValue(label?.trim() ?? '');
	await expect(search).toHaveAttribute('aria-expanded', 'false');
	await page.waitForTimeout(350);
	await expect(search).toHaveAttribute('aria-expanded', 'false');

	await search.fill('情');
	await page.waitForTimeout(350);
	await expect(search).toHaveAttribute('aria-expanded', 'true');
	await search.fill('情報化');
	await search.press('Escape');
	await page.waitForTimeout(350);
	await expect(search).toHaveAttribute('aria-expanded', 'false');

	await search.fill('情');
	await page.waitForTimeout(350);
	await expect(search).toHaveAttribute('aria-expanded', 'true');
	await page.getByRole('heading', { name: 'Natsume Simple' }).focus();
	await expect(search).toHaveAttribute('aria-expanded', 'false');
});

test('keeps autocomplete dismissed when submission invalidates a pending lookup', async ({
	page
}) => {
	let releaseLookup = () => {};
	const lookupReleased = new Promise<void>((resolve) => (releaseLookup = resolve));
	let markLookupStarted = () => {};
	const lookupStarted = new Promise<void>((resolve) => (markLookupStarted = resolve));
	await page.route('**/api/suggestions**', async (route) => {
		const query = new URL(route.request().url()).searchParams.get('q');
		if (query !== '情') {
			await route.continue();
			return;
		}
		markLookupStarted();
		await lookupReleased;
		await route.continue();
	});

	await page.goto('/');
	const search = page.getByRole('combobox', { name: 'Search term' });
	await search.fill('情');
	await lookupStarted;
	await page.getByRole('button', { name: 'Update results' }).click();
	releaseLookup();
	await page.waitForResponse((response) => {
		const url = new URL(response.url());
		return url.pathname === '/api/suggestions' && url.searchParams.get('q') === '情';
	});
	await expect(search).toHaveAttribute('aria-expanded', 'false');
});

test('keeps autocomplete dismissed when a pending lookup completes', async ({ page }) => {
	let releaseLookup = () => {};
	const lookupReleased = new Promise<void>((resolve) => (releaseLookup = resolve));
	let markLookupStarted = () => {};
	const lookupStarted = new Promise<void>((resolve) => (markLookupStarted = resolve));
	await page.route('**/api/suggestions**', async (route) => {
		const query = new URL(route.request().url()).searchParams.get('q');
		if (query !== '情') {
			await route.continue();
			return;
		}
		markLookupStarted();
		await lookupReleased;
		await route.continue();
	});

	await page.goto('/');
	const search = page.getByRole('combobox', { name: 'Search term' });
	await search.fill('情');
	await lookupStarted;
	await search.press('Escape');
	releaseLookup();
	await page.waitForResponse((response) => {
		const url = new URL(response.url());
		return url.pathname === '/api/suggestions' && url.searchParams.get('q') === '情';
	});
	await expect(search).toHaveAttribute('aria-expanded', 'false');
});

test('renders a keyboard-scrollable particle spreadsheet', async ({ page }) => {
	await expectSpreadsheet(page, 375);
	await expectSpreadsheet(page, 1280);
	await expectSpreadsheet(page, 3200);
});

test('reuses the primary controls on a mobile viewport', async ({ page }) => {
	await page.setViewportSize({ width: 390, height: 844 });
	await page.goto('/');

	await expect(page.getByLabel('Search direction')).toBeVisible();
	await expect(page.getByRole('combobox', { name: 'Search term' })).toBeVisible();
	await expect(page.getByLabel('Bar scale')).toBeVisible();
	await expect(page.getByRole('button', { name: 'Toggle dark mode' })).toBeVisible();
});

test('shows corpus attribution, license, and contact information', async ({ page }) => {
	await page.goto('/');

	const footer = page.getByRole('contentinfo');
	await expect(footer.getByText('Japanese Wikipedia')).toBeVisible();
	await expect(footer.getByText('Journal of Natural Language Processing')).toBeVisible();
	await expect(footer.getByText('TED Talks')).toBeVisible();
	await expect(footer.getByText('no license grant is asserted', { exact: false })).toHaveCount(0);
	await expect(
		footer.getByRole('link', { name: 'IWSLT 2017 Japanese–English dataset' })
	).toHaveAttribute(
		'href',
		'https://huggingface.co/datasets/IWSLT/iwslt2017/tree/c18a4f81a47ae6fa079fe9d32db288ddde38451d/data/2017-01-trnted/texts/ja/en'
	);
	await expect(footer.getByRole('link', { name: 'CC BY-SA 4.0' })).toHaveAttribute(
		'href',
		'https://creativecommons.org/licenses/by-sa/4.0/'
	);
	await expect(footer.getByRole('link', { name: 'Contact' })).toHaveAttribute(
		'href',
		'mailto:dev@bor.space'
	);
});
