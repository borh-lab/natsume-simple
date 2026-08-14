import { expect, test } from '@playwright/test';

async function expectSpreadsheet(page: import('@playwright/test').Page, width: number) {
	await page.setViewportSize({ width, height: 844 });
	await page.goto('/');
	await page.getByRole('combobox', { name: 'Search term' }).fill('情報');
	await page.getByRole('button', { name: 'Go' }).click();
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
		page.getByRole('button', { name: 'Go' }).click()
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
		page.getByRole('button', { name: 'Go' }).click()
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
	await page.getByRole('button', { name: 'Go' }).click();
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
