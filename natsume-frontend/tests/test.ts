import { expect, test } from '@playwright/test';

test('searches, filters, reranks, and safely expands examples', async ({ page }) => {
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

	const collocation = page.locator('summary').filter({ hasText: '集める' });
	await collocation.click();
	await expect(page.getByText('情報を集める。', { exact: false }).first()).toBeVisible();
	await expect(page.locator('img[src="x"]')).toHaveCount(0);

	await page.locator('#corpus-alpha').uncheck();
	await expect(page.locator('#corpus-alpha')).not.toBeChecked();
	await expect(page.locator('summary').filter({ hasText: '集める' })).toBeVisible();

	await page.getByLabel('Rank').selectOption('raw');
	await expect(page.getByLabel('Rank')).toHaveValue('raw');
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

	await search.focus();
	await expect(search).toHaveAttribute('aria-expanded', 'true');
	const suggestion = page.locator('[role="option"] button').first();
	const label = await suggestion.textContent();
	await suggestion.click();
	await expect(search).toHaveValue(label?.trim() ?? '');
	await expect(search).toHaveAttribute('aria-expanded', 'false');

	await search.fill('情');
	await page.waitForTimeout(350);
	await expect(search).toHaveAttribute('aria-expanded', 'true');
	await page.getByRole('heading', { name: 'Natsume Simple' }).focus();
	await expect(search).toHaveAttribute('aria-expanded', 'false');
});

test('reuses the primary controls on a mobile viewport', async ({ page }) => {
	await page.setViewportSize({ width: 390, height: 844 });
	await page.goto('/');

	await expect(page.getByLabel('Search direction')).toBeVisible();
	await expect(page.getByRole('combobox', { name: 'Search term' })).toBeVisible();
	await expect(page.getByLabel('Rank')).toBeVisible();
	await expect(page.getByRole('button', { name: 'Toggle dark mode' })).toBeVisible();
});

test('shows corpus attribution, license, and contact information', async ({ page }) => {
	await page.goto('/');

	const footer = page.getByRole('contentinfo');
	await expect(footer.getByText('Japanese Wikipedia')).toBeVisible();
	await expect(footer.getByText('Journal of Natural Language Processing')).toBeVisible();
	await expect(footer.getByRole('link', { name: 'CC BY-SA 4.0' })).toHaveAttribute(
		'href',
		'https://creativecommons.org/licenses/by-sa/4.0/'
	);
	await expect(footer.getByRole('link', { name: 'Contact' })).toHaveAttribute(
		'href',
		'mailto:dev@bor.space'
	);
});
