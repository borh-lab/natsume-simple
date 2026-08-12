import { expect, test } from '@playwright/test';

test('searches the fixture backend and safely expands an example', async ({ page }) => {
	await page.goto('/');

	const search = page.locator('#search-input');
	await search.fill('情報');
	await search.press('Enter');

	await expect(page.getByRole('heading', { name: 'を', exact: true })).toBeVisible();
	const collocation = page.locator('summary').filter({ hasText: '集める' });
	await expect(collocation).toBeVisible();
	await collocation.click();

	await expect(page.getByText('情報を集める。', { exact: false }).first()).toBeVisible();
	await expect(page.locator('img[src="x"]')).toHaveCount(0);
});

test('preserves direction, particle order, options, and theme controls', async ({ page }) => {
	await page.goto('/');

	const direction = page.locator('header').first().locator('select').filter({ visible: true });
	await direction.selectOption('verb');
	const search = page.locator('#search-input');
	await search.fill('集める');
	await Promise.all([
		page.waitForResponse((response) => response.url().includes('/npv/verb/')),
		search.press('Enter')
	]);
	await expect(page.locator('summary').filter({ hasText: '情報' })).toBeVisible();

	await direction.selectOption('noun');
	await search.fill('情報');
	await Promise.all([
		page.waitForResponse((response) => response.url().includes('/npv/noun/')),
		search.press('Enter')
	]);
	await expect(page.locator('summary').filter({ hasText: '集める' })).toBeVisible();
	expect(await page.locator('header:nth-of-type(2) h2').allTextContents()).toEqual(['が', 'を']);

	await page.locator('#options-button').click();
	const normalization = page.locator('#use-normalization');
	await expect(normalization).toBeChecked();
	await normalization.uncheck();
	await expect(normalization).not.toBeChecked();
	await page.locator('#corpus-alpha').uncheck();
	await expect(page.locator('summary').filter({ hasText: '集める' })).toBeVisible();

	const theme = page.locator('header').first().locator('button').last();
	await theme.click();
	await expect(page.locator('html')).toHaveClass(/dark/);
});

test('offers search, options, and theme controls on a mobile viewport', async ({ page }) => {
	await page.setViewportSize({ width: 390, height: 844 });
	await page.goto('/');

	const search = page.getByPlaceholder('Search term');
	await search.fill('情報');
	await page.getByRole('button', { name: 'Go' }).click();
	await expect(page.locator('summary').filter({ hasText: '集める' })).toBeVisible();

	await page
		.locator('header')
		.first()
		.locator('button')
		.filter({ has: page.locator('div') })
		.click();
	await expect(page.getByRole('button', { name: /Select Search Type/ })).toBeVisible();
	await expect(page.getByRole('button', { name: /Options/ })).toBeVisible();
	await expect(page.getByText('Dark Mode', { exact: true })).toBeVisible();
});
