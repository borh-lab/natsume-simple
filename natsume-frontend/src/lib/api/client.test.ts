import { describe, expect, it, vi } from 'vitest';
import { ApiClient, ApiClientError } from './client';

describe('ApiClient', () => {
	it('sends canonical corpus selection and ranking parameters', async () => {
		const requests: string[] = [];
		const fetcher = vi.fn(async (input: RequestInfo | URL) => {
			requests.push(String(input));
			return new Response(
				JSON.stringify({
					particleGroups: [],
					selectedCorpusIds: ['alpha', 'beta'],
					rankBy: 'raw',
					databaseBuildId: 'fixture'
				}),
				{ headers: { 'content-type': 'application/json' } }
			);
		});
		const client = new ApiClient('https://example.test', fetcher);

		await client.getCollocations({
			term: '情報',
			pos: 'noun',
			corpusIds: ['alpha', 'beta'],
			rankBy: 'raw'
		});

		const url = new URL(requests[0]);
		expect(url.pathname).toBe('/api/collocations');
		expect(url.searchParams.get('term')).toBe('情報');
		expect(url.searchParams.getAll('corpusId')).toEqual(['alpha', 'beta']);
		expect(url.searchParams.get('rankBy')).toBe('raw');
	});

	it('turns the public error envelope into a stable client error', async () => {
		const fetcher = vi.fn(
			async () =>
				new Response(
					JSON.stringify({
						error: {
							code: 'database_unavailable',
							message: 'The database artifact is unavailable',
							requestId: 'request-1'
						}
					}),
					{ status: 503, headers: { 'content-type': 'application/json' } }
				)
		);

		await expect(new ApiClient('', fetcher).getCorpora()).rejects.toEqual(
			new ApiClientError(
				'database_unavailable',
				'The database artifact is unavailable',
				'request-1',
				503
			)
		);
	});
});
