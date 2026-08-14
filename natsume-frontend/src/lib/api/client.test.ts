import { describe, expect, it, vi } from 'vitest';
import { ApiClient, ApiClientError } from './client';

describe('ApiClient', () => {
	it('sends canonical corpus selection without a display-only ranking parameter', async () => {
		const requests: string[] = [];
		const fetcher = vi.fn(async (input: RequestInfo | URL) => {
			requests.push(String(input));
			return new Response(
				JSON.stringify({
					particleGroups: [],
					selectedCorpusIds: ['alpha', 'beta'],
					databaseBuildId: 'fixture'
				}),
				{ headers: { 'content-type': 'application/json' } }
			);
		});
		const client = new ApiClient('https://example.test', fetcher);

		await client.getCollocations({
			term: '情報',
			pos: 'noun',
			corpusIds: ['alpha', 'beta']
		});

		const url = new URL(requests[0]);
		expect(url.pathname).toBe('/api/collocations');
		expect(url.searchParams.get('term')).toBe('情報');
		expect(url.searchParams.getAll('corpusId')).toEqual(['alpha', 'beta']);
		expect(url.searchParams.has('rankBy')).toBe(false);
	});

	it('serializes targeted collocation and example offsets', async () => {
		const requests: string[] = [];
		const fetcher = vi.fn(async (input: RequestInfo | URL) => {
			requests.push(String(input));
			return new Response(
				JSON.stringify({
					particleGroups: [],
					examples: [],
					hasMore: false,
					selectedCorpusIds: ['alpha', 'beta', 'ted'],
					databaseBuildId: 'fixture'
				}),
				{ headers: { 'content-type': 'application/json' } }
			);
		});
		const client = new ApiClient('https://example.test', fetcher);

		await client.getCollocations({
			term: '情報',
			pos: 'noun',
			corpusIds: ['alpha', 'beta', 'ted'],
			particle: 'を',
			offsetPerParticle: 150,
			limitPerParticle: 150
		});
		await client.getExamples({
			noun: '情報',
			particle: 'を',
			verb: '集める',
			corpusIds: ['alpha', 'beta', 'ted'],
			offset: 5,
			limit: 20
		});

		const collocations = new URL(requests[0]).searchParams;
		expect(collocations.get('particle')).toBe('を');
		expect(collocations.get('offsetPerParticle')).toBe('150');
		expect(collocations.get('limitPerParticle')).toBe('150');
		const examples = new URL(requests[1]).searchParams;
		expect(examples.get('offset')).toBe('5');
		expect(examples.get('limit')).toBe('20');
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
