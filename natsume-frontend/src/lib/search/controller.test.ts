import { describe, expect, it } from 'vitest';
import type { CollocationsResponse, CorporaResponse } from '$lib/api/types';
import {
	collocationPageSize,
	responseMatchesResult,
	SearchController,
	type SearchApi
} from './controller.svelte';

function deferred<T>() {
	let resolve!: (value: T) => void;
	let reject!: (reason: unknown) => void;
	const promise = new Promise<T>((yes, no) => {
		resolve = yes;
		reject = no;
	});
	return { promise, resolve, reject };
}

function response(
	databaseBuildId: string,
	selectedCorpusIds: string[] = ['alpha']
): CollocationsResponse {
	return {
		particleGroups: [],
		selectedCorpusIds,
		databaseBuildId
	};
}

describe('SearchController', () => {
	it('derives a response-safe collocation page size', () => {
		expect(collocationPageSize(1)).toBe(200);
		expect(collocationPageSize(2)).toBe(200);
		expect(collocationPageSize(3)).toBe(150);
	});

	it('matches responses only to the submitted build and canonical corpora', () => {
		const candidate = {
			databaseBuildId: 'build-1',
			selectedCorpusIds: ['alpha', 'beta']
		};

		expect(responseMatchesResult(candidate, 'build-1', ['alpha', 'beta'])).toBe(true);
		expect(responseMatchesResult(candidate, 'build-2', ['alpha', 'beta'])).toBe(false);
		expect(responseMatchesResult(candidate, 'build-1', ['alpha'])).toBe(false);
		expect(responseMatchesResult(candidate, 'build-1', ['beta', 'alpha'])).toBe(false);
	});

	it('submits the selection-dependent page size', async () => {
		const limits: Array<number | undefined> = [];
		const api: SearchApi = {
			getCorpora: async () => ({ corpora: [], databaseBuildId: 'x' }),
			getCollocations: async (args) => {
				limits.push(args.limitPerParticle);
				return response('submitted', args.corpusIds);
			}
		};
		const controller = new SearchController(api);
		controller.selectedCorpusIds = ['alpha', 'beta', 'ted'];
		await controller.submit();
		controller.selectedCorpusIds = ['alpha', 'beta'];
		await controller.submit();

		expect(limits).toEqual([150, 200]);
	});

	it('allows only the latest request to replace visible results', async () => {
		const first = deferred<CollocationsResponse>();
		const second = deferred<CollocationsResponse>();
		let call = 0;
		const api: SearchApi = {
			getCorpora: async (): Promise<CorporaResponse> => ({ corpora: [], databaseBuildId: 'x' }),
			getCollocations: async () => (++call === 1 ? first.promise : second.promise)
		};
		const controller = new SearchController(api);
		controller.selectedCorpusIds = ['alpha'];

		const firstRequest = controller.submit();
		controller.term = '研究';
		const secondRequest = controller.submit();
		second.resolve(response('second'));
		await secondRequest;
		first.resolve(response('first'));
		await firstRequest;

		expect(controller.result?.response.databaseBuildId).toBe('second');
		expect(controller.status).toBe('empty');
	});

	it('retains the last success as stale when the latest request fails', async () => {
		let fail = false;
		const api: SearchApi = {
			getCorpora: async () => ({ corpora: [], databaseBuildId: 'x' }),
			getCollocations: async () => {
				if (fail) throw new Error('offline');
				return response('success');
			}
		};
		const controller = new SearchController(api);
		controller.selectedCorpusIds = ['alpha'];
		await controller.submit();
		fail = true;
		controller.term = '失敗';

		await controller.submit();

		expect(controller.result?.response.databaseBuildId).toBe('success');
		expect(controller.status).toBe('error');
		expect(controller.resultIsStale).toBe(true);
	});

	it('keeps visible results tied to the submitted term and direction', async () => {
		const api: SearchApi = {
			getCorpora: async () => ({ corpora: [], databaseBuildId: 'x' }),
			getCollocations: async () => response('submitted')
		};
		const controller = new SearchController(api);
		controller.term = '情報';
		controller.pos = 'noun';
		controller.selectedCorpusIds = ['alpha'];

		await controller.submit();
		controller.term = '集める';
		controller.pos = 'verb';

		expect(controller.result?.input).toEqual({
			term: '情報',
			pos: 'noun',
			corpusIds: ['alpha']
		});
		expect(controller.draftDiffersFromResult).toBe(true);
		expect(controller.resultIsStale).toBe(true);
	});
});
