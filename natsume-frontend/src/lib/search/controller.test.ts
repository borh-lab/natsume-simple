import { describe, expect, it } from 'vitest';
import type { CollocationsResponse, CorporaResponse } from '$lib/api/types';
import { SearchController, type SearchApi } from './controller.svelte';

function deferred<T>() {
	let resolve!: (value: T) => void;
	let reject!: (reason: unknown) => void;
	const promise = new Promise<T>((yes, no) => {
		resolve = yes;
		reject = no;
	});
	return { promise, resolve, reject };
}

function response(databaseBuildId: string): CollocationsResponse {
	return {
		particleGroups: [],
		selectedCorpusIds: ['alpha'],
		rankBy: 'raw',
		databaseBuildId
	};
}

describe('SearchController', () => {
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

		expect(controller.result?.databaseBuildId).toBe('second');
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

		expect(controller.result?.databaseBuildId).toBe('success');
		expect(controller.status).toBe('error');
		expect(controller.resultIsStale).toBe(true);
	});
});
