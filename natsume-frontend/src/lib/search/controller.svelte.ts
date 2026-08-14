import type { CollocationsResponse, CorporaResponse, Corpus, SearchPosition } from '$lib/api/types';

export type SearchApi = {
	getCorpora(signal?: AbortSignal): Promise<CorporaResponse>;
	getCollocations(
		args: {
			term: string;
			pos: SearchPosition;
			corpusIds: string[];
		},
		signal?: AbortSignal
	): Promise<CollocationsResponse>;
};

export type SearchStatus = 'idle' | 'loading' | 'success' | 'empty' | 'error';

export type SearchInput = Readonly<{
	term: string;
	pos: SearchPosition;
	corpusIds: readonly string[];
}>;

export type VisibleSearchResult = Readonly<{
	response: CollocationsResponse;
	input: SearchInput;
}>;

export class SearchController {
	term = $state('時間');
	pos = $state<SearchPosition>('noun');
	corpora = $state<Corpus[]>([]);
	selectedCorpusIds = $state<string[]>([]);
	status = $state<SearchStatus>('idle');
	result = $state<VisibleSearchResult | null>(null);
	errorMessage = $state<string | null>(null);

	get draftDiffersFromResult(): boolean {
		if (!this.result) return false;
		return (
			this.term.trim() !== this.result.input.term ||
			this.pos !== this.result.input.pos ||
			!sameValues(this.selectedCorpusIds, this.result.input.corpusIds)
		);
	}

	get resultIsStale(): boolean {
		return (
			this.result !== null &&
			(this.draftDiffersFromResult || this.status === 'loading' || this.status === 'error')
		);
	}

	private generation = 0;
	private request: AbortController | null = null;

	constructor(private readonly api: SearchApi) {}

	async initialize(): Promise<void> {
		try {
			const response = await this.api.getCorpora();
			this.corpora = response.corpora;
			this.selectedCorpusIds = response.corpora.map((corpus) => corpus.id);
			await this.submit();
		} catch (error) {
			this.status = 'error';
			this.errorMessage = errorMessage(error);
		}
	}

	async submit(): Promise<void> {
		if (!this.term.trim() || this.selectedCorpusIds.length === 0) return;
		const input: SearchInput = {
			term: this.term.trim(),
			pos: this.pos,
			corpusIds: [...this.selectedCorpusIds]
		};
		this.request?.abort();
		const request = new AbortController();
		this.request = request;
		const generation = ++this.generation;
		this.status = 'loading';
		this.errorMessage = null;

		try {
			const result = await this.api.getCollocations(
				{
					term: input.term,
					pos: input.pos,
					corpusIds: [...input.corpusIds]
				},
				request.signal
			);
			if (generation !== this.generation) return;
			this.result = {
				response: result,
				input: { ...input, corpusIds: [...result.selectedCorpusIds] }
			};
			this.selectedCorpusIds = result.selectedCorpusIds;
			this.status = result.particleGroups.some((group) => group.items.length > 0)
				? 'success'
				: 'empty';
		} catch (error) {
			if (generation !== this.generation || isAbort(error)) return;
			this.status = 'error';
			this.errorMessage = errorMessage(error);
		}
	}

	async toggleCorpus(corpusId: string): Promise<void> {
		if (this.selectedCorpusIds.includes(corpusId)) {
			if (this.selectedCorpusIds.length === 1) return;
			this.selectedCorpusIds = this.selectedCorpusIds.filter((id) => id !== corpusId);
		} else {
			this.selectedCorpusIds = this.corpora
				.map((corpus) => corpus.id)
				.filter((id) => id === corpusId || this.selectedCorpusIds.includes(id));
		}
		await this.submit();
	}
}

function sameValues(left: readonly string[], right: readonly string[]): boolean {
	return left.length === right.length && left.every((value, index) => value === right[index]);
}

function isAbort(error: unknown): boolean {
	return error instanceof DOMException && error.name === 'AbortError';
}

function errorMessage(error: unknown): string {
	return error instanceof Error ? error.message : 'The service request failed';
}
