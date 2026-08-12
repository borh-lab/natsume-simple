import type {
	CollocationsResponse,
	CorporaResponse,
	Corpus,
	RankBy,
	SearchPosition
} from '$lib/api/types';

export type SearchApi = {
	getCorpora(signal?: AbortSignal): Promise<CorporaResponse>;
	getCollocations(
		args: {
			term: string;
			pos: SearchPosition;
			corpusIds: string[];
			rankBy: RankBy;
		},
		signal?: AbortSignal
	): Promise<CollocationsResponse>;
};

export type SearchStatus = 'idle' | 'loading' | 'success' | 'empty' | 'error';

export class SearchController {
	term = $state('時間');
	pos = $state<SearchPosition>('noun');
	corpora = $state<Corpus[]>([]);
	selectedCorpusIds = $state<string[]>([]);
	rankBy = $state<RankBy>('meanPerMillion');
	status = $state<SearchStatus>('idle');
	result = $state<CollocationsResponse | null>(null);
	errorMessage = $state<string | null>(null);
	resultIsStale = $state(false);

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
		this.request?.abort();
		const request = new AbortController();
		this.request = request;
		const generation = ++this.generation;
		this.status = 'loading';
		this.errorMessage = null;
		this.resultIsStale = this.result !== null;

		try {
			const result = await this.api.getCollocations(
				{
					term: this.term.trim(),
					pos: this.pos,
					corpusIds: this.selectedCorpusIds,
					rankBy: this.rankBy
				},
				request.signal
			);
			if (generation !== this.generation) return;
			this.result = result;
			this.selectedCorpusIds = result.selectedCorpusIds;
			this.rankBy = result.rankBy;
			this.resultIsStale = false;
			this.status = result.particleGroups.some((group) => group.items.length > 0)
				? 'success'
				: 'empty';
		} catch (error) {
			if (generation !== this.generation || isAbort(error)) return;
			this.status = 'error';
			this.errorMessage = errorMessage(error);
			this.resultIsStale = this.result !== null;
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

	async selectRank(rankBy: RankBy): Promise<void> {
		if (rankBy === this.rankBy) return;
		this.rankBy = rankBy;
		await this.submit();
	}
}

function isAbort(error: unknown): boolean {
	return error instanceof DOMException && error.name === 'AbortError';
}

function errorMessage(error: unknown): string {
	return error instanceof Error ? error.message : 'The service request failed';
}
