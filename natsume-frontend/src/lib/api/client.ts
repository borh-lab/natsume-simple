import type {
	CollocationsResponse,
	CorporaResponse,
	ExamplesResponse,
	PublicErrorEnvelope,
	SearchPosition,
	SuggestionsResponse
} from './types';

type Fetcher = (input: RequestInfo | URL, init?: RequestInit) => Promise<Response>;

export class ApiClientError extends Error {
	constructor(
		public readonly code: string,
		message: string,
		public readonly requestId: string | null,
		public readonly status: number
	) {
		super(message);
		this.name = 'ApiClientError';
	}
}

export class ApiClient {
	private readonly baseUrl: string;

	constructor(
		baseUrl = '',
		private readonly fetcher: Fetcher = fetch
	) {
		this.baseUrl = baseUrl.replace(/\/$/, '');
	}

	getCorpora(signal?: AbortSignal): Promise<CorporaResponse> {
		return this.request('/api/corpora', new URLSearchParams(), signal);
	}

	getSuggestions(
		query: string,
		position: SearchPosition,
		signal?: AbortSignal
	): Promise<SuggestionsResponse> {
		return this.request(
			'/api/suggestions',
			new URLSearchParams({ q: query, pos: position }),
			signal
		);
	}

	getCollocations(
		args: {
			term: string;
			pos: SearchPosition;
			corpusIds: string[];
			limitPerParticle?: number;
		},
		signal?: AbortSignal
	): Promise<CollocationsResponse> {
		const params = new URLSearchParams({
			term: args.term,
			pos: args.pos,
			limitPerParticle: String(args.limitPerParticle ?? 150)
		});
		for (const corpusId of args.corpusIds) params.append('corpusId', corpusId);
		return this.request('/api/collocations', params, signal);
	}

	getExamples(
		args: {
			noun: string;
			particle: string;
			verb: string;
			corpusIds: string[];
			limit?: number;
		},
		signal?: AbortSignal
	): Promise<ExamplesResponse> {
		const params = new URLSearchParams({
			noun: args.noun,
			particle: args.particle,
			verb: args.verb,
			limit: String(args.limit ?? 5)
		});
		for (const corpusId of args.corpusIds) params.append('corpusId', corpusId);
		return this.request('/api/examples', params, signal);
	}

	private async request<T>(
		path: string,
		params: URLSearchParams,
		signal?: AbortSignal
	): Promise<T> {
		const query = params.size ? `?${params}` : '';
		const response = await this.fetcher(`${this.baseUrl}${path}${query}`, { signal });
		if (!response.headers.get('content-type')?.includes('application/json')) {
			throw new ApiClientError(
				'unexpected_response',
				'The service returned an invalid response',
				null,
				response.status
			);
		}
		const payload: unknown = await response.json();
		if (!response.ok) {
			if (isPublicErrorEnvelope(payload)) {
				throw new ApiClientError(
					payload.error.code,
					payload.error.message,
					payload.error.requestId,
					response.status
				);
			}
			throw new ApiClientError(
				'unexpected_response',
				'The service request failed',
				null,
				response.status
			);
		}
		return payload as T;
	}
}

function isPublicErrorEnvelope(value: unknown): value is PublicErrorEnvelope {
	if (!value || typeof value !== 'object' || !('error' in value)) return false;
	const error = value.error;
	return (
		!!error &&
		typeof error === 'object' &&
		'code' in error &&
		typeof error.code === 'string' &&
		'message' in error &&
		typeof error.message === 'string' &&
		'requestId' in error &&
		typeof error.requestId === 'string'
	);
}
