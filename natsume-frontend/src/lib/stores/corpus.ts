import { writable, derived } from 'svelte/store';
import type { Collocate } from '$lib/query';

export type CorpusStats = {
	normalizationFactor: number;
	collocationCount: number;
};

export const corpusNorm = writable<Record<string, CorpusStats>>({});
export const selectedCorpora = writable<string[]>([]);
export const useNormalization = writable(true);
export const searchElapsedTime = writable(0);
export const resultCount = writable(0);
export type Frequency = {
	normalized: number;
	raw: number;
};

export const particleGroups = writable<
	Record<
		string,
		{
			collocates: Collocate[];
			maxFrequency: {
				normalized: number;
				raw: number;
			};
			distribution: Record<string, Frequency>;
		}
	>
>({});

export const filteredResultCount = derived(
	[particleGroups, selectedCorpora],
	([$particleGroups, $selectedCorpora]) => {
		return Object.values($particleGroups).reduce((total, group) => {
			if (!group.collocates) return total;
			return (
				total +
				group.collocates.filter((collocate) =>
					collocate.contributions.some((c) => $selectedCorpora.includes(c.corpus))
				).length
			);
		}, 0);
	}
);
