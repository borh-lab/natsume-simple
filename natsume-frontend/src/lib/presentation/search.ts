import type { CorpusContribution, RankBy } from '$lib/api/types';

export type StackSegment = { corpusId: string; percentage: number; offset: number };

export function stackSegments(
	contributions: readonly CorpusContribution[],
	selectedCorpusIds: readonly string[],
	corpusCounts: Readonly<Record<string, number>> = {},
	rankBy: RankBy = 'raw'
): StackSegment[] {
	const contributionByCorpus = new Map(contributions.map((item) => [item.corpusId, item]));
	const values = selectedCorpusIds.map((corpusId) => {
		const raw = contributionByCorpus.get(corpusId)?.rawFrequency ?? 0;
		return {
			corpusId,
			value:
				rankBy === 'meanPerMillion' && corpusCounts[corpusId]
					? (raw / corpusCounts[corpusId]) * 1_000_000
					: raw
		};
	});
	const total = values.reduce((sum, item) => sum + item.value, 0);
	let offset = 0;
	return values
		.filter((item) => item.value > 0)
		.map((item) => {
			const percentage = total > 0 ? (item.value / total) * 100 : 0;
			const segment = { corpusId: item.corpusId, percentage, offset };
			offset += percentage;
			return segment;
		});
}

export function corpusColorMap(corpora: readonly { id: string }[]): Record<string, number> {
	return Object.fromEntries(corpora.map((corpus, index) => [corpus.id, index]));
}
