import type { CollocationItem, ParticleGroup } from '$lib/api/types';

export type BarScale = 'particle' | 'global';

export type BarSegment = {
	corpusId: string;
	rawFrequency: number;
	frequencyPerMillion: number;
	percentage: number;
	offset: number;
};

export type ParticleMass = {
	percentage: number;
	segments: BarSegment[];
};

export function barReference(
	groups: readonly ParticleGroup[],
	group: ParticleGroup,
	scale: BarScale
): number {
	const items = scale === 'global' ? groups.flatMap((candidate) => candidate.items) : group.items;
	return Math.max(0, ...items.map((item) => item.meanFrequencyPerMillion));
}

export function itemBarSegments(
	item: CollocationItem,
	selectedCorpusIds: readonly string[],
	corpusCounts: Readonly<Record<string, number>>,
	referenceFrequencyPerMillion: number
): BarSegment[] {
	if (referenceFrequencyPerMillion <= 0 || item.meanFrequencyPerMillion <= 0) return [];

	const contributionByCorpus = new Map(
		item.contributions.map((contribution) => [contribution.corpusId, contribution])
	);
	const contributions = selectedCorpusIds.flatMap((corpusId) => {
		const contribution = contributionByCorpus.get(corpusId);
		const corpusCount = corpusCounts[corpusId];
		if (!contribution || !corpusCount) return [];
		return [
			{
				corpusId,
				rawFrequency: contribution.rawFrequency,
				frequencyPerMillion: (contribution.rawFrequency / corpusCount) * 1_000_000
			}
		];
	});
	const contributionTotal = contributions.reduce(
		(sum, contribution) => sum + contribution.frequencyPerMillion,
		0
	);
	if (contributionTotal <= 0) return [];

	const totalPercentage = (item.meanFrequencyPerMillion / referenceFrequencyPerMillion) * 100;
	let offset = 0;
	return contributions.map((contribution) => {
		const percentage = (contribution.frequencyPerMillion / contributionTotal) * totalPercentage;
		const segment = { ...contribution, percentage, offset };
		offset += percentage;
		return segment;
	});
}

export function particleMassSegments(
	group: ParticleGroup,
	groups: readonly ParticleGroup[]
): ParticleMass {
	const totalMass = groups.reduce(
		(sum, candidate) =>
			sum +
			candidate.corpusDistribution.reduce(
				(distributionSum, contribution) => distributionSum + contribution.frequencyPerMillion,
				0
			),
		0
	);
	if (totalMass <= 0) return { percentage: 0, segments: [] };

	let offset = 0;
	const segments = group.corpusDistribution
		.filter((contribution) => contribution.frequencyPerMillion > 0)
		.map((contribution) => {
			const percentage = (contribution.frequencyPerMillion / totalMass) * 100;
			const segment = { ...contribution, percentage, offset };
			offset += percentage;
			return segment;
		});
	return { percentage: segments.reduce((sum, segment) => sum + segment.percentage, 0), segments };
}

export function corpusColorMap(corpora: readonly { id: string }[]): Record<string, number> {
	return Object.fromEntries(corpora.map((corpus, index) => [corpus.id, index]));
}
