import { describe, expect, it } from 'vitest';
import type { CollocationItem, Particle, ParticleGroup } from '$lib/api/types';
import { barReference, corpusColorMap, itemBarSegments, particleMassSegments } from './search';

const alphaItem: CollocationItem = {
	noun: '情報',
	particle: 'を',
	verb: '集める',
	totalRawFrequency: 30,
	meanFrequencyPerMillion: 150_000,
	contributions: [
		{ corpusId: 'beta', rawFrequency: 10 },
		{ corpusId: 'alpha', rawFrequency: 20 }
	]
};

function group(
	particle: Particle,
	items: CollocationItem[],
	distribution: ParticleGroup['corpusDistribution']
): ParticleGroup {
	return {
		particle,
		totalMatchingCollocations: items.length,
		returnedCount: items.length,
		items,
		corpusDistribution: distribution
	};
}

describe('presentation search math', () => {
	it('uses the server score for total bar width and normalized corpus rates for its segments', () => {
		const contributions = [...alphaItem.contributions];

		const result = itemBarSegments(
			alphaItem,
			['alpha', 'beta'],
			{ alpha: 100, beta: 100 },
			300_000
		);

		expect(result).toMatchObject([
			{
				corpusId: 'alpha',
				rawFrequency: 20,
				frequencyPerMillion: 200_000
			},
			{
				corpusId: 'beta',
				rawFrequency: 10,
				frequencyPerMillion: 100_000
			}
		]);
		expect(result[0].percentage).toBeCloseTo(100 / 3);
		expect(result[0].offset).toBe(0);
		expect(result[1].percentage).toBeCloseTo(50 / 3);
		expect(result[1].offset).toBeCloseTo(100 / 3);
		expect(result.reduce((sum, segment) => sum + segment.percentage, 0)).toBeCloseTo(50);
		expect(contributions).toEqual(alphaItem.contributions);
	});

	it('chooses a per-particle or response-wide reference without changing item order', () => {
		const small = { ...alphaItem, verb: '調べる', meanFrequencyPerMillion: 40_000 };
		const other: CollocationItem = {
			...alphaItem,
			particle: 'が',
			verb: '進める',
			meanFrequencyPerMillion: 500_000
		};
		const groups = [group('を', [alphaItem, small], []), group('が', [other], [])];

		expect(barReference(groups, groups[0], 'particle')).toBe(150_000);
		expect(barReference(groups, groups[0], 'global')).toBe(500_000);
		expect(groups[0].items.map((item) => item.verb)).toEqual(['集める', '調べる']);
	});

	it('expresses complete particle mass as a share of all particle mass', () => {
		const groups = [
			group(
				'を',
				[],
				[
					{ corpusId: 'alpha', rawFrequency: 6, frequencyPerMillion: 60 },
					{ corpusId: 'beta', rawFrequency: 2, frequencyPerMillion: 20 }
				]
			),
			group('が', [], [{ corpusId: 'alpha', rawFrequency: 2, frequencyPerMillion: 20 }])
		];

		expect(particleMassSegments(groups[0], groups)).toEqual({
			percentage: 80,
			segments: [
				{
					corpusId: 'alpha',
					rawFrequency: 6,
					frequencyPerMillion: 60,
					percentage: 60,
					offset: 0
				},
				{
					corpusId: 'beta',
					rawFrequency: 2,
					frequencyPerMillion: 20,
					percentage: 20,
					offset: 60
				}
			]
		});
	});

	it('assigns stable color slots in metadata order', () => {
		expect(corpusColorMap([{ id: 'beta' }, { id: 'alpha' }])).toEqual({ beta: 0, alpha: 1 });
	});
});
