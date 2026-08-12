import { describe, expect, it } from 'vitest';
import { corpusColorMap, stackSegments } from './search';

describe('presentation search math', () => {
	it('builds a non-mutating percentage stack in selected corpus order', () => {
		const contributions = [
			{ corpusId: 'beta', rawFrequency: 1 },
			{ corpusId: 'alpha', rawFrequency: 3 }
		];

		const result = stackSegments(contributions, ['alpha', 'beta']);

		expect(result).toEqual([
			{ corpusId: 'alpha', percentage: 75, offset: 0 },
			{ corpusId: 'beta', percentage: 25, offset: 75 }
		]);
		expect(contributions.map(({ corpusId }) => corpusId)).toEqual(['beta', 'alpha']);
	});

	it('assigns stable color slots in metadata order', () => {
		expect(corpusColorMap([{ id: 'beta' }, { id: 'alpha' }])).toEqual({ beta: 0, alpha: 1 });
	});
});
