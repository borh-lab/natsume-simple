import { describe, expect, it } from 'vitest';

import { sentenceSegments } from './sentence';

describe('sentenceSegments', () => {
	it('keeps HTML-shaped corpus content as plain text', () => {
		const text = '<img src=x onerror=alert(1)>を読む';
		const segments = sentenceSegments(text, [
			{ start: 0, end: 28, type: 'noun' },
			{ start: 28, end: 29, type: 'particle' },
			{ start: 29, end: 31, type: 'verb' }
		]);

		expect(segments.map(({ text }) => text).join('')).toBe(text);
		expect(segments[0].text).toBe('<img src=x onerror=alert(1)>');
	});

	it('ignores invalid and overlapping spans', () => {
		expect(
			sentenceSegments('情報を得る', [
				{ start: -1, end: 2, type: 'noun' },
				{ start: 0, end: 2, type: 'noun' },
				{ start: 1, end: 3, type: 'particle' }
			])
		).toEqual([
			{ text: '情報', className: 'font-bold text-blue-600 dark:text-blue-400' },
			{ text: 'を得る' }
		]);
	});
});
