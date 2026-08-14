import { describe, expect, it } from 'vitest';
import { CORPUS_STYLES, ROLE_TEXT_CLASSES, corpusStyleForSlot } from './colors';

describe('semantic presentation colors', () => {
	it('pins grammatical roles and keeps corpus slots disjoint', () => {
		expect(ROLE_TEXT_CLASSES).toEqual({
			noun: 'font-bold text-blue-600 dark:text-blue-400',
			particle: 'font-bold text-red-600 dark:text-red-400',
			verb: 'font-bold text-green-600 dark:text-green-400'
		});
		expect(CORPUS_STYLES).toEqual([
			{ color: '#7c3aed', titleClass: 'text-violet-700 dark:text-violet-300' },
			{ color: '#ea580c', titleClass: 'text-orange-700 dark:text-orange-300' },
			{ color: '#0891b2', titleClass: 'text-cyan-700 dark:text-cyan-300' }
		]);
		const roleHex = new Set(['#2563eb', '#dc2626', '#16a34a']);
		expect(CORPUS_STYLES.every(({ color }) => !roleHex.has(color))).toBe(true);
	});

	it('uses a neutral fourth slot instead of aliasing a valid corpus', () => {
		const fallback = corpusStyleForSlot(3);
		expect(fallback).toEqual({
			color: '#6b7280',
			titleClass: 'text-gray-700 dark:text-gray-300'
		});
		expect(CORPUS_STYLES.map(({ color }) => color)).not.toContain(fallback.color);
	});
});
