export const ROLE_TEXT_CLASSES = {
	noun: 'font-bold text-blue-600 dark:text-blue-400',
	particle: 'font-bold text-red-600 dark:text-red-400',
	verb: 'font-bold text-green-600 dark:text-green-400'
} as const;

export const CORPUS_STYLES = [
	{ color: '#7c3aed', titleClass: 'text-violet-700 dark:text-violet-300' },
	{ color: '#ea580c', titleClass: 'text-orange-700 dark:text-orange-300' },
	{ color: '#0891b2', titleClass: 'text-cyan-700 dark:text-cyan-300' }
] as const;

const FALLBACK_CORPUS_STYLE = {
	color: '#6b7280',
	titleClass: 'text-gray-700 dark:text-gray-300'
} as const;

export function corpusStyleForSlot(slot: number) {
	return CORPUS_STYLES[slot] ?? FALLBACK_CORPUS_STYLE;
}
