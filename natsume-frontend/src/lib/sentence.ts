export type SentenceSpan = {
	start: number;
	end: number;
	type: 'noun' | 'particle' | 'verb';
};

export type SentenceSegment = {
	text: string;
	className?: string;
};

const highlightClasses: Record<SentenceSpan['type'], string> = {
	noun: 'font-bold text-blue-600 dark:text-blue-400',
	particle: 'font-bold text-red-600 dark:text-red-400',
	verb: 'font-bold text-green-600 dark:text-green-400'
};

export function sentenceSegments(text: string, spans: SentenceSpan[]): SentenceSegment[] {
	const sortedSpans = [...spans]
		.filter(({ start, end }) => 0 <= start && start < end && end <= text.length)
		.sort((a, b) => a.start - b.start);
	const segments: SentenceSegment[] = [];
	let offset = 0;

	for (const span of sortedSpans) {
		if (span.start < offset) continue;
		if (offset < span.start) segments.push({ text: text.slice(offset, span.start) });
		segments.push({
			text: text.slice(span.start, span.end),
			className: highlightClasses[span.type]
		});
		offset = span.end;
	}

	if (offset < text.length) segments.push({ text: text.slice(offset) });
	return segments;
}
