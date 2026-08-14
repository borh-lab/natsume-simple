import { ROLE_TEXT_CLASSES } from '$lib/presentation/colors';

export type SentenceSpan = {
	start: number;
	end: number;
	type: 'noun' | 'particle' | 'verb';
};

export type SentenceSegment = {
	text: string;
	className?: string;
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
			className: ROLE_TEXT_CLASSES[span.type]
		});
		offset = span.end;
	}

	if (offset < text.length) segments.push({ text: text.slice(offset) });
	return segments;
}
