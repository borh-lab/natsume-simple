<script lang="ts">
	import type { ApiClient } from '$lib/api/client';
	import type { CollocationItem, Corpus, SearchPosition } from '$lib/api/types';
	import { CORPUS_COLORS, itemBarSegments } from '$lib/presentation/search';
	import SentenceExamples from './SentenceExamples.svelte';

	let {
		client,
		item,
		corpora,
		selectedCorpusIds,
		databaseBuildId,
		reference,
		colorSlots,
		pos
	}: {
		client: ApiClient;
		item: CollocationItem;
		corpora: Corpus[];
		selectedCorpusIds: string[];
		databaseBuildId: string;
		reference: number;
		colorSlots: Record<string, number>;
		pos: SearchPosition;
	} = $props();
	const counts = $derived(
		Object.fromEntries(corpora.map((corpus) => [corpus.id, corpus.collocationCount]))
	);
	const corpusLabels = $derived(
		Object.fromEntries(corpora.map((corpus) => [corpus.id, corpus.label]))
	);
	const segments = $derived(itemBarSegments(item, selectedCorpusIds, counts, reference));
	const totalPercentage = $derived(segments.reduce((sum, segment) => sum + segment.percentage, 0));
	let expanded = $state(false);
</script>

<div class="py-1">
	<details class="w-full min-w-0" ontoggle={(event) => (expanded = event.currentTarget.open)}>
		<summary
			class="grid cursor-pointer grid-cols-[minmax(6rem,2fr)_minmax(0,3fr)] items-center gap-2 font-medium"
		>
			<svg
				viewBox="0 0 100 16"
				preserveAspectRatio="none"
				class="h-4 w-full overflow-hidden rounded"
				role="img"
				aria-label={`${item.meanFrequencyPerMillion.toLocaleString(undefined, { maximumFractionDigits: 1 })} mean frequency per million; ${totalPercentage.toLocaleString(undefined, { maximumFractionDigits: 1 })}% of the selected reference`}
				data-testid="item-bar"
			>
				<rect
					width="100"
					height="16"
					fill="currentColor"
					class="text-gray-200 dark:text-gray-700"
				/>
				{#each segments as segment (segment.corpusId)}
					<rect
						x={segment.offset}
						width={segment.percentage}
						height="16"
						fill={CORPUS_COLORS[colorSlots[segment.corpusId] % CORPUS_COLORS.length]}
						data-testid="bar-segment"
					>
						<title
							>{corpusLabels[segment.corpusId]}: {segment.rawFrequency.toLocaleString()} occurrences ·
							{segment.frequencyPerMillion.toLocaleString(undefined, { maximumFractionDigits: 1 })} per
							million</title
						>
					</rect>
				{/each}
			</svg>
			<span class="min-w-0 truncate">{pos === 'noun' ? item.verb : item.noun}</span>
		</summary>
		<SentenceExamples {client} {item} {selectedCorpusIds} {databaseBuildId} {expanded} />
	</details>
</div>
