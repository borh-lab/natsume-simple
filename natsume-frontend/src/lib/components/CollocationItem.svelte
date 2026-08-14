<script lang="ts">
	import type { ApiClient } from '$lib/api/client';
	import type { CollocationItem, Corpus, SearchPosition } from '$lib/api/types';
	import { corpusStyleForSlot } from '$lib/presentation/colors';
	import { itemBarSegments } from '$lib/presentation/search';
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

<div>
	<details
		class="group w-full min-w-0 border-b border-gray-200 dark:border-gray-700 [&>summary::-webkit-details-marker]:hidden [&>summary]:list-none"
		ontoggle={(event) => (expanded = event.currentTarget.open)}
	>
		<summary
			class="grid min-h-0 w-full cursor-pointer grid-cols-[auto_minmax(5rem,2fr)_minmax(0,3fr)] items-center gap-1 px-1 py-1 text-sm leading-5 hover:bg-gray-100 focus-visible:outline-2 focus-visible:outline-offset-0 focus-visible:outline-gray-700 group-open:bg-gray-200 dark:hover:bg-gray-800 dark:focus-visible:outline-gray-200 dark:group-open:bg-gray-700"
		>
			<span aria-hidden="true" class="disclosure-chevron inline-block transition-transform">▶</span>
			<svg
				viewBox="0 0 100 16"
				preserveAspectRatio="none"
				class="h-2.5 w-full overflow-hidden rounded"
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
						fill={corpusStyleForSlot(colorSlots[segment.corpusId]).color}
						data-testid="bar-segment"
						data-corpus-id={segment.corpusId}
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
		<SentenceExamples
			{client}
			{item}
			{corpora}
			{colorSlots}
			{selectedCorpusIds}
			{databaseBuildId}
			{expanded}
		/>
	</details>
</div>

<style>
	details[open] .disclosure-chevron {
		transform: rotate(90deg);
	}
</style>
