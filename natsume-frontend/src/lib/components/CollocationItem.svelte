<script lang="ts">
	import type { ApiClient } from '$lib/api/client';
	import type { CollocationItem, Corpus, RankBy, SearchPosition } from '$lib/api/types';
	import { stackSegments } from '$lib/presentation/search';
	import SentenceExamples from './SentenceExamples.svelte';

	let {
		client,
		item,
		corpora,
		selectedCorpusIds,
		rankBy,
		pos
	}: {
		client: ApiClient;
		item: CollocationItem;
		corpora: Corpus[];
		selectedCorpusIds: string[];
		rankBy: RankBy;
		pos: SearchPosition;
	} = $props();
	const counts = $derived(
		Object.fromEntries(corpora.map((corpus) => [corpus.id, corpus.collocationCount]))
	);
	const segments = $derived(stackSegments(item.contributions, selectedCorpusIds, counts, rankBy));
	const colors = ['#dc2626', '#7c3aed', '#16a34a', '#2563eb', '#ca8a04', '#db2777'];
</script>

<div class="flex items-start gap-2 py-1">
	<svg width="64" height="20" aria-hidden="true" class="mt-1 shrink-0 rounded">
		{#each segments as segment, index (segment.corpusId)}
			<rect
				x={`${segment.offset}%`}
				width={`${segment.percentage}%`}
				height="20"
				fill={colors[index % colors.length]}
			/>
		{/each}
	</svg>
	<SentenceExamples
		{client}
		{item}
		{selectedCorpusIds}
		label={pos === 'noun' ? item.verb : item.noun}
	/>
</div>
