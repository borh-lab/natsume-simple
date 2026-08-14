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
	let expanded = $state(false);
</script>

<div class="py-1">
	<details class="w-full min-w-0" ontoggle={(event) => (expanded = event.currentTarget.open)}>
		<summary class="flex cursor-pointer items-center gap-2 font-medium">
			<svg width="64" height="20" aria-hidden="true" class="shrink-0 rounded">
				{#each segments as segment, index (segment.corpusId)}
					<rect
						x={`${segment.offset}%`}
						width={`${segment.percentage}%`}
						height="20"
						fill={colors[index % colors.length]}
					/>
				{/each}
			</svg>
			<span>{pos === 'noun' ? item.verb : item.noun}</span>
		</summary>
		<SentenceExamples {client} {item} {selectedCorpusIds} {expanded} />
	</details>
</div>
