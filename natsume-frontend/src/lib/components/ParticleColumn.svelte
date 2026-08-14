<script lang="ts">
	import type { ApiClient } from '$lib/api/client';
	import type { Corpus, ParticleGroup, RankBy, SearchPosition } from '$lib/api/types';
	import CollocationItem from './CollocationItem.svelte';

	let {
		client,
		group,
		corpora,
		selectedCorpusIds,
		rankBy,
		pos
	}: {
		client: ApiClient;
		group: ParticleGroup;
		corpora: Corpus[];
		selectedCorpusIds: string[];
		rankBy: RankBy;
		pos: SearchPosition;
	} = $props();
</script>

<section
	class="w-80 shrink-0 border-r px-3 py-2 last:border-r-0 dark:border-gray-700"
	data-testid="particle-column"
>
	<h2 class="text-xl font-bold">{group.particle}</h2>
	<p class="text-xs text-gray-500">{group.returnedCount} of {group.totalMatchingCollocations}</p>
	{#each group.items as item (JSON.stringify( [selectedCorpusIds, item.noun, item.particle, item.verb] ))}
		<CollocationItem {client} {item} {corpora} {selectedCorpusIds} {rankBy} {pos} />
	{/each}
</section>
