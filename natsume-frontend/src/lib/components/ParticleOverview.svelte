<script lang="ts">
	import type { ApiClient } from '$lib/api/client';
	import type { CollocationsResponse, Corpus, SearchPosition } from '$lib/api/types';
	import ParticleColumn from './ParticleColumn.svelte';

	let {
		client,
		result,
		corpora,
		pos
	}: { client: ApiClient; result: CollocationsResponse; corpora: Corpus[]; pos: SearchPosition } =
		$props();
</script>

<div class="grid gap-3 sm:grid-cols-2 xl:grid-cols-4">
	{#each result.particleGroups as group (group.particle)}
		<ParticleColumn
			{client}
			{group}
			{corpora}
			selectedCorpusIds={result.selectedCorpusIds}
			rankBy={result.rankBy}
			{pos}
		/>
	{/each}
</div>
