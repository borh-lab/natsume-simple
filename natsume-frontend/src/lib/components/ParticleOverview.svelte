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

<!-- The overflowing region must be keyboard-focusable so native horizontal scrolling is operable. -->
<!-- svelte-ignore a11y_no_noninteractive_tabindex -->
<div
	class="flex overflow-x-auto border-y focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-blue-600 dark:border-gray-700"
	role="region"
	aria-label="Particle collocations"
	tabindex="0"
	data-testid="particle-overview"
>
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
