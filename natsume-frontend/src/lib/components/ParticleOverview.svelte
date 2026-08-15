<script lang="ts">
	import type { ApiClient } from '$lib/api/client';
	import type { Corpus } from '$lib/api/types';
	import type { BarScale } from '$lib/presentation/search';
	import type { VisibleSearchResult } from '$lib/search/controller.svelte';
	import ParticleColumn from './ParticleColumn.svelte';

	let {
		client,
		result,
		corpora,
		barScale
	}: { client: ApiClient; result: VisibleSearchResult; corpora: Corpus[]; barScale: BarScale } =
		$props();
	const response = $derived(result.response);
</script>

<!-- The overflowing region must be keyboard-focusable so native horizontal scrolling is operable. -->
<!-- svelte-ignore a11y_no_noninteractive_tabindex -->
<div
	class="flex min-w-full overflow-x-auto border-y focus-visible:outline-2 focus-visible:outline-offset-0 focus-visible:outline-gray-700 dark:border-gray-700 dark:focus-visible:outline-gray-200"
	role="region"
	aria-label="Particle collocations"
	tabindex="0"
	data-testid="particle-overview"
>
	{#each response.particleGroups as group (group.particle)}
		<ParticleColumn
			{client}
			{group}
			groups={response.particleGroups}
			{corpora}
			selectedCorpusIds={response.selectedCorpusIds}
			databaseBuildId={response.databaseBuildId}
			searchInput={result.input}
			{barScale}
			pos={result.input.pos}
		/>
	{/each}
</div>
