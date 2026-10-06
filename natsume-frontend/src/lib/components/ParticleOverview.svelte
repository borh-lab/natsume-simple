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
		barScale,
		selectedParticle
	}: {
		client: ApiClient;
		result: VisibleSearchResult;
		corpora: Corpus[];
		barScale: BarScale;
		selectedParticle: string;
	} = $props();
	const response = $derived(result.response);
	let region = $state<HTMLDivElement>();
	$effect(() => {
		const index = response.particleGroups.findIndex((group) => group.particle === selectedParticle);
		const column = region?.children[index];
		if (region && column instanceof HTMLElement) region.scrollLeft = column.offsetLeft;
	});
</script>

<!-- The overflowing region must be keyboard-focusable so native horizontal scrolling is operable. -->
<!-- svelte-ignore a11y_no_noninteractive_tabindex -->
<div
	bind:this={region}
	class="relative flex min-w-full overflow-x-auto border-y focus-visible:outline-2 focus-visible:outline-offset-0 focus-visible:outline-gray-700 dark:border-gray-700 dark:focus-visible:outline-gray-200"
	role="region"
	aria-label="Particle collocations"
	tabindex="0"
	data-testid="particle-overview"
>
	{#each response.particleGroups as group (group.particle)}
		<div
			class={`${group.particle === selectedParticle ? 'block' : 'hidden'} min-w-0 flex-1 md:block md:min-w-80 md:border-r last:border-r-0 dark:border-gray-700`}
		>
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
		</div>
	{/each}
</div>
