<script lang="ts">
	import type { ApiClient } from '$lib/api/client';
	import type { CollocationsResponse, Corpus, SearchPosition } from '$lib/api/types';
	import type { BarScale } from '$lib/presentation/search';
	import ParticleColumn from './ParticleColumn.svelte';

	let {
		client,
		result,
		corpora,
		pos
	}: { client: ApiClient; result: CollocationsResponse; corpora: Corpus[]; pos: SearchPosition } =
		$props();
	let barScale = $state<BarScale>('particle');
</script>

<div class="space-y-2">
	<div class="flex w-fit items-center gap-2 text-sm">
		<label for="bar-scale">Bar scale</label>
		<select
			id="bar-scale"
			class="rounded border bg-white px-2 py-1 dark:border-gray-600 dark:bg-gray-800"
			bind:value={barScale}
		>
			<option value="particle">Within particle</option>
			<option value="global">Across particles</option>
		</select>
	</div>

	<!-- The overflowing region must be keyboard-focusable so native horizontal scrolling is operable. -->
	<!-- svelte-ignore a11y_no_noninteractive_tabindex -->
	<div
		class="flex min-w-full overflow-x-auto border-y focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-blue-600 dark:border-gray-700"
		role="region"
		aria-label="Particle collocations"
		tabindex="0"
		data-testid="particle-overview"
	>
		{#each result.particleGroups as group (group.particle)}
			<ParticleColumn
				{client}
				{group}
				groups={result.particleGroups}
				{corpora}
				selectedCorpusIds={result.selectedCorpusIds}
				{barScale}
				{pos}
			/>
		{/each}
	</div>
</div>
