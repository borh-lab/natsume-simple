<script lang="ts">
	import type { ApiClient } from '$lib/api/client';
	import type { Corpus, ParticleGroup, SearchPosition } from '$lib/api/types';
	import {
		barReference,
		CORPUS_COLORS,
		corpusColorMap,
		particleMassSegments,
		type BarScale
	} from '$lib/presentation/search';
	import CollocationItem from './CollocationItem.svelte';

	let {
		client,
		group,
		groups,
		corpora,
		selectedCorpusIds,
		barScale,
		pos
	}: {
		client: ApiClient;
		group: ParticleGroup;
		groups: ParticleGroup[];
		corpora: Corpus[];
		selectedCorpusIds: string[];
		barScale: BarScale;
		pos: SearchPosition;
	} = $props();
	const reference = $derived(barReference(groups, group, barScale));
	const mass = $derived(particleMassSegments(group, groups));
	const colorSlots = $derived(corpusColorMap(corpora));
	const corpusLabels = $derived(
		Object.fromEntries(corpora.map((corpus) => [corpus.id, corpus.label]))
	);
	const percentage = $derived(
		mass.percentage.toLocaleString(undefined, { maximumFractionDigits: 1 })
	);
</script>

<section
	class="min-w-80 flex-1 border-r px-3 py-2 last:border-r-0 dark:border-gray-700"
	data-testid="particle-column"
>
	<div class="flex items-baseline justify-between gap-2">
		<h2 class="text-xl font-bold">{group.particle}</h2>
		<span class="text-xs tabular-nums text-gray-500">{percentage}% of total</span>
	</div>
	<div
		class="relative mt-1 h-1.5 w-full overflow-hidden rounded-full bg-gray-200 dark:bg-gray-700"
		role="img"
		aria-label={`${group.particle}: ${percentage}% of selected frequency`}
		data-testid="particle-mass"
	>
		{#each mass.segments as segment (segment.corpusId)}
			<span
				class="absolute inset-y-0"
				style:left={`${segment.offset}%`}
				style:width={`${segment.percentage}%`}
				style:background-color={CORPUS_COLORS[colorSlots[segment.corpusId] % CORPUS_COLORS.length]}
				title={`${corpusLabels[segment.corpusId]}: ${segment.rawFrequency.toLocaleString()} occurrences · ${segment.frequencyPerMillion.toLocaleString(undefined, { maximumFractionDigits: 1 })} per million`}
			></span>
		{/each}
	</div>
	<p class="mt-1 text-xs text-gray-500">
		{group.returnedCount} of {group.totalMatchingCollocations}
	</p>
	{#each group.items as item (JSON.stringify( [selectedCorpusIds, item.noun, item.particle, item.verb] ))}
		<CollocationItem {client} {item} {corpora} {selectedCorpusIds} {reference} {colorSlots} {pos} />
	{/each}
</section>
