<script lang="ts">
	import { untrack } from 'svelte';
	import type { ApiClient } from '$lib/api/client';
	import type { Corpus, ParticleGroup, SearchPosition } from '$lib/api/types';
	import {
		barReference,
		CORPUS_COLORS,
		corpusColorMap,
		particleMassSegments,
		type BarScale
	} from '$lib/presentation/search';
	import {
		collocationPageSize,
		responseMatchesResult,
		type SearchInput
	} from '$lib/search/controller.svelte';
	import CollocationItem from './CollocationItem.svelte';

	let {
		client,
		group,
		groups,
		corpora,
		selectedCorpusIds,
		databaseBuildId,
		searchInput,
		barScale,
		pos
	}: {
		client: ApiClient;
		group: ParticleGroup;
		groups: ParticleGroup[];
		corpora: Corpus[];
		selectedCorpusIds: string[];
		databaseBuildId: string;
		searchInput: SearchInput;
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
	let items = $state(untrack(() => [...group.items]));
	let pageStatus = $state<'idle' | 'loading' | 'request-error' | 'identity-error'>('idle');
	let request: AbortController | null = null;
	const pageSize = $derived(collocationPageSize(selectedCorpusIds.length));
	const remaining = $derived(group.totalMatchingCollocations - items.length);

	async function loadMore() {
		if (pageStatus === 'loading' || remaining <= 0) return;
		pageStatus = 'loading';
		request = new AbortController();
		try {
			const response = await client.getCollocations(
				{
					term: searchInput.term,
					pos: searchInput.pos,
					corpusIds: [...selectedCorpusIds],
					particle: group.particle,
					offsetPerParticle: items.length,
					limitPerParticle: pageSize
				},
				request.signal
			);
			if (!responseMatchesResult(response, databaseBuildId, selectedCorpusIds)) {
				pageStatus = 'identity-error';
				return;
			}
			const next = response.particleGroups.find(
				(candidate) => candidate.particle === group.particle
			);
			if (!next || next.totalMatchingCollocations !== group.totalMatchingCollocations) {
				pageStatus = 'identity-error';
				return;
			}
			if (next.items.length === 0 && remaining > 0) {
				pageStatus = 'request-error';
				return;
			}
			items = [...items, ...next.items];
			pageStatus = 'idle';
		} catch (error) {
			if (!(error instanceof DOMException && error.name === 'AbortError')) {
				pageStatus = 'request-error';
			}
		} finally {
			request = null;
		}
	}

	$effect(() => () => request?.abort());
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
		Showing {items.length} of {group.totalMatchingCollocations}
	</p>
	{#each items as item (item)}
		<CollocationItem
			{client}
			{item}
			{corpora}
			{selectedCorpusIds}
			{databaseBuildId}
			{reference}
			{colorSlots}
			{pos}
		/>
	{/each}
	{#if remaining > 0}
		<div class="mt-2 border-t pt-2 dark:border-gray-700">
			{#if pageStatus === 'identity-error'}
				<p class="text-sm text-amber-700 dark:text-amber-300">
					Data changed — update the search before loading more.
				</p>
			{:else}
				<button
					type="button"
					class="w-full rounded border px-3 py-2 text-sm hover:bg-gray-100 disabled:opacity-60 dark:border-gray-600 dark:hover:bg-gray-800"
					disabled={pageStatus === 'loading'}
					onclick={loadMore}
				>
					{pageStatus === 'loading'
						? 'Loading…'
						: pageStatus === 'request-error'
							? 'Try again'
							: `Load ${Math.min(pageSize, remaining)} more`}
				</button>
			{/if}
		</div>
	{/if}
</section>
