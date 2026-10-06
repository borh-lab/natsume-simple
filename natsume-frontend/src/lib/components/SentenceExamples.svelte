<script lang="ts">
	import type { ApiClient } from '$lib/api/client';
	import type { CollocationItem, Corpus, Example } from '$lib/api/types';
	import { corpusStyleForSlot } from '$lib/presentation/colors';
	import { responseMatchesResult } from '$lib/search/controller.svelte';
	import { sentenceSegments } from '$lib/sentence';

	let {
		client,
		item,
		corpora,
		colorSlots,
		selectedCorpusIds,
		databaseBuildId,
		expanded
	}: {
		client: ApiClient;
		item: CollocationItem;
		corpora: Corpus[];
		colorSlots: Record<string, number>;
		selectedCorpusIds: string[];
		databaseBuildId: string;
		expanded: boolean;
	} = $props();
	const corpusLabels = $derived(
		Object.fromEntries(corpora.map((corpus) => [corpus.id, corpus.label]))
	);
	let status = $state<
		'idle' | 'loading' | 'success' | 'empty' | 'request-error' | 'identity-error'
	>('idle');
	let examples = $state<Example[]>([]);
	let hasMore = $state(false);
	let initialRequested = false;
	let request: AbortController | null = null;

	async function load(limit: number) {
		if (status === 'loading' || status === 'identity-error') return;
		status = 'loading';
		request = new AbortController();
		try {
			const response = await client.getExamples(
				{
					noun: item.noun,
					particle: item.particle,
					verb: item.verb,
					corpusIds: selectedCorpusIds,
					offset: examples.length,
					limit
				},
				request.signal
			);
			if (!responseMatchesResult(response, databaseBuildId, selectedCorpusIds)) {
				status = 'identity-error';
				return;
			}
			examples = [...examples, ...response.examples];
			hasMore = response.hasMore;
			status = examples.length ? 'success' : 'empty';
		} catch (error) {
			if (!(error instanceof DOMException && error.name === 'AbortError')) {
				status = 'request-error';
			}
		} finally {
			request = null;
		}
	}

	$effect(() => {
		if (expanded && !initialRequested) {
			initialRequested = true;
			void load(5);
		}
	});
	$effect(() => () => request?.abort());
</script>

<div class="w-full text-sm leading-relaxed" aria-live="polite" data-testid="sentence-examples">
	{#if status === 'loading' && examples.length === 0}
		<p class="px-1 py-1">Loading examples…</p>
	{:else if status === 'empty'}
		<p>No examples found.</p>
	{:else}
		{#if examples.length > 0}
			<p class="mb-0.5 px-1 text-xs text-gray-600 dark:text-gray-400">
				{examples.length} examples shown
			</p>
			<ul class="divide-y divide-gray-200 dark:divide-gray-700">
				{#each examples as example, exampleIndex (exampleIndex)}
					{@const corpusStyle = corpusStyleForSlot(colorSlots[example.corpusId])}
					<li
						class="border-l-2 py-1 pl-1"
						style:border-left-color={corpusStyle.color}
						data-testid="example-row"
						data-corpus-id={example.corpusId}
					>
						{#each sentenceSegments( example.text, [{ ...example.nounSpan, type: 'noun' }, { ...example.particleSpan, type: 'particle' }, { ...example.verbSpan, type: 'verb' }] ) as segment, index (index)}
							{#if segment.className}<span class={segment.className}>{segment.text}</span
								>{:else}{segment.text}{/if}
						{/each}
						<span
							class={`block truncate text-xs font-medium ${corpusStyle.titleClass}`}
							title={`${corpusLabels[example.corpusId] ?? example.corpusId} · ${example.sourceTitle}`}
							data-testid="example-source"
							>{selectedCorpusIds.length > 1
								? `${corpusLabels[example.corpusId] ?? example.corpusId} · `
								: ''}{example.sourceTitle}:</span
						>
					</li>
				{/each}
			</ul>
		{/if}
		{#if status === 'loading'}
			<div
				class="mt-1 border border-gray-300 bg-gray-50 px-2 py-1 text-gray-900 dark:border-gray-600 dark:bg-gray-800 dark:text-gray-100"
			>
				Loading more examples…
			</div>
		{:else if status === 'request-error'}
			<div
				class="mt-1 border border-red-200 bg-red-50 px-2 py-1 dark:border-red-800 dark:bg-red-950"
			>
				{#if examples.length === 0}
					<p class="mb-0.5 text-red-700 dark:text-red-300">Examples could not be loaded.</p>
				{/if}
				<button
					type="button"
					class="font-medium underline"
					onclick={() => load(examples.length ? 20 : 5)}>Try again</button
				>
			</div>
		{:else if status === 'identity-error'}
			<p class="mt-1 bg-amber-50 px-2 py-1 text-amber-800 dark:bg-amber-950 dark:text-amber-200">
				Data changed — update the search before loading more.
			</p>
		{:else if hasMore}
			<button
				type="button"
				class="mt-1 w-full border border-gray-300 bg-gray-50 px-2 py-1 text-gray-900 hover:bg-gray-100 dark:border-gray-600 dark:bg-gray-800 dark:text-gray-100 dark:hover:bg-gray-700"
				onclick={() => load(20)}
			>
				Load more examples
			</button>
		{:else if examples.length > 0}
			<div class="mt-1 bg-gray-100 px-2 py-1 text-gray-600 dark:bg-gray-800 dark:text-gray-300">
				All examples shown
			</div>
		{/if}
	{/if}
</div>
