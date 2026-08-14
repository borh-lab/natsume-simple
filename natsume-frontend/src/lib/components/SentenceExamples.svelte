<script lang="ts">
	import type { ApiClient } from '$lib/api/client';
	import type { CollocationItem, Example } from '$lib/api/types';
	import { responseMatchesResult } from '$lib/search/controller.svelte';
	import { sentenceSegments } from '$lib/sentence';

	let {
		client,
		item,
		selectedCorpusIds,
		databaseBuildId,
		expanded
	}: {
		client: ApiClient;
		item: CollocationItem;
		selectedCorpusIds: string[];
		databaseBuildId: string;
		expanded: boolean;
	} = $props();
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

<div class="mt-2 w-full text-sm" aria-live="polite" data-testid="sentence-examples">
	{#if status === 'loading' && examples.length === 0}
		<p>Loading examples…</p>
	{:else if status === 'empty'}
		<p>No examples found.</p>
	{:else}
		{#if examples.length > 0}
			<p class="mb-1 text-xs text-gray-500">{examples.length} examples shown</p>
			<ul class="space-y-2">
				{#each examples as example, exampleIndex (exampleIndex)}
					<li class="rounded bg-gray-100 p-2 dark:bg-gray-800">
						<strong>{example.sourceTitle}:</strong>
						{#each sentenceSegments( example.text, [{ ...example.nounSpan, type: 'noun' }, { ...example.particleSpan, type: 'particle' }, { ...example.verbSpan, type: 'verb' }] ) as segment, index (index)}
							{#if segment.className}<span class={segment.className}>{segment.text}</span
								>{:else}{segment.text}{/if}
						{/each}
					</li>
				{/each}
			</ul>
		{/if}
		{#if status === 'loading'}
			<div
				class="mt-2 rounded border border-blue-200 bg-blue-50 p-2 text-blue-900 dark:border-blue-800 dark:bg-blue-950 dark:text-blue-100"
			>
				Loading more examples…
			</div>
		{:else if status === 'request-error'}
			<div
				class="mt-2 rounded border border-red-200 bg-red-50 p-2 dark:border-red-800 dark:bg-red-950"
			>
				{#if examples.length === 0}
					<p class="mb-1 text-red-700 dark:text-red-300">Examples could not be loaded.</p>
				{/if}
				<button
					type="button"
					class="font-medium underline"
					onclick={() => load(examples.length ? 20 : 5)}>Try again</button
				>
			</div>
		{:else if status === 'identity-error'}
			<p class="mt-2 rounded bg-amber-50 p-2 text-amber-800 dark:bg-amber-950 dark:text-amber-200">
				Data changed — update the search before loading more.
			</p>
		{:else if hasMore}
			<button
				type="button"
				class="mt-2 w-full rounded border border-blue-200 bg-blue-50 p-2 text-blue-900 hover:bg-blue-100 dark:border-blue-800 dark:bg-blue-950 dark:text-blue-100 dark:hover:bg-blue-900"
				onclick={() => load(20)}
			>
				Load more examples
			</button>
		{:else if examples.length > 0}
			<div class="mt-2 rounded bg-gray-100 p-2 text-gray-600 dark:bg-gray-800 dark:text-gray-300">
				All examples shown
			</div>
		{/if}
	{/if}
</div>
