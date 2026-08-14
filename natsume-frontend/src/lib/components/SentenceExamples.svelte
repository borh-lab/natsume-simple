<script lang="ts">
	import type { ApiClient } from '$lib/api/client';
	import type { CollocationItem, Example } from '$lib/api/types';
	import { sentenceSegments } from '$lib/sentence';

	let {
		client,
		item,
		selectedCorpusIds,
		expanded
	}: {
		client: ApiClient;
		item: CollocationItem;
		selectedCorpusIds: string[];
		expanded: boolean;
	} = $props();
	let status = $state<'idle' | 'loading' | 'success' | 'empty' | 'error'>('idle');
	let examples = $state<Example[]>([]);
	let request: AbortController | null = null;

	async function load() {
		if (status !== 'idle') return;
		status = 'loading';
		request = new AbortController();
		try {
			const response = await client.getExamples(
				{
					noun: item.noun,
					particle: item.particle,
					verb: item.verb,
					corpusIds: selectedCorpusIds,
					limit: 5
				},
				request.signal
			);
			examples = response.examples;
			status = examples.length ? 'success' : 'empty';
		} catch (error) {
			if (!(error instanceof DOMException && error.name === 'AbortError')) status = 'error';
		}
	}

	$effect(() => {
		if (expanded) void load();
	});
	$effect(() => () => request?.abort());
</script>

<div class="mt-2 w-full text-sm" aria-live="polite" data-testid="sentence-examples">
	{#if status === 'loading'}
		<p>Loading examples…</p>
	{:else if status === 'empty'}
		<p>No examples found.</p>
	{:else if status === 'error'}
		<p class="text-red-700 dark:text-red-300">Examples could not be loaded.</p>
	{:else}
		<ul class="space-y-2">
			{#each examples as example (example)}
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
</div>
