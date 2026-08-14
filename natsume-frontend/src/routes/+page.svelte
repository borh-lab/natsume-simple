<script lang="ts">
	import { onMount } from 'svelte';
	import { ApiClient } from '$lib/api/client';
	import CorpusOptions from '$lib/components/CorpusOptions.svelte';
	import ParticleOverview from '$lib/components/ParticleOverview.svelte';
	import SearchControls from '$lib/components/SearchControls.svelte';
	import SearchSummary from '$lib/components/SearchSummary.svelte';
	import ThemeSwitch from '$lib/components/ThemeSwitch.svelte';
	import { SearchController } from '$lib/search/controller.svelte';
	import '../tailwind.css';

	const client = new ApiClient(import.meta.env.VITE_API_URL || '');
	const controller = new SearchController(client);

	onMount(() => controller.initialize());
</script>

<svelte:head><title>Natsume Simple</title></svelte:head>

<header class="border-b bg-white dark:border-gray-700 dark:bg-gray-900">
	<div class="flex w-full flex-wrap items-center gap-3 p-4">
		<div class="flex items-center gap-3" data-testid="brand">
			<img class="h-8 w-8" src="/favicon.png" alt="Natsume Simple" />
			<h1 class="text-2xl font-bold" tabindex="-1">Natsume Simple</h1>
		</div>
		<div
			class="ml-auto flex flex-wrap items-center justify-end gap-2"
			data-testid="header-controls"
		>
			<SearchControls
				bind:term={controller.term}
				bind:pos={controller.pos}
				loading={controller.status === 'loading'}
				dirty={controller.draftDiffersFromResult}
				onsubmit={() => controller.submit()}
				findSuggestions={async (query, pos) =>
					(await client.getSuggestions(query, pos)).suggestions}
			/>
			<ThemeSwitch />
		</div>
	</div>
</header>

<main class="w-full min-w-0 space-y-4 p-4">
	<CorpusOptions
		corpora={controller.corpora}
		selectedCorpusIds={controller.selectedCorpusIds}
		disabled={controller.status === 'loading'}
		ontoggle={(corpusId) => controller.toggleCorpus(corpusId)}
	/>
	<SearchSummary
		result={controller.result}
		stale={controller.resultIsStale}
		draftDiffers={controller.draftDiffersFromResult}
	/>
	{#if controller.status === 'error'}
		<p class="rounded bg-red-100 p-3 text-red-900" role="alert">{controller.errorMessage}</p>
	{:else if controller.status === 'empty'}
		<p>No collocations found.</p>
	{/if}
	{#if controller.result}
		{#key controller.result}
			<ParticleOverview {client} result={controller.result} corpora={controller.corpora} />
		{/key}
	{/if}
</main>

<footer
	class="mt-8 border-t bg-gray-50 text-sm text-gray-700 dark:border-gray-700 dark:bg-gray-900 dark:text-gray-300"
>
	<div class="w-full space-y-2 p-4">
		<p>
			Corpus sentences are extracted and normalized from the
			<a class="underline" href="https://www.anlp.jp/resource/journal_latex/"
				>Journal of Natural Language Processing</a
			>
			(<a class="underline" href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>) and
			<a class="underline" href="https://ja.wikipedia.org/">Japanese Wikipedia</a>
			(<a class="underline" href="https://creativecommons.org/licenses/by-sa/4.0/">CC BY-SA 4.0</a
			>), and TED Talks via the
			<a
				class="underline"
				href="https://huggingface.co/datasets/IWSLT/iwslt2017/tree/c18a4f81a47ae6fa079fe9d32db288ddde38451d/data/2017-01-trnted/texts/ja/en"
				>IWSLT 2017 Japanese–English dataset</a
			>.
		</p>
		<p>
			For attribution details, corrections, or takedown requests, see the corpus notices or
			<a class="underline" href="mailto:dev@bor.space">Contact</a>.
		</p>
	</div>
</footer>
