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
	<div class="mx-auto flex max-w-screen-2xl flex-wrap items-center justify-between gap-3 p-4">
		<h1 class="text-2xl font-bold">Natsume Simple</h1>
		<ThemeSwitch />
		<SearchControls
			bind:term={controller.term}
			bind:pos={controller.pos}
			loading={controller.status === 'loading'}
			onsubmit={() => controller.submit()}
			findSuggestions={async (query, pos) => (await client.getSuggestions(query, pos)).suggestions}
		/>
	</div>
</header>

<main class="mx-auto max-w-screen-2xl space-y-4 p-4">
	<CorpusOptions
		corpora={controller.corpora}
		selectedCorpusIds={controller.selectedCorpusIds}
		rankBy={controller.rankBy}
		disabled={controller.status === 'loading'}
		ontoggle={(corpusId) => controller.toggleCorpus(corpusId)}
		onrank={(rankBy) => controller.selectRank(rankBy)}
	/>
	<SearchSummary result={controller.result} stale={controller.resultIsStale} />
	{#if controller.status === 'error'}
		<p class="rounded bg-red-100 p-3 text-red-900" role="alert">{controller.errorMessage}</p>
	{:else if controller.status === 'empty'}
		<p>No collocations found.</p>
	{/if}
	{#if controller.result}
		<ParticleOverview
			{client}
			result={controller.result}
			corpora={controller.corpora}
			pos={controller.pos}
		/>
	{/if}
</main>
