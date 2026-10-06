<script lang="ts">
	import { onMount } from 'svelte';
	import { base } from '$app/paths';
	import { ApiClient } from '$lib/api/client';
	import CorpusOptions from '$lib/components/CorpusOptions.svelte';
	import ParticleOverview from '$lib/components/ParticleOverview.svelte';
	import SearchControls from '$lib/components/SearchControls.svelte';
	import SearchSummary from '$lib/components/SearchSummary.svelte';
	import ThemeSwitch from '$lib/components/ThemeSwitch.svelte';
	import { corpusStyleForSlot } from '$lib/presentation/colors';
	import { corpusColorMap } from '$lib/presentation/search';
	import type { BarScale } from '$lib/presentation/search';
	import { SearchController } from '$lib/search/controller.svelte';
	import '../tailwind.css';

	const client = new ApiClient(import.meta.env.VITE_API_URL || base);
	const controller = new SearchController(client);
	let barScale = $state<BarScale>('particle');
	let selectedParticle = $state('が');
	const activeParticle = $derived(
		controller.result?.response.particleGroups.find((group) => group.particle === selectedParticle)
			?.particle ??
			controller.result?.response.particleGroups[0]?.particle ??
			'が'
	);
	const colorSlots = $derived(corpusColorMap(controller.corpora));
	const visibleCorpusIds = $derived(
		controller.result?.response.selectedCorpusIds ?? controller.selectedCorpusIds
	);
	let optionsOpen = $state(false);
	let optionsWidget = $state<HTMLDivElement>();
	const optionsLabel = $derived(
		controller.selectedCorpusIds.length === controller.corpora.length
			? 'Options'
			: `Options · ${controller.selectedCorpusIds.length}/${controller.corpora.length}`
	);

	function closeOptionsOnOutside(event: PointerEvent) {
		if (
			optionsOpen &&
			event.target instanceof Node &&
			optionsWidget &&
			!optionsWidget.contains(event.target)
		) {
			optionsOpen = false;
		}
	}

	function closeOptionsOnFocusout(event: FocusEvent & { currentTarget: HTMLDivElement }) {
		const next = event.relatedTarget;
		if (next instanceof Node && next !== document.body && !event.currentTarget.contains(next)) {
			optionsOpen = false;
		}
	}

	function closeOptionsOnEscape(event: KeyboardEvent) {
		if (event.key === 'Escape') optionsOpen = false;
	}

	onMount(() => controller.initialize());
</script>

<svelte:window onpointerdown={closeOptionsOnOutside} onkeydown={closeOptionsOnEscape} />

<svelte:head><title>Natsume Simple</title></svelte:head>

<header class="sticky top-0 z-40 border-b bg-white dark:border-gray-700 dark:bg-gray-900">
	<div
		class="grid w-full grid-cols-[minmax(0,1fr)_auto] items-center gap-x-3 gap-y-2 px-3 py-2 lg:grid-cols-[1fr_minmax(0,auto)_1fr]"
	>
		<div class="flex min-w-0 flex-nowrap items-center gap-2" data-testid="brand">
			<img class="h-8 w-8" src={`${base}/favicon.png`} alt="Natsume Simple" />
			<h1 class="whitespace-nowrap text-xl font-bold sm:text-2xl" tabindex="-1">Natsume Simple</h1>
		</div>
		<div
			class="col-span-2 row-start-2 flex w-full min-w-0 items-center justify-center lg:col-span-1 lg:col-start-2 lg:row-start-1 lg:justify-self-center"
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
		</div>
		<div class="col-start-2 row-start-1 justify-self-end lg:col-start-3"><ThemeSwitch /></div>
	</div>
	<div class="px-3">
		{#if controller.corpora.length > 0}
			<div
				class="relative flex h-10 min-w-0 items-center gap-2 border-t border-gray-200 dark:border-gray-700"
				data-testid="results-toolbar"
			>
				<div
					class="relative shrink-0"
					bind:this={optionsWidget}
					onfocusout={closeOptionsOnFocusout}
				>
					<button
						type="button"
						class="h-8 rounded border border-gray-400 bg-white px-2 text-sm font-medium hover:bg-gray-100 focus-visible:outline-2 focus-visible:outline-offset-0 dark:border-gray-600 dark:bg-gray-900 dark:hover:bg-gray-800 xl:hidden"
						aria-expanded={optionsOpen}
						aria-controls="results-options"
						onclick={() => (optionsOpen = !optionsOpen)}
					>
						{optionsLabel}
					</button>
					<div
						id="results-options"
						class={`${optionsOpen ? 'flex' : 'hidden xl:flex'} absolute left-0 top-full z-30 mt-1 w-[min(24rem,calc(100vw-2rem))] flex-col gap-2 rounded border border-gray-300 bg-white p-2 shadow-lg dark:border-gray-600 dark:bg-gray-900 xl:static xl:mt-0 xl:w-auto xl:flex-row xl:items-center xl:border-0 xl:bg-transparent xl:p-0 xl:shadow-none xl:dark:bg-transparent`}
					>
						<CorpusOptions
							corpora={controller.corpora}
							selectedCorpusIds={controller.selectedCorpusIds}
							disabled={controller.status === 'loading'}
							ontoggle={(corpusId) => controller.toggleCorpus(corpusId)}
						/>
						{#if controller.result}
							<div
								class="flex shrink-0 items-center gap-1.5 border-gray-300 text-sm xl:border-l xl:pl-2 dark:border-gray-600"
							>
								<label for="bar-scale">Scale</label>
								<select
									id="bar-scale"
									aria-label="Bar scale"
									title="Bar length: mean frequency per million; scale per particle or shared"
									class="h-8 rounded border bg-white px-1.5 dark:border-gray-600 dark:bg-gray-800"
									bind:value={barScale}
								>
									<option value="particle">Per particle</option>
									<option value="global">Shared</option>
								</select>
							</div>
						{/if}
					</div>
				</div>
				<SearchSummary
					result={controller.result}
					stale={controller.resultIsStale}
					draftDiffers={controller.draftDiffersFromResult}
				/>
			</div>
			<div
				class="flex h-5 items-center gap-3 overflow-hidden text-xs xl:hidden"
				aria-label="Corpus legend"
				data-testid="corpus-legend"
			>
				{#each controller.corpora.filter( (corpus) => visibleCorpusIds.includes(corpus.id) ) as corpus (corpus.id)}
					<span class="inline-flex min-w-0 items-center gap-1 whitespace-nowrap">
						<span
							class="h-2 w-2 shrink-0 rounded-sm"
							style:background-color={corpusStyleForSlot(colorSlots[corpus.id]).color}
							aria-hidden="true"
						></span>
						{corpus.label}
					</span>
				{/each}
			</div>
		{/if}
		{#if controller.result}
			<nav
				class="flex min-w-0 items-center gap-1 overflow-x-auto border-t border-gray-200 py-1 dark:border-gray-700"
				aria-label="Particles"
			>
				{#each controller.result.response.particleGroups as group (group.particle)}
					<button
						type="button"
						aria-label={`Particle ${group.particle}`}
						title={`${group.totalMatchingCollocations} matches`}
						aria-pressed={activeParticle === group.particle}
						onclick={() => (selectedParticle = group.particle)}
						class={`flex h-6 shrink-0 items-center gap-1 rounded-sm px-2 text-sm focus-visible:outline-2 focus-visible:outline-offset-0 ${activeParticle === group.particle ? 'bg-gray-900 text-white dark:bg-gray-100 dark:text-gray-900' : 'text-gray-700 hover:bg-gray-100 dark:text-gray-300 dark:hover:bg-gray-800'}`}
					>
						<span class="font-semibold">{group.particle}</span>
						<span class="hidden text-xs tabular-nums opacity-80 sm:inline"
							>{group.totalMatchingCollocations}</span
						>
					</button>
				{/each}
			</nav>
		{/if}
	</div>
</header>

<main class="w-full min-w-0 space-y-2 px-3 py-2">
	{#if controller.status === 'error'}
		<p class="rounded bg-red-100 p-3 text-red-900" role="alert">{controller.errorMessage}</p>
	{:else if controller.status === 'empty'}
		<p>No collocations found.</p>
	{/if}
	{#if controller.result}
		{#key controller.result}
			<ParticleOverview
				{client}
				result={controller.result}
				corpora={controller.corpora}
				{barScale}
				selectedParticle={activeParticle}
			/>
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
