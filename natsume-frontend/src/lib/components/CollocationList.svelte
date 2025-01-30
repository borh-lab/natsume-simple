<script lang="ts">
	// Add this function to handle highlighting with spans
	function highlightSentence(
		text: string,
		spans: Array<{ start: number; end: number; type: 'noun' | 'particle' | 'verb' }>
	): string {
		// Sort spans by start position in reverse order to handle overlapping spans
		const sortedSpans = [...spans].sort((a, b) => b.start - a.start);

		let result = text;
		for (const span of sortedSpans) {
			const before = result.slice(0, span.start);
			const highlight = result.slice(span.start, span.end);
			const after = result.slice(span.end);

			const className = {
				noun: 'font-bold text-blue-600 dark:text-blue-400',
				particle: 'font-bold text-red-600 dark:text-red-400',
				verb: 'font-bold text-green-600 dark:text-green-400'
			}[span.type];

			result = `${before}<span class="${className}">${highlight}</span>${after}`;
		}

		return result;
	}
	import type { CombinedResult, Result } from '$lib/query';
	import { getContext, onMount } from 'svelte';
	import type { Writable } from 'svelte/store';
	const {
		collocates,
		combinedSearch,
		getColor,
		tooltipAction,
		renderContributions,
		renderContributionsCombined,
		useNormalization,
		selectedCorpora,
		getSolidColor,
		corpusNorm,
		searchType
	}: {
		collocates: Result[];
		combinedSearch: Writable<boolean>;
		getColor: (corpus: string) => string;
		tooltipAction: (
			node: HTMLLIElement,
			{
				getTooltipData,
				useNormalization
			}: {
				getTooltipData: () => Record<string, number>;
				useNormalization: boolean;
			}
		) => void;
		renderContributions: (
			collocate: Result,
			collocates: Result[],
			selectedCorpora: string[],
			useNormalization: boolean
		) => { corpus: string; width: number; xOffset: number; value: number }[];
		renderContributionsCombined: (
			collocate: CombinedResult,
			collocates: Result[],
			selectedCorpora: string[],
			useNormalization: boolean
		) => { corpus: string; width: number; xOffset: number; value: number }[];
		useNormalization: Writable<boolean>;
		selectedCorpora: Writable<string[]>;
		getSolidColor: (corpus: string) => string;
		corpusNorm: Record<string, number>;
		searchType: 'verb' | 'noun';
	} = $props();

	const apiUrl = getContext<string>('apiUrl');

	let sentencesMap: Record<
		string,
		Array<{
			corpus: string;
			text: string;
			n_begin: number;
			n_end: number;
			p_begin: number;
			p_end: number;
			v_begin: number;
			v_end: number;
		}>
	> = $state({});

	async function loadSentences(collocate: Result) {
		const key = `${collocate.n}-${collocate.p}-${collocate.v}`;

		// Only fetch if we haven't already loaded these sentences
		if (!sentencesMap[key]) {
			const response = await fetch(
				`${apiUrl}/sentences/${collocate.n}/${collocate.p}/${collocate.v}/5`
			);
			const data = await response.json();
			sentencesMap[key] = data;
		}
	}
</script>

{#snippet frequencyBar(collocate: Result | CombinedResult)}
	<svg width="50" height="20" class="mr-2">
		{#if !$combinedSearch}
			{#each renderContributions(collocate as Result, collocates, $selectedCorpora, $useNormalization) as { corpus, width, xOffset, value }}
				<rect style="fill: {getSolidColor(corpus)}" height="20" width="{width}%" x="{xOffset}%" />
			{/each}
		{:else}
			{#each renderContributionsCombined(collocate as CombinedResult, collocates, $selectedCorpora, $useNormalization) as { corpus, width, xOffset, value }}
				<rect style="fill: {getSolidColor(corpus)}" height="20" width="{width}%" x="{xOffset}%" />
			{/each}
		{/if}
	</svg>
{/snippet}

{#snippet collocationContent(collocate: Result)}
	<span class="text-left font-medium">
		{searchType === 'verb' ? collocate.n : collocate.v}
	</span>
	{#if !$combinedSearch}
		<span class="text-sm text-gray-600 dark:text-gray-200 ml-2 text-left">{collocate.corpus}</span>
	{/if}
{/snippet}

<ul class="list-none p-0 space-y-0 mt-2">
	{#each collocates as collocate}
		<!-- TODO: Add "Click to expand" to tooltip -->
		<li
			class="flex items-center justify-start p-0"
			style="background-color: {$combinedSearch ? 'transparent' : getColor(collocate.corpus)}"
			use:tooltipAction={{
				getTooltipData: () => {
					const tooltipData: Record<string, number> = {};
					if ($combinedSearch) {
						collocate.contributions?.forEach(
							({ corpus, frequency }: { corpus: string; frequency: number }) => {
								tooltipData[corpus] = $useNormalization
									? frequency * corpusNorm[corpus]
									: frequency;
							}
						);
					} else {
						tooltipData[collocate.corpus] = $useNormalization
							? collocate.frequency * corpusNorm[collocate.corpus]
							: collocate.frequency;
					}
					return tooltipData;
				},
				useNormalization: $useNormalization
			}}
		>
			<details
				ontoggle={(e) => {
					// Load sentences when details is opened
					if ((e.target as HTMLDetailsElement).open) {
						loadSentences(collocate);
					}
				}}
			>
				<summary class="list-none flex items-center justify-start p-0">
					{@render frequencyBar(collocate)}
					{@render collocationContent(collocate)}
				</summary>
				<ul class="list-none p-0 space-y-0 mt-2">
					{#each sentencesMap[`${collocate.n}-${collocate.p}-${collocate.v}`] || [] as sentence}
						<li>
							{sentence.corpus}:{' '}
							{@html highlightSentence(sentence.text, [
								{ start: sentence.n_begin, end: sentence.n_end, type: 'noun' },
								{ start: sentence.p_begin, end: sentence.p_end, type: 'particle' },
								{ start: sentence.v_begin, end: sentence.v_end, type: 'verb' }
							])}
						</li>
					{/each}
				</ul>
			</details>
		</li>
	{/each}
</ul>
