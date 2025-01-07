<script lang="ts">
import type { CombinedResult, Result } from "$lib/query";
import type { Writable } from "svelte/store";
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
}: {
	collocates: Result[];
	combinedSearch: Writable<boolean>;
	getColor: (corpus: string) => string;
	tooltipAction: (
		node: HTMLLIElement,
		{
			getTooltipData,
			useNormalization,
		}: {
			getTooltipData: () => Record<string, number>;
			useNormalization: boolean;
		},
	) => void;
	renderContributions: (
		collocate: Result,
		collocates: Result[],
		selectedCorpora: string[],
		useNormalization: boolean,
	) => { corpus: string; width: number; xOffset: number; value: number }[];
	renderContributionsCombined: (
		collocate: CombinedResult,
		collocates: Result[],
		selectedCorpora: string[],
		useNormalization: boolean,
	) => { corpus: string; width: number; xOffset: number; value: number }[];
	useNormalization: Writable<boolean>;
	selectedCorpora: Writable<string[]>;
	getSolidColor: (corpus: string) => string;
	corpusNorm: Record<string, number>;
} = $props();
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
	<span class="text-left font-medium">{collocate.n}{collocate.v}</span>
	{#if !$combinedSearch}
		<span class="text-sm text-gray-600 dark:text-gray-200 ml-2 text-left">{collocate.corpus}</span>
	{/if}
{/snippet}

<ul class="list-none p-0 space-y-0 mt-2">
	{#each collocates as collocate}
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
			{@render frequencyBar(collocate)}
			{@render collocationContent(collocate)}
		</li>
	{/each}
</ul>
