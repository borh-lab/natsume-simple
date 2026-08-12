<script lang="ts">
	import type { Collocate } from '$lib/query';
	import { sentenceSegments } from '$lib/sentence';
	import type { Frequency } from '$lib/stores/corpus';
	import { getContext } from 'svelte';
	import type { Writable } from 'svelte/store';

	type TooltipAction = (
		node: HTMLLIElement,
		parameters: { getTooltipData: () => Record<string, string> }
	) => { destroy: () => void };

	const {
		collocates,
		maxFrequency,
		getColor,
		tooltipAction,
		useNormalization,
		selectedCorpora,
		getSolidColor,
		searchType
	}: {
		collocates: Collocate[];
		maxFrequency: Frequency;
		getColor: (corpus: string) => string;
		tooltipAction: TooltipAction;
		useNormalization: Writable<boolean>;
		selectedCorpora: Writable<string[]>;
		getSolidColor: (corpus: string) => string;
		searchType: 'verb' | 'noun';
	} = $props();

	function hasVisibleContributions(collocate: Collocate): boolean {
		return collocate.contributions.some(({ corpus }) => $selectedCorpora.includes(corpus));
	}

	function getTotalFrequency(collocate: Collocate): number {
		return collocate.contributions
			.filter(({ corpus }) => $selectedCorpora.includes(corpus))
			.reduce((sum, { normalizedFrequency, rawFrequency }) => {
				return sum + ($useNormalization ? normalizedFrequency : rawFrequency);
			}, 0);
	}

	const sortedCollocates = $derived(
		collocates
			.filter(hasVisibleContributions)
			.sort((a, b) => getTotalFrequency(b) - getTotalFrequency(a))
	);

	const apiUrl = getContext<string>('apiUrl');

	function renderContributions(collocate: Collocate) {
		let xOffset = 0;
		return collocate.contributions
			.filter(({ corpus }) => $selectedCorpora.includes(corpus))
			.sort((a, b) => $selectedCorpora.indexOf(a.corpus) - $selectedCorpora.indexOf(b.corpus))
			.map(({ corpus, normalizedFrequency, rawFrequency }) => {
				// The normalizedFrequency already includes the normalization factor from the server
				const frequency = $useNormalization
					? normalizedFrequency // Already normalized, don't multiply again
					: rawFrequency;
				const maxFreq = $useNormalization ? maxFrequency.normalized : maxFrequency.raw;

				// Ensure we don't divide by zero and handle undefined
				const width = maxFreq > 0 ? (frequency / maxFreq) * 100 : 0;

				console.log('Bar calculation:', {
					corpus,
					normalizedFrequency,
					rawFrequency,
					frequency,
					maxFreq,
					width,
					xOffset
				});

				const result = {
					corpus,
					width,
					xOffset
				};

				xOffset += width;
				return result;
			});
	}

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

	async function loadSentences(collocate: Collocate) {
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

<ul class="list-none p-0 space-y-0 mt-2">
	{#each sortedCollocates as collocate (`${collocate.n}-${collocate.p}-${collocate.v}`)}
		<li
			class="flex items-center justify-start p-0"
			use:tooltipAction={{
				getTooltipData: () => {
					const tooltipData: Record<string, string> = {};
					collocate.contributions?.forEach(({ corpus, normalizedFrequency, rawFrequency }) => {
						const value = $useNormalization
							? Number(normalizedFrequency).toFixed(2) // Use normalizedFrequency for normalized view
							: Math.round(rawFrequency).toString(); // Use rawFrequency for raw view
						tooltipData[corpus] = value;
					});
					return tooltipData;
				}
			}}
		>
			<details
				ontoggle={(e) => {
					if ((e.target as HTMLDetailsElement).open) {
						loadSentences(collocate);
					}
				}}
			>
				<summary class="list-none flex items-center justify-start p-0">
					<svg width="50" height="20" class="mr-2">
						{#each renderContributions(collocate) as { corpus, width, xOffset } (corpus)}
							<rect
								style="fill: {getSolidColor(corpus)}"
								height="20"
								width="{width}%"
								x="{xOffset}%"
							/>
						{/each}
					</svg>
					<span class="text-left font-medium">
						{searchType === 'verb' ? collocate.n : collocate.v}
					</span>
				</summary>
				<ul class="list-none p-0 space-y-0 mt-2">
					{#each sentencesMap[`${collocate.n}-${collocate.p}-${collocate.v}`] || [] as sentence (`${sentence.corpus}-${sentence.text}`)}
						<li class="my-1 p-1 rounded" style="background-color: {getColor(sentence.corpus)}">
							<span class="font-medium">{sentence.corpus}:</span>
							{#each sentenceSegments( sentence.text, [{ start: sentence.n_begin, end: sentence.n_end, type: 'noun' }, { start: sentence.p_begin, end: sentence.p_end, type: 'particle' }, { start: sentence.v_begin, end: sentence.v_end, type: 'verb' }] ) as segment, index (index)}
								{#if segment.className}
									<span class={segment.className}>{segment.text}</span>
								{:else}
									{segment.text}
								{/if}
							{/each}
						</li>
					{/each}
				</ul>
			</details>
		</li>
	{/each}
</ul>
