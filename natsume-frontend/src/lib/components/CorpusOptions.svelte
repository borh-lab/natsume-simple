<script lang="ts">
	import type { Corpus } from '$lib/api/types';
	import { CORPUS_COLORS, corpusColorMap } from '$lib/presentation/search';

	let {
		corpora,
		selectedCorpusIds,
		disabled,
		ontoggle
	}: {
		corpora: Corpus[];
		selectedCorpusIds: string[];
		disabled: boolean;
		ontoggle: (corpusId: string) => void | Promise<void>;
	} = $props();
	const colorSlots = $derived(corpusColorMap(corpora));
</script>

<fieldset class="flex flex-wrap items-center gap-x-4 gap-y-2" {disabled}>
	<legend class="sr-only">Search options</legend>
	{#each corpora as corpus (corpus.id)}
		<label class="inline-flex items-center gap-1">
			<input
				id={`corpus-${corpus.id}`}
				type="checkbox"
				checked={selectedCorpusIds.includes(corpus.id)}
				disabled={selectedCorpusIds.length === 1 && selectedCorpusIds.includes(corpus.id)}
				onchange={() => ontoggle(corpus.id)}
			/>
			<span
				class="h-2.5 w-2.5 rounded-sm"
				style:background-color={CORPUS_COLORS[colorSlots[corpus.id] % CORPUS_COLORS.length]}
				aria-hidden="true"
			></span>
			{corpus.label}
		</label>
	{/each}
</fieldset>
