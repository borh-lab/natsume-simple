<script lang="ts">
	import type { Corpus, RankBy } from '$lib/api/types';

	let {
		corpora,
		selectedCorpusIds,
		rankBy,
		disabled,
		ontoggle,
		onrank
	}: {
		corpora: Corpus[];
		selectedCorpusIds: string[];
		rankBy: RankBy;
		disabled: boolean;
		ontoggle: (corpusId: string) => void | Promise<void>;
		onrank: (rankBy: RankBy) => void | Promise<void>;
	} = $props();
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
			{corpus.label}
		</label>
	{/each}
	<label class="inline-flex items-center gap-1">
		Rank
		<select
			class="rounded border bg-white px-2 py-1 dark:border-gray-600 dark:bg-gray-800"
			value={rankBy}
			onchange={(event) => onrank((event.currentTarget as HTMLSelectElement).value as RankBy)}
		>
			<option value="meanPerMillion">Per million</option>
			<option value="raw">Raw count</option>
		</select>
	</label>
</fieldset>
