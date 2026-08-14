<script lang="ts">
	import type { VisibleSearchResult } from '$lib/search/controller.svelte';
	let {
		result,
		stale,
		draftDiffers
	}: { result: VisibleSearchResult | null; stale: boolean; draftDiffers: boolean } = $props();
	const count = $derived(
		result?.response.particleGroups.reduce(
			(sum, group) => sum + group.totalMatchingCollocations,
			0
		) ?? 0
	);
	const direction = $derived(
		result?.input.pos === 'verb' ? 'Verb–particle search' : 'Noun–particle search'
	);
</script>

{#if result}
	<div class="space-y-1 text-sm" aria-live="polite">
		<p class="text-gray-600 dark:text-gray-300">
			{count.toLocaleString()} matching results for “{result.input.term}” · {direction}
		</p>
		{#if draftDiffers}
			<p class="font-medium text-amber-700 dark:text-amber-300">
				Controls changed — update results to apply them.
			</p>
		{:else if stale}
			<p class="font-medium text-amber-700 dark:text-amber-300">Previous result</p>
		{/if}
	</div>
{/if}
