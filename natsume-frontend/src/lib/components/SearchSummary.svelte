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
	const direction = $derived(result?.input.pos === 'verb' ? 'Verb' : 'Noun');
	const identity = $derived(
		result ? `${count.toLocaleString()} matches · “${result.input.term}” · ${direction}` : ''
	);
</script>

{#if result}
	<div class="flex min-w-0 flex-1 items-center gap-2 text-sm" aria-live="polite">
		<p class="min-w-0 truncate text-gray-600 dark:text-gray-300" title={identity}>{identity}</p>
		{#if draftDiffers}
			<span
				class="shrink-0 rounded border border-gray-400 bg-gray-100 px-1.5 py-0.5 font-medium text-gray-700 dark:border-gray-600 dark:bg-gray-800 dark:text-gray-200"
				>Not applied</span
			>
		{:else if stale}
			<span
				class="shrink-0 rounded border border-gray-400 bg-gray-100 px-1.5 py-0.5 font-medium text-gray-700 dark:border-gray-600 dark:bg-gray-800 dark:text-gray-200"
				>Previous result</span
			>
		{/if}
	</div>
{/if}
