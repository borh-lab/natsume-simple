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
	const countLabel = $derived(`${count.toLocaleString()} ${count === 1 ? 'match' : 'matches'}`);
	const pattern = $derived(
		result?.input.pos === 'verb'
			? `noun–particle–“${result.input.term}”`
			: `“${result?.input.term ?? ''}”–particle–verb`
	);
	const identity = $derived(result ? `${countLabel} · ${pattern}` : '');
	const accessibleIdentity = $derived(
		result
			? `${countLabel}; searched ${result.input.pos} “${result.input.term}”; pattern noun, particle, verb`
			: ''
	);
</script>

{#if result}
	<div class="flex min-w-0 flex-1 items-center gap-2 text-sm" aria-live="polite">
		<p
			class="min-w-0 truncate text-gray-600 dark:text-gray-300"
			title={identity}
			aria-label={accessibleIdentity}
		>
			{countLabel} · {#if result.input.pos === 'verb'}noun–particle–<strong
					class="font-semibold text-gray-800 dark:text-gray-100">“{result.input.term}”</strong
				>{:else}<strong class="font-semibold text-gray-800 dark:text-gray-100"
					>“{result.input.term}”</strong
				>–particle–verb{/if}
		</p>
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
