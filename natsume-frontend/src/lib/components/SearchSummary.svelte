<script lang="ts">
	import type { CollocationsResponse } from '$lib/api/types';
	let { result, stale }: { result: CollocationsResponse | null; stale: boolean } = $props();
	const count = $derived(
		result?.particleGroups.reduce((sum, group) => sum + group.returnedCount, 0) ?? 0
	);
</script>

{#if result}
	<p class="text-sm text-gray-600 dark:text-gray-300" aria-live="polite">
		{count.toLocaleString()} results
		{#if stale}<strong class="ml-2 text-amber-700 dark:text-amber-300">Previous result</strong>{/if}
	</p>
{/if}
