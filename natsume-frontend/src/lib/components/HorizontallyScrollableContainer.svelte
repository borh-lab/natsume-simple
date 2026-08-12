<script lang="ts">
	import type { Snippet } from 'svelte';

	let {
		children,
		syncScroll,
		scrollContainer = $bindable(),
		style = ''
	}: {
		children: Snippet;
		syncScroll: (e: Event) => void;
		scrollContainer: HTMLElement;
		style?: string;
	} = $props();
</script>

<div
	class="overflow-x-auto hide-scrollbar container mx-auto p-0"
	bind:this={scrollContainer}
	onscroll={syncScroll}
	{style}
>
	{@render children()}
</div>

<style>
	.hide-scrollbar {
		-ms-overflow-style: none; /* IE and Edge */
		scrollbar-width: none; /* Firefox */
	}
	.hide-scrollbar::-webkit-scrollbar {
		display: none; /* Chrome, Safari and Opera */
	}

	.overflow-x-auto {
		overflow-x: auto !important;
		-webkit-overflow-scrolling: touch;
	}
</style>
