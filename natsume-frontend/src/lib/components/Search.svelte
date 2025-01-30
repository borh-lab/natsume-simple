<script lang="ts">
import pkg from "lodash";
import { onMount } from "svelte";
import type { Writable } from "svelte/store";
import { writable } from "svelte/store";
import EosIconsLoading from "~icons/eos-icons/loading";
import MaterialSymbolsSearch from "~icons/material-symbols/search";

const { debounce } = pkg;

let {
	searchTerm = $bindable(""),
	searchType = $bindable<"verb" | "noun">("verb"),
	performSearch = () => {},
	isLoading = false,
} = $props();

// Reactive stores
const suggestions = writable<Array<{ word: string; pos: string }>>([]);
const showSuggestions = writable(false);

const apiUrl = import.meta.env.VITE_API_URL || "http://localhost:8000";
// Debounced search function，delay 300ms
const debouncedFetchSuggestions = debounce(fetchSuggestions, 300);
async function fetchSuggestions(query: string) {
	if (query.length === 0) {
		suggestions.set([]);
		showSuggestions.set(false);
		return;
	}
	try {
		const response = await fetch(
			`${apiUrl}/search/${encodeURIComponent(query)}`,
		);
		const data = await response.json();

		// Filter the proper part of speech
		const filteredSuggestions = data
			.filter((item: [string, string]) => {
				const [word, pos] = item;
				return pos === (searchType === "noun" ? "n" : "v");
			})
			.slice(0, 10) // Top 10 outcomes
			.map((item: [string, string]) => {
				const [word, pos] = item;
				return { word, pos };
			});
		suggestions.set(filteredSuggestions);
		showSuggestions.set(filteredSuggestions.length > 0);
	} catch (error) {
		console.error("Error fetching suggestions:", error);
		showSuggestions.set(false);
	}
}

$effect(() => {
	debouncedFetchSuggestions(searchTerm);
});

let containerElement: HTMLDivElement;

function handleClickOutside(event: MouseEvent) {
	if (!containerElement.contains(event.target as Node)) {
		showSuggestions.set(false);
	}
}

onMount(() => {
	document.addEventListener("click", handleClickOutside);
	return () => {
		document.removeEventListener("click", handleClickOutside);
	};
});
</script>

<div class="flex items-center space-x-1 relative" bind:this={containerElement}>
	<!-- Select searching pattern -->
	<select
    class="border rounded h-8 dark:bg-gray-700 dark:text-white dark:border-gray-600"
    bind:value={searchType}
  >
  <option value="verb">Verb-Particle Collocations</option>
  <option value="noun">Noun-Particle Collocations</option>
</select>

	<div class="relative inline-block">
	<input
	type="search"
	class="text-lg p-1 border rounded mr-2 h-8 w-40 sm:w-auto dark:bg-gray-700 dark:text-white dark:border-gray-600"
	id="search-input"
	name="search-input"
		bind:value={searchTerm}
		onclick={() => {
			if ($suggestions.length > 0) {
			  showSuggestions.set(true);
			}
		  }}
		onkeypress={(e) => {
			if (e.key === 'Enter') {
				performSearch();
				showSuggestions.set(false); // hide suggested word
			}
		}}
		placeholder={searchType === 'verb' ? 'Search Verb' : 'Search Noun'}
		autocomplete="off"
	/>

    {#if $showSuggestions}
      <ul class="absolute top-full left-0 bg-white border border-gray-300 min-w-full z-[1000] dark:bg-[#2d2d2d] dark:text-white">
        {#each $suggestions as suggestion}
          <li
            class="p-2 cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-600 dark:bg-gray-800"

          >
		  <button
		  	class="w-full  text-left"
		              onclick={() => {
              searchTerm = suggestion.word;
              showSuggestions.set(false);
              performSearch();
            }}
		  >
			      {suggestion.word}({suggestion.pos})
		  </button>

          </li>
        {/each}
      </ul>
    {/if}
  </div>

  <button
    class="bg-red-700 hover:bg-red-500 text-white font-bold py-1 px-2 rounded h-8 text-sm dark:bg-red-600 dark:hover:bg-red-500"
    onclick={() => performSearch()}
    disabled={isLoading}
  >
    {#if isLoading}
      <EosIconsLoading />
    {:else}
      <MaterialSymbolsSearch />
    {/if}
  </button>
</div>
