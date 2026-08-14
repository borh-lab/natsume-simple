<script lang="ts">
	import type { SearchPosition, Suggestion } from '$lib/api/types';

	let {
		term = $bindable(),
		pos = $bindable(),
		loading,
		onsubmit,
		findSuggestions
	}: {
		term: string;
		pos: SearchPosition;
		loading: boolean;
		onsubmit: () => void | Promise<void>;
		findSuggestions: (query: string, pos: SearchPosition) => Promise<Suggestion[]>;
	} = $props();

	let suggestions = $state<Suggestion[]>([]);
	let open = $state(false);
	let active = $state(-1);
	let focusedWithin = $state(false);
	let dismissedQuery: string | null = null;
	let requestGeneration = 0;

	$effect(() => {
		const query = term.trim();
		const position = pos;
		const generation = ++requestGeneration;
		if (!query) {
			suggestions = [];
			open = false;
			return;
		}
		const timer = setTimeout(async () => {
			try {
				const found = await findSuggestions(query, position);
				if (generation !== requestGeneration) return;
				suggestions = found;
				open = dismissedQuery !== query && focusedWithin && suggestions.length > 0;
				active = -1;
			} catch {
				if (generation !== requestGeneration) return;
				suggestions = [];
				open = false;
			}
		}, 300);
		return () => clearTimeout(timer);
	});

	function dismissAutocomplete() {
		dismissedQuery = term.trim();
		requestGeneration += 1;
		open = false;
		active = -1;
	}

	function choose(suggestion: Suggestion) {
		term = suggestion.lemma;
		dismissedQuery = suggestion.lemma.trim();
		requestGeneration += 1;
		open = false;
		active = -1;
		onsubmit();
	}

	function focusout(event: FocusEvent & { currentTarget: HTMLFormElement }) {
		const next = event.relatedTarget;
		if (!(next instanceof Node) || !event.currentTarget.contains(next)) {
			focusedWithin = false;
			open = false;
		}
	}

	function keydown(event: KeyboardEvent) {
		if (event.key === 'Escape') {
			dismissAutocomplete();
			return;
		}
		if (!open || suggestions.length === 0) return;
		if (event.key === 'ArrowDown') {
			event.preventDefault();
			active = (active + 1) % suggestions.length;
		} else if (event.key === 'ArrowUp') {
			event.preventDefault();
			active = (active - 1 + suggestions.length) % suggestions.length;
		} else if (event.key === 'Enter' && active >= 0) {
			event.preventDefault();
			choose(suggestions[active]);
		}
	}
</script>

<form
	class="flex flex-wrap items-center gap-2"
	onfocusin={() => (focusedWithin = true)}
	onfocusout={focusout}
	onsubmit={(event) => {
		event.preventDefault();
		dismissAutocomplete();
		onsubmit();
	}}
>
	<label class="sr-only" for="search-position">Search direction</label>
	<select
		id="search-position"
		class="h-10 rounded border bg-white px-2 dark:border-gray-600 dark:bg-gray-800"
		bind:value={pos}
	>
		<option value="noun">Noun-Particle Collocations</option>
		<option value="verb">Verb-Particle Collocations</option>
	</select>
	<div class="relative">
		<label class="sr-only" for="search-input">Search term</label>
		<input
			id="search-input"
			name="search-input"
			type="search"
			class="h-10 rounded border bg-white px-3 dark:border-gray-600 dark:bg-gray-800"
			placeholder="Search term"
			autocomplete="off"
			role="combobox"
			aria-expanded={open}
			aria-controls="search-suggestions"
			aria-activedescendant={active >= 0 ? `suggestion-${active}` : undefined}
			bind:value={term}
			oninput={() => (dismissedQuery = null)}
			onkeydown={keydown}
			onfocus={() => (open = dismissedQuery !== term.trim() && suggestions.length > 0)}
		/>
		{#if open}
			<ul
				id="search-suggestions"
				role="listbox"
				class="absolute z-20 mt-1 max-h-64 min-w-full overflow-y-auto rounded border bg-white shadow dark:border-gray-600 dark:bg-gray-800"
			>
				{#each suggestions as suggestion, index (`${suggestion.pos}-${suggestion.lemma}`)}
					<li
						id={`suggestion-${index}`}
						role="option"
						aria-selected={index === active}
						class:font-bold={index === active}
					>
						<button
							class="w-full px-3 py-2 text-left"
							type="button"
							onclick={() => choose(suggestion)}
						>
							{suggestion.lemma}
						</button>
					</li>
				{/each}
			</ul>
		{/if}
	</div>
	<button
		type="submit"
		class="h-10 rounded bg-red-700 px-4 font-bold text-white hover:bg-red-600 disabled:opacity-60"
		disabled={loading}
	>
		{loading ? 'Searching…' : 'Go'}
	</button>
</form>
