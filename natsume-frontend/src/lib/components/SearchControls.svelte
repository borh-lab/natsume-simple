<script lang="ts">
	import type { SearchPosition, Suggestion } from '$lib/api/types';
	import { ROLE_TEXT_CLASSES } from '$lib/presentation/colors';

	let {
		term = $bindable(),
		pos = $bindable(),
		loading,
		dirty,
		onsubmit,
		findSuggestions
	}: {
		term: string;
		pos: SearchPosition;
		loading: boolean;
		dirty: boolean;
		onsubmit: () => void | Promise<void>;
		findSuggestions: (query: string, pos: SearchPosition) => Promise<Suggestion[]>;
	} = $props();

	let suggestions = $state<Suggestion[]>([]);
	let active = $state(-1);
	let focusedWithin = $state(false);
	let dismissedQuery = $state<string | null>(null);
	let requestGeneration = 0;
	let open = $derived(focusedWithin && suggestions.length > 0 && dismissedQuery !== term.trim());

	$effect(() => {
		const query = term.trim();
		const position = pos;
		const generation = ++requestGeneration;
		if (!query) {
			suggestions = [];
			return;
		}
		const timer = setTimeout(async () => {
			try {
				const found = await findSuggestions(query, position);
				if (generation !== requestGeneration) return;
				suggestions = found;
				active = -1;
			} catch {
				if (generation !== requestGeneration) return;
				suggestions = [];
			}
		}, 300);
		return () => clearTimeout(timer);
	});

	function dismissAutocomplete() {
		dismissedQuery = term.trim();
		requestGeneration += 1;
		active = -1;
	}

	function choose(suggestion: Suggestion) {
		term = suggestion.lemma;
		dismissedQuery = suggestion.lemma.trim();
		requestGeneration += 1;
		active = -1;
		onsubmit();
	}

	function focusout(event: FocusEvent & { currentTarget: HTMLFormElement }) {
		const next = event.relatedTarget;
		if (!(next instanceof Node) || !event.currentTarget.contains(next)) {
			focusedWithin = false;
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
	<fieldset class="flex h-10 rounded border border-gray-400 dark:border-gray-600">
		<legend class="sr-only">Search direction</legend>
		<label class="relative flex h-full">
			<input
				class="peer absolute inset-0 h-full w-full cursor-pointer appearance-none opacity-0"
				type="radio"
				name="search-position"
				value="noun"
				aria-label="Noun-particle collocations"
				bind:group={pos}
			/>
			<span
				class="flex h-full items-center gap-1 rounded-l px-2 text-sm peer-checked:bg-gray-200 peer-focus-visible:outline-2 peer-focus-visible:outline-offset-0 peer-focus-visible:outline-gray-900 dark:peer-checked:bg-gray-700 dark:peer-focus-visible:outline-gray-100"
			>
				<span class={ROLE_TEXT_CLASSES.noun} data-role="noun">Noun</span><span aria-hidden="true"
					>→</span
				><span class={ROLE_TEXT_CLASSES.particle} data-role="particle">Particle</span><span
					aria-hidden="true">→</span
				><span class={ROLE_TEXT_CLASSES.verb} data-role="verb">Verb</span>
			</span>
		</label>
		<label class="relative flex h-full border-l border-gray-400 dark:border-gray-600">
			<input
				class="peer absolute inset-0 h-full w-full cursor-pointer appearance-none opacity-0"
				type="radio"
				name="search-position"
				value="verb"
				aria-label="Verb-particle collocations"
				bind:group={pos}
			/>
			<span
				class="flex h-full items-center gap-1 rounded-r px-2 text-sm peer-checked:bg-gray-200 peer-focus-visible:outline-2 peer-focus-visible:outline-offset-0 peer-focus-visible:outline-gray-900 dark:peer-checked:bg-gray-700 dark:peer-focus-visible:outline-gray-100"
			>
				<span class={ROLE_TEXT_CLASSES.noun} data-role="noun">Noun</span><span aria-hidden="true"
					>←</span
				><span class={ROLE_TEXT_CLASSES.particle} data-role="particle">Particle</span><span
					aria-hidden="true">←</span
				><span class={ROLE_TEXT_CLASSES.verb} data-role="verb">Verb</span>
			</span>
		</label>
	</fieldset>
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
		class="h-10 rounded bg-gray-900 px-4 font-bold text-white hover:bg-gray-700 disabled:opacity-60 dark:bg-gray-100 dark:text-gray-900 dark:hover:bg-gray-300"
		disabled={loading}
	>
		{loading ? 'Searching…' : dirty ? 'Update results' : 'Go'}
	</button>
</form>
