<script lang="ts">
import { useNormalization, selectedCorpora } from '$lib/stores/corpus';
import type { DropdownOption } from '$lib/types';
import ThemeSwitch from './ThemeSwitch.svelte';
import Stats from './menus/Stats.svelte';
import Options from './menus/Options.svelte';

let {
  searchType = $bindable<'verb' | 'noun'>('noun'),
  mobileDropdownOption = $bindable<DropdownOption>(null),
  corpusNorm,
  filteredResultCount,
  searchElapsedTime,
  formatNumber,
  getColor,
  getSolidColor,
  handleCheckboxChange
} = $props();
</script>

<div class="block lg:hidden bg-white dark:bg-gray-800 border-t border-gray-200 dark:border-gray-700 mt-2 p-2">
  <div class="space-y-2">
    <!-- Select dropdown -->
    <button
      class="w-full bg-gray-200 hover:bg-gray-300 py-2 px-4 rounded text-left dark:bg-gray-700 dark:hover:bg-gray-600"
      onclick={() => mobileDropdownOption = mobileDropdownOption === 'select' ? null : 'select'}
    >
      Select Search Type
      <span class="float-right">
        {mobileDropdownOption === 'select' ? '▲' : '▼'}
      </span>
    </button>
    {#if mobileDropdownOption === 'select'}
      <div class="p-2 bg-gray-100 dark:bg-gray-700 rounded">
        <select
          class="border rounded h-8 w-full dark:bg-gray-600 dark:text-white dark:border-gray-500"
          bind:value={searchType}
        >
          <option value="verb">Verb (Noun-Particle-Verb)</option>
          <option value="noun">Noun (Noun-Particle-Verb)</option>
        </select>
      </div>
    {/if}

    <!-- Stats dropdown -->
    <button
      class="w-full bg-gray-200 hover:bg-gray-300 py-2 px-4 rounded text-left dark:bg-gray-700 dark:hover:bg-gray-600"
      onclick={() => mobileDropdownOption = mobileDropdownOption === 'stats' ? null : 'stats'}
    >
      Stats
      <span class="float-right">
        {mobileDropdownOption === 'stats' ? '▲' : '▼'}
      </span>
    </button>
    {#if mobileDropdownOption === 'stats'}
      <div class="p-2 bg-gray-100 dark:bg-gray-700 rounded">
        <Stats {corpusNorm} {filteredResultCount} {searchElapsedTime} {formatNumber} />
      </div>
    {/if}

    <!-- Options dropdown -->
    <button
      class="w-full bg-gray-200 hover:bg-gray-300 py-2 px-4 rounded text-left dark:bg-gray-700 dark:hover:bg-gray-600"
      onclick={() => mobileDropdownOption = mobileDropdownOption === 'options' ? null : 'options'}
    >
      Options
      <span class="float-right">
        {mobileDropdownOption === 'options' ? '▲' : '▼'}
      </span>
    </button>
    {#if mobileDropdownOption === 'options'}
      <div class="p-2 bg-gray-100 dark:bg-gray-700 rounded">
        <Options
          {useNormalization}
          {selectedCorpora}
          {getColor}
          {getSolidColor}
          {corpusNorm}
          {handleCheckboxChange}
        />
      </div>
    {/if}

    <!-- Theme Switch -->
    <div class="w-full bg-gray-200 dark:bg-gray-700 py-2 px-4 rounded flex justify-between items-center">
      <span>Dark Mode</span>
      <ThemeSwitch />
    </div>
  </div>
</div>
