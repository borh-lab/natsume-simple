<script lang="ts">
import CollocationList from "$lib/components/CollocationList.svelte";
import Search from "$lib/components/Search.svelte";
import ThemeSwitch from "$lib/components/ThemeSwitch.svelte";
import Options from "$lib/components/menus/Options.svelte";
import Stats from "$lib/components/menus/Stats.svelte";

import { afterUpdate, onMount, setContext, tick } from "svelte";
import { type Writable, derived, writable } from "svelte/store";
import "./../tailwind.css";
import Loading from "$lib/components/Loading.svelte";
import resolveConfig from "tailwindcss/resolveConfig";
import tailwindConfig from "../../tailwind.config.js";

// Icons
import ZondiconsCheveronDown from "~icons/zondicons/cheveron-down";

const twFullConfig = resolveConfig(tailwindConfig);

import { themeManager } from "$lib/theme.svelte";

function formatNumber(num: number): string {
	return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
}

import HorizontallyScrollableContainer from "$lib/components/HorizontallyScrollableContainer.svelte";
import type { CombinedResult, Result } from "$lib/query";

const apiUrl = import.meta.env.VITE_API_URL || "http://localhost:8000";
setContext("apiUrl", apiUrl);

const particles = ["が", "を", "に", "で", "から", "より", "と", "へ"];
// TODO: Convert to runes
const results = writable<Result[]>([]);

let searchType: "verb" | "noun" = "noun";
let searchTerm = "時間";

const lastSearchedNoun = writable("");
let corpusNorm: Record<string, number> = {};
let corpusSizes: string[] = [];
const resultCount = writable(0);
const useNormalization = writable(true);
const selectedCorpora = writable<string[]>([]);
const combinedSearch = writable(false);
const searchElapsedTime = writable(0);
let statsDropdownOpen = false;
let optionsDropdownOpen = false;

let isLoading = false;
let showMobileMenu = false;
let mobileDropdownOption: "select" | "stats" | "options" | null = null;

// TODO: Convert to runes
const d: Writable<Record<string, Result[]>> = writable({});

const filteredResultCount = derived(
	[d, selectedCorpora],
	([$d, $selectedCorpora]) => {
		if (!$d || Object.keys($d).length === 0) return 0;
		return Object.values($d).reduce((total, collocates) => {
			return (
				total +
				collocates.filter(
					(collocate) =>
						$selectedCorpora.includes(collocate.corpus) ||
						collocate.contributions?.some((c) =>
							$selectedCorpora.includes(c.corpus),
						),
				).length
			);
		}, 0);
	},
);

const highlightColors = [
	twFullConfig.theme.colors.red[200],
	twFullConfig.theme.colors.purple[200],
	twFullConfig.theme.colors.green[200],
	twFullConfig.theme.colors.blue[200],
	twFullConfig.theme.colors.yellow[200],
	twFullConfig.theme.colors.pink[200],
];

const highlightColorsDark = [
	twFullConfig.theme.colors.red[800],
	twFullConfig.theme.colors.purple[800],
	twFullConfig.theme.colors.green[800],
	twFullConfig.theme.colors.blue[800],
	twFullConfig.theme.colors.yellow[800],
	twFullConfig.theme.colors.pink[800],
];

const solidColors = [
	twFullConfig.theme.colors.red[500],
	twFullConfig.theme.colors.purple[500],
	twFullConfig.theme.colors.green[500],
	twFullConfig.theme.colors.blue[500],
	twFullConfig.theme.colors.yellow[500],
	twFullConfig.theme.colors.pink[500],
];

function getColor(corpus: string): string {
	const index = Object.keys(corpusNorm).indexOf(corpus);
	return themeManager.isDarkMode
		? highlightColorsDark[index % highlightColorsDark.length]
		: highlightColors[index % highlightColors.length];
}

function getSolidColor(corpus: string): string {
	const index = Object.keys(corpusNorm).indexOf(corpus);
	return solidColors[index % solidColors.length];
}

function renderContributions(
	collocate: Result,
	collocates: (Result | CombinedResult)[],
	selectedCorpora: string[],
	useNormalization: boolean,
): Array<{ corpus: string; width: number; xOffset: number; value: number }> {
	let xOffset = 0;

	const contributions =
		collocate.contributions && collocate.contributions.length > 0
			? collocate.contributions
			: [{ corpus: collocate.corpus, frequency: collocate.frequency }];

	return contributions
		.filter(({ corpus }) => selectedCorpora.includes(corpus))
		.sort(
			(a, b) =>
				selectedCorpora.indexOf(a.corpus) - selectedCorpora.indexOf(b.corpus),
		)
		.map(({ corpus, frequency }) => {
			const width =
				((frequency * (useNormalization ? corpusNorm[corpus] : 1)) /
					getMax(collocates, useNormalization)) *
				100;
			const value = useNormalization
				? frequency * corpusNorm[corpus]
				: frequency;
			const item = { corpus, width, xOffset, value };
			xOffset += width;
			return item;
		});
}

$: {
	console.log("selectedCorpora changed:", $selectedCorpora);
}

function renderContributionsCombined(
	collocate: CombinedResult,
	collocates: (Result | CombinedResult)[],
	selectedCorpora: string[],
	useNormalization: boolean,
): Array<{ corpus: string; width: number; xOffset: number; value: number }> {
	let xOffset = 0;
	return collocate.contributions
		.filter(({ corpus }) => selectedCorpora.includes(corpus))
		.sort(
			(a, b) =>
				selectedCorpora.indexOf(a.corpus) - selectedCorpora.indexOf(b.corpus),
		)
		.map(({ corpus, frequency }) => {
			const width =
				((frequency * (useNormalization ? corpusNorm[corpus] : 1)) /
					getMax(collocates, useNormalization)) *
				100;
			const value = useNormalization
				? frequency * corpusNorm[corpus]
				: frequency;
			const result = { corpus, width, xOffset, value };
			xOffset += width;
			return result;
		});
}
function calculateCorpusDistribution(
	collocates: (Result | CombinedResult)[],
	useNormalization: boolean,
	corpusNorm: Record<string, number>,
): Record<string, number> {
	console.log("Calculating corpus distribution", collocates);
	const distribution: Record<string, number> = {};
	for (const collocate of collocates) {
		if (collocate.contributions && collocate.contributions.length > 0) {
			// Combined search mode
			for (const { corpus, frequency } of collocate.contributions) {
				distribution[corpus] =
					(distribution[corpus] || 0) +
					frequency *
						(useNormalization && corpusNorm[corpus] ? corpusNorm[corpus] : 1);
			}
		} else if ("corpus" in collocate) {
			// Non-combined search mode
			const corpus = collocate.corpus;
			distribution[corpus] =
				(distribution[corpus] || 0) +
				collocate.frequency *
					(useNormalization && corpusNorm[corpus] ? corpusNorm[corpus] : 1);
		}
	}
	console.log("Corpus distribution result:", distribution);
	return distribution;
}

function ParticleHeader({
	particle,
	collocates,
	maxFrequency,
	useNormalization,
	corpusNorm,
	totalFrequency,
	selectedCorpora,
}: {
	particle: string;
	collocates: (Result | CombinedResult)[];
	maxFrequency: number;
	useNormalization: boolean;
	corpusNorm: Record<string, number>;
	totalFrequency: number;
	selectedCorpora: string[];
}) {
	console.log("ParticleHeader called with:", {
		particle,
		collocates,
		selectedCorpora,
	});
	const distribution = calculateCorpusDistribution(
		collocates,
		useNormalization,
		corpusNorm,
	);
	const particleTotal = Object.values(distribution).reduce(
		(sum, value) => sum + value,
		0,
	);

	let cumulativeFrequency = 0;
	const distributionWithOffsets = Object.entries(distribution)
		.filter(([corpus]) => selectedCorpora.includes(corpus))
		.map(([corpus, frequency]) => {
			const width = totalFrequency > 0 ? (frequency / totalFrequency) * 100 : 0;
			const xOffset =
				totalFrequency > 0 ? (cumulativeFrequency / totalFrequency) * 100 : 0;
			cumulativeFrequency += frequency;
			return { corpus, frequency, width, xOffset };
		});

	console.log("ParticleHeader result:", {
		particle,
		distribution,
		particleTotal,
		distributionWithOffsets,
	});
	return {
		particle,
		distribution,
		particleTotal,
		distributionWithOffsets,
		maxFrequency,
	};
}

onMount(() => {
	const fetchData = async () => {
		try {
			const response = await fetch(`${apiUrl}/corpus/norm`);
			corpusNorm = await response.json();
			corpusSizes = Object.entries(corpusNorm).map(
				([corpus, size]) => `${corpus}: ${size.toFixed(2)}`,
			);
			selectedCorpora.set(Object.keys(corpusNorm));

			// Call getNPVs to fetch initial data
			await performSearch();
		} catch (error) {
			console.error("Error fetching corpus norm:", error);
		}

		updateScrollButtonsVisibility();
	};

	fetchData();

	window.addEventListener("resize", updateScrollButtonsVisibility);
	window.addEventListener("scroll", updateScrollButtonsVisibility);

	if (mainScrollContainer) {
		console.log("Main scroll container properties:");
		console.log("scrollWidth:", mainScrollContainer.scrollWidth);
		console.log("clientWidth:", mainScrollContainer.clientWidth);
		console.log("offsetWidth:", mainScrollContainer.offsetWidth);
		console.log("style.overflowX:", mainScrollContainer.style.overflowX);
	}

	return () => {
		window.removeEventListener("resize", updateScrollButtonsVisibility);
		window.removeEventListener("scroll", updateScrollButtonsVisibility);
	};
});

// TODO: Extract to a separate file
function computeDerivedData(
	results: Result[],
	useNormalization: boolean,
	selectedCorpora: string[],
	combinedSearch: boolean,
) {
	console.log("Computing derived data");
	let derivedData: Record<string, CombinedResult[]>;

	const appendedResults = results.map((result) => ({
		...result,
		n: searchTerm,
	}));

	if (combinedSearch) {
		// Combine results across all particles first
		const combinedResults = appendedResults.reduce<
			Record<string, CombinedResult>
		>((acc, curr) => {
			if (selectedCorpora.includes(curr.corpus as string)) {
				const key = `${curr.v}-${curr.p}`; // Use both verb and particle as key
				if (!acc[key]) {
					acc[key] = {
						n: searchTerm,
						v: curr.v,
						p: curr.p,
						frequency: 0,
						contributions: [],
					};
				}
				acc[key].frequency +=
					curr.frequency * (useNormalization ? corpusNorm[curr.corpus] : 1);
				acc[key].contributions.push({
					corpus: curr.corpus as string,
					frequency: curr.frequency,
				});
			}
			return acc;
		}, {});

		// Then separate by particle
		derivedData = Object.fromEntries(
			particles.map((particle) => [
				particle,
				Object.values(combinedResults)
					.filter((d: CombinedResult) => d.p === particle)
					.sort((a, b) => b.frequency - a.frequency),
			]),
		);
	} else {
		// Non-combined logic remains the same
		derivedData = Object.fromEntries(
			particles.map((particle) => [
				particle,
				appendedResults
					.filter(
						(d: { p: string; corpus: string }) =>
							d.p === particle && selectedCorpora.includes(d.corpus),
					)
					.map((d: Result) => ({ ...d, contributions: d.contributions || [] }))
					.sort((a: Result, b: Result): number => {
						const aValue =
							a.frequency * (useNormalization ? corpusNorm[a.corpus] : 1);
						const bValue =
							b.frequency * (useNormalization ? corpusNorm[b.corpus] : 1);
						return bValue - aValue;
					}),
			]),
		);
	}
	return derivedData;
}

async function updateDerivedData() {
	const derivedData = computeDerivedData(
		$results,
		$useNormalization,
		$selectedCorpora,
		$combinedSearch,
	);
	d.set(derivedData as Record<string, Result[]>);
	await tick(); // Wait for the next DOM update
}

async function handleCheckboxChange() {
	isLoading = true;
	await updateDerivedData();
	isLoading = false;
}

async function performSearch(): Promise<void> {
	console.log({ d: $d });
	console.log("performSearch");
	try {
		isLoading = true; // Set loading state
		const startTime = performance.now();

		// Construct API endpoint based on search type
		const endpoint =
			searchType === "verb"
				? `/npv/verb/${searchTerm}`
				: `/npv/noun/${searchTerm}`;
		const response = await fetch(`${apiUrl}${endpoint}`);
		const data = await response.json();

		const endTime = performance.now();
		searchElapsedTime.set((endTime - startTime) / 1000); // Calculate elapsed time
		results.set(data as Result[]);
		resultCount.set(data.length);
		lastSearchedNoun.set(searchTerm); // Update the last searched term

		await updateDerivedData(); // Update derived data

		console.log({ d: $d });
	} catch (error) {
		console.error("Error fetching results:", error);
	} finally {
		isLoading = false; // Turn off loading state
	}
}

function getMax(
	collocates: (Result | CombinedResult)[],
	useNormalization: boolean,
): number {
	return Math.max(
		...collocates.map((x) => {
			if (x.contributions && x.contributions.length > 0) {
				return Math.max(
					...x.contributions.map(
						(c) => c.frequency * (useNormalization ? corpusNorm[c.corpus] : 1),
					),
				);
			}

			return (
				x.frequency *
				(useNormalization && "corpus" in x ? corpusNorm[x.corpus] : 1)
			);
		}),
		0.0,
	);
}

function showTooltip(
	event: { pageX: number; pageY: number },
	text: string | null,
) {
	const tooltip = document.getElementById("tooltip");
	if (tooltip) {
		tooltip.textContent = text;
		tooltip.style.left = `${event.pageX + 10}px`;
		tooltip.style.top = `${event.pageY + 10}px`;
		tooltip.classList.remove("hidden");
	}
}

function hideTooltip() {
	const tooltip = document.getElementById("tooltip");
	if (tooltip) {
		tooltip.classList.add("hidden");
	}
}

function tooltipAction(
	node: HTMLLIElement,
	{
		getTooltipData,
		useNormalization,
	}: {
		getTooltipData: () => Record<string, number>;
		useNormalization: boolean;
	},
) {
	const handleMouseover = (event: MouseEvent) => {
		const tooltipData = getTooltipData();
		const tooltipText = Object.entries(tooltipData)
			.map(([corpus, value]) => {
				const formattedValue = useNormalization
					? value.toFixed(2)
					: Math.round(value).toString();
				return `${corpus}: ${formattedValue}`;
			})
			.join(", ");
		showTooltip(event, tooltipText);
	};
	const handleMouseout = hideTooltip;

	node.addEventListener("mouseover", handleMouseover);
	node.addEventListener("mouseout", handleMouseout);

	return {
		destroy() {
			node.removeEventListener("mouseover", handleMouseover);
			node.removeEventListener("mouseout", handleMouseout);
		},
	};
}

const columnWidth = 200; // Set a fixed width for all columns
const columnSpacing = 4; // Set a fixed spacing between columns

let headerScrollContainer: HTMLElement;
let mainScrollContainer: HTMLElement;

function syncScroll(event: Event) {
	const scrollingElement = event.target as HTMLElement;
	if (scrollingElement === headerScrollContainer) {
		mainScrollContainer.scrollLeft = scrollingElement.scrollLeft;
	} else if (scrollingElement === mainScrollContainer) {
		headerScrollContainer.scrollLeft = scrollingElement.scrollLeft;
	}
}

function scrollOneColumn(direction: "left" | "right") {
	console.log(`Scrolling ${direction}`);
	if (mainScrollContainer) {
		const currentScroll = mainScrollContainer.scrollLeft;
		const scrollAmount = columnWidth + columnSpacing;
		const newScroll =
			direction === "left"
				? Math.max(0, currentScroll - scrollAmount)
				: currentScroll + scrollAmount;

		console.log(
			`Before scroll - Current scroll: ${currentScroll}, Target scroll: ${newScroll}`,
		);

		requestAnimationFrame(() => {
			mainScrollContainer.scrollLeft = newScroll;

			// Force a reflow to ensure the scroll has been applied
			void mainScrollContainer.offsetWidth;

			console.log(
				`After scroll - New scroll position: ${mainScrollContainer.scrollLeft}`,
			);
		});
	} else {
		console.log("mainScrollContainer is not defined");
	}
}

function updateScrollButtonsVisibility() {
	if (mainScrollContainer) {
		const hasOverflow =
			mainScrollContainer.scrollWidth > mainScrollContainer.clientWidth;
		const leftButton = document.querySelector(".left-button") as HTMLElement;
		const rightButton = document.querySelector(".right-button") as HTMLElement;

		if (leftButton && rightButton) {
			const scrollLeft = mainScrollContainer.scrollLeft;
			const maxScrollLeft =
				mainScrollContainer.scrollWidth - mainScrollContainer.clientWidth;

			const containerRect = mainScrollContainer.getBoundingClientRect();
			const viewportHeight = window.innerHeight;
			const containerVisibleTop = Math.max(0, containerRect.top);
			const containerVisibleBottom = Math.min(
				viewportHeight,
				containerRect.bottom,
			);
			const visibleCenterY =
				containerVisibleTop +
				(containerVisibleBottom - containerVisibleTop) / 2;

			leftButton.style.display =
				hasOverflow && scrollLeft > 0 ? "flex" : "none";
			rightButton.style.display =
				hasOverflow && scrollLeft < maxScrollLeft ? "flex" : "none";

			// Adjust the top position to account for both sticky headers
			const topHeader = document.querySelector("header:first-of-type");
			const particlesHeader = document.querySelector("header:nth-of-type(2)");
			const totalHeaderHeight =
				(topHeader?.clientHeight || 0) + (particlesHeader?.clientHeight || 0);
			const buttonTop = Math.max(visibleCenterY, totalHeaderHeight + 20); // 20px padding

			leftButton.style.top = `${buttonTop}px`;
			rightButton.style.top = `${buttonTop}px`;
		}
	}
}

function handleClickOutside(event: MouseEvent) {
	const statsDropdown = document.querySelector("#stats-dropdown");
	const optionsDropdown = document.querySelector("#options-dropdown");
	const statsButton = document.querySelector("#stats-button");
	const optionsButton = document.querySelector("#options-button");

	if (
		statsDropdown &&
		!statsDropdown.contains(event.target as Node) &&
		!statsButton?.contains(event.target as Node)
	) {
		statsDropdownOpen = false;
	}
	if (
		optionsDropdown &&
		!optionsDropdown.contains(event.target as Node) &&
		!optionsButton?.contains(event.target as Node)
	) {
		optionsDropdownOpen = false;
	}
}

afterUpdate(() => {
	updateScrollButtonsVisibility();
});
</script>

<div class="flex flex-col min-h-screen bg-white dark:bg-gray-900 text-gray-900 dark:text-gray-100">
	<!-- Fixed top header with search bar and info dropdown -->
	<header class="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 p-2 z-20">
		<div
			class="container mx-auto flex flex-col md:flex-row justify-center md:justify-between items-center space-y-2 md:space-y-0"
		>
			<div class="flex justify-center">
				<h1
					class="text-xl text-red-600 dark:text-red-400 font-bold mr-2 whitespace-nowrap flex items-center gap-[0.3em]"
				>
					<img src="/favicon.png" class="h-6 w-6 inline-block" alt="Natsume logo" />
					<span class="hidden lg:inline">Natsume Simple</span>
				</h1>

				<!-- Search bar on small screen (next to logo) -->
				<div class="lg:hidden flex-grow mx-2">
					<div class="flex">
						<select
							class="border rounded-l h-8 dark:bg-gray-700 dark:text-white dark:border-gray-600"
							bind:value={searchType}
						>
							<option value="verb">Verb</option>
							<option value="noun">Noun</option>
						</select>
						<input
							type="text"
							class="border flex-grow h-8 px-2 text-sm dark:bg-gray-700 dark:text-white dark:border-gray-600"
							bind:value={searchTerm}
							placeholder="Search term"
						/>
						<button
							class="bg-red-700 hover:bg-red-500 text-white font-bold px-2 rounded-r h-8 text-sm dark:bg-red-600 dark:hover:bg-red-500"
							onclick={performSearch}
							disabled={isLoading}
						>
							{isLoading ? 'Searching...' : 'Go'}
						</button>
					</div>
				</div>

				<!-- The hamburger menu on small screen-->
				<button
					class="ml-2 block lg:hidden focus:outline-none"
					onclick={() => (showMobileMenu = !showMobileMenu)}
				>
					<!-- A simple icon（3 lines now）-->
					<div class="w-5 h-1 bg-gray-700 dark:bg-gray-300 mb-1"></div>
					<div class="w-5 h-1 bg-gray-700 dark:bg-gray-300 mb-1"></div>
					<div class="w-5 h-1 bg-gray-700 dark:bg-gray-300"></div>
				</button>
			</div>

			<!--Search bar and button on big screen -->
			<div class="hidden lg:flex justify-center">
				<Search bind:searchTerm {performSearch} {isLoading} bind:searchType />
			</div>

			<!--Stats and options dropdown on big screen -->
			<div class="hidden lg:flex justify-center items-center space-x-2">
				<div class="relative inline-block text-left">
					<div class="flex space-x-2">
						<button
							id="stats-button"
							class="gap-1 bg-gray-200 hover:bg-gray-300 text-gray-800 font-bold py-1 px-2 rounded inline-flex items-center text-sm dark:bg-gray-700 dark:hover:bg-gray-600 dark:text-gray-200"
							onclick={() => (statsDropdownOpen = !statsDropdownOpen)}
						>
							<span>Stats</span>
							<ZondiconsCheveronDown />
						</button>
						<button
							id="options-button"
							class="gap-1 bg-gray-200 hover:bg-gray-300 text-gray-800 font-bold py-1 px-2 rounded inline-flex items-center text-sm dark:bg-gray-700 dark:hover:bg-gray-600 dark:text-gray-200"
							onclick={() => (optionsDropdownOpen = !optionsDropdownOpen)}
						>
							<span>Options</span>
							<ZondiconsCheveronDown />
						</button>
					</div>
					{#if statsDropdownOpen}
						<Stats {corpusNorm} {filteredResultCount} {searchElapsedTime} {formatNumber} />
					{/if}
					{#if optionsDropdownOpen}
						<Options
							{useNormalization}
							{combinedSearch}
							{selectedCorpora}
							{getColor}
							{getSolidColor}
							{corpusNorm}
							{handleCheckboxChange}
						/>
					{/if}
				</div>
				<ThemeSwitch />
			</div>
		</div>

		<!-- Mobile dropdown menu -->
		{#if showMobileMenu}
			<div
				class="block lg:hidden bg-white dark:bg-gray-800 border-t border-gray-200 dark:border-gray-700 mt-2 p-2"
			>
				<div class="space-y-2">
					<!-- Select dropdown -->
					<button
						class="w-full bg-gray-200 hover:bg-gray-300 py-2 px-4 rounded text-left dark:bg-gray-700 dark:hover:bg-gray-600"
						onclick={() =>
							(mobileDropdownOption = mobileDropdownOption === 'select' ? null : 'select')}
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
						onclick={() =>
							(mobileDropdownOption = mobileDropdownOption === 'stats' ? null : 'stats')}
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
						onclick={() =>
							(mobileDropdownOption = mobileDropdownOption === 'options' ? null : 'options')}
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
								{combinedSearch}
								{selectedCorpora}
								{getColor}
								{getSolidColor}
								{corpusNorm}
								{handleCheckboxChange}
							/>
						</div>
					{/if}

					<!-- Theme Switch -->
					<div
						class="w-full bg-gray-200 dark:bg-gray-700 py-2 px-4 rounded flex justify-between items-center"
					>
						<span>Dark Mode</span>
						<ThemeSwitch />
					</div>
				</div>
			</div>
		{/if}
	</header>

	<!-- Sticky particles header -->
	<header
		class="sticky top-0 bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 p-2 z-10"
	>
		<HorizontallyScrollableContainer {syncScroll} bind:scrollContainer={headerScrollContainer}>
			<div class="flex" style="gap: {columnSpacing}px;">
				{#if $d && Object.keys($d).length > 0}
					{#each Object.entries($d) as [particle, collocates]}
						{#if collocates && collocates.length > 0}
							<div style="width: {columnWidth}px; flex-shrink: 0;">
								<h2 class="text-center text-2xl font-semibold">{particle}</h2>
								<div class="h-1 w-full relative">
									{#each ParticleHeader( { particle, collocates, maxFrequency: getMax(collocates, $useNormalization), useNormalization: $useNormalization, corpusNorm, totalFrequency: collocates.reduce((sum, c) => sum + c.frequency, 0), selectedCorpora: $selectedCorpora } ).distributionWithOffsets as { corpus, width, xOffset }}
										<div
											class="absolute h-full"
											style="left: {xOffset}%; width: {width}%; background-color: {getSolidColor(
												corpus
											)};"
										></div>
									{/each}
								</div>
							</div>
						{/if}
					{/each}
				{/if}
			</div>
		</HorizontallyScrollableContainer>
	</header>

	<!-- Main content area -->
	<main class="flex-1 pt-0">
		<div class="container mx-auto p-0">
			<div class="relative">
				<button class="scroll-button left-button" onclick={() => scrollOneColumn('left')}>
					<span class="arrow">←</span>
				</button>
				<HorizontallyScrollableContainer
					{syncScroll}
					bind:scrollContainer={mainScrollContainer}
					style="width: 100%; max-width: 100vw;"
				>
					<div class="flex" style="gap: {columnSpacing}px;">
						{#if $d && Object.keys($d).length > 0}
							{#each Object.entries($d) as [particle, collocates]}
								{#if collocates && collocates.length > 0}
									<div style="width: {columnWidth}px; flex-shrink: 0;">
										<CollocationList
											{collocates}
											{combinedSearch}
											{getColor}
											{tooltipAction}
											{renderContributions}
											{renderContributionsCombined}
											{useNormalization}
											{selectedCorpora}
											{getSolidColor}
											{corpusNorm}
										/>
									</div>
								{/if}
							{/each}
						{:else}
							<p class="text-center text-gray-600 mt-4">
								{#if $results.length > 0}
									Showing results for {searchType === 'verb' ? 'Verb' : 'Noun'}: "{searchTerm}"
								{:else}
									No results found for {searchType === 'verb' ? 'Verb' : 'Noun'}: "{searchTerm}"
								{/if}
							</p>
						{/if}
					</div>
				</HorizontallyScrollableContainer>
				<button class="scroll-button right-button" onclick={() => scrollOneColumn('right')}>
					<span class="arrow">→</span>
				</button>
			</div>
		</div>
	</main>
</div>

{#if isLoading}
	<Loading />
{/if}

<div
	id="tooltip"
	class="hidden absolute bg-gray-700 dark:bg-gray-200 text-white dark:text-gray-800 text-sm p-2 rounded"
></div>

<svelte:window on:click={handleClickOutside} />

<style>
	#tooltip {
		pointer-events: none;
		z-index: 20;
	}

	.scroll-button {
		position: fixed;
		width: 30px;
		height: 30px;
		background-color: rgba(0, 0, 0, 0.5);
		color: white;
		border: none;
		border-radius: 50%;
		display: none;
		align-items: center;
		justify-content: center;
		cursor: pointer;
		z-index: 10;
		transition: background-color 0.3s ease;
	}

	.scroll-button:hover {
		background-color: rgba(0, 0, 0, 0.7);
	}

	.left-button {
		left: 10px;
	}

	.right-button {
		right: 10px;
	}

	.arrow {
		font-size: 18px;
		line-height: 1;
	}

	.relative {
		position: relative;
	}
</style>
