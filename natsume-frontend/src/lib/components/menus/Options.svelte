<script>
const {
	useNormalization,
	selectedCorpora,
	getColor,
	getSolidColor,
	corpusNorm,
	handleCheckboxChange,
} = $props();
</script>

<div
	id="options-dropdown"
	class="origin-top-right absolute right-0 mt-2 w-64 rounded-md shadow-lg bg-white dark:bg-gray-800 ring-1 ring-black ring-opacity-5 focus:outline-none z-30"
>
	<div class="py-1">
		<!-- Existing Options -->
		<div class="px-4 py-2">
			<label class="flex items-center mb-2">
				<input
					type="checkbox"
					id="use-normalization"
					name="use-normalization"
					checked={$useNormalization}
					onchange={() => {
						$useNormalization = !$useNormalization;
						handleCheckboxChange();
					}}
					class="mr-2"
				/>
				<span class="dark:text-gray-300">Use Corpus Normalization</span>
			</label>
			<div class="mt-2">
				<span class="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2"
					>Corpus Selection</span
				>
				{#each Object.keys($corpusNorm) as corpus}
					<label
						class="flex items-center mb-2 px-2 py-1 rounded border-2"
						style="border-color: {getSolidColor(corpus)}; background-color: {getColor(corpus)}"
					>
						<input
							type="checkbox"
							id="corpus-{corpus}"
							name="corpus-{corpus}"
							bind:group={$selectedCorpora}
							value={corpus}
							onchange={handleCheckboxChange}
							class="mr-2"
						/>
						<span class="dark:text-white">{corpus}</span>
					</label>
				{/each}
			</div>
		</div>
	</div>
</div>
