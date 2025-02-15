import { sveltekit } from "@sveltejs/kit/vite";
import Icons from "unplugin-icons/vite";
import { defineConfig } from "vitest/config";

export default defineConfig({
	plugins: [
		sveltekit(),
		Icons({
			compiler: "svelte",
		}),
	],
	test: {
		include: ["src/**/*.{test,spec}.{js,ts}"],
	},
	build: {
		sourcemap: true, // Enable source maps
	},
	server: {
		fs: {
			allow: ["./tailwind.config.js"],
		},
	},
});
