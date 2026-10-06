import adapter from '@sveltejs/adapter-static';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

/** @type {import('@sveltejs/kit').Config} */
const config = {
	// compilerOptions: {
	// 	enableSourcemap: true
	// },
	// Consult https://kit.svelte.dev/docs/integrations#preprocessors
	// for more information about preprocessors
	preprocess: vitePreprocess(),

	kit: {
		paths: { base: process.env.VITE_BASE_PATH || '' },
		...(process.env.VITE_BASE_PATH
			? {
					csp: {
						mode: 'hash',
						directives: {
							'default-src': ['self'],
							'script-src': ['self'],
							'style-src': ['self', 'unsafe-inline'],
							'img-src': ['self', 'data:'],
							'object-src': ['none'],
							'base-uri': ['self']
						}
					}
				}
			: {}),
		// adapter-auto only supports some environments, see https://kit.svelte.dev/docs/adapter-auto for a list.
		// If your environment is not supported, or you settled on a specific environment, switch out the adapter.
		// See https://kit.svelte.dev/docs/adapters for more information about adapters.
		adapter: adapter({
			// default options are shown. On some platforms
			// these options are set automatically — see below
			pages: 'build',
			assets: 'build',
			fallback: 'index.html',
			precompress: false,
			strict: true
		})
	}
};

export default config;
