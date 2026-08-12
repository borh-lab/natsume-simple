import type { PlaywrightTestConfig } from '@playwright/test';

const config: PlaywrightTestConfig = {
	webServer: [
		{
			command: 'cd .. && uv run -q --extra cpu python -m tests.fixture_server',
			port: 8000
		},
		{
			command:
				'VITE_API_URL=http://127.0.0.1:8000 npm run build && npm run preview -- --host 127.0.0.1',
			port: 4173
		}
	],
	use: {
		baseURL: 'http://127.0.0.1:4173',
		trace: 'retain-on-failure',
		screenshot: 'only-on-failure'
	},
	testDir: 'tests',
	testMatch: /(.+\.)?(test|spec)\.[jt]s/
};

export default config;
