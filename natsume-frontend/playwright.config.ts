import type { PlaywrightTestConfig } from '@playwright/test';

const apiPort = Number(process.env.NATSUME_TEST_API_PORT ?? 8000);
const frontendPort = Number(process.env.NATSUME_TEST_FRONTEND_PORT ?? 4173);

const config: PlaywrightTestConfig = {
	webServer: [
		{
			command:
				process.env.NATSUME_FIXTURE_COMMAND ??
				'cd .. && uv run -q --extra backend python -m tests.fixture_server',
			port: apiPort
		},
		{
			command: `VITE_API_URL=http://127.0.0.1:${apiPort} npm run build && npm run preview -- --host 127.0.0.1 --port ${frontendPort}`,
			port: frontendPort
		}
	],
	use: {
		baseURL: `http://127.0.0.1:${frontendPort}`,
		viewport: { width: 1440, height: 900 },
		trace: 'retain-on-failure',
		screenshot: 'only-on-failure'
	},
	testDir: 'tests',
	testMatch: /(.+\.)?(test|spec)\.[jt]s/
};

export default config;
