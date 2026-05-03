import { defineConfig } from 'vite';

// Default config for the raw TypeScript variant (src/main.ts + index.html)
export default defineConfig({
	server: {
		proxy: {
			'/sessions': 'http://localhost:8000',
		},
	},
});
