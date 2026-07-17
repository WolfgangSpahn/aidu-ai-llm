import { defineConfig } from 'vite';
import solidPlugin from 'vite-plugin-solid';
import { fileURLToPath } from 'node:url';
import { dirname, resolve } from 'node:path';

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Config for the frontend.
export default defineConfig({
  plugins: [solidPlugin()],
  resolve: {
    alias: {
      'applet-support/blended-marked': resolve(__dirname, '../../applet-support/src/blended-marked/render.ts'),
    },
  },
  server: {
    proxy: {
      '/sessions': 'http://localhost:8000',
      '/runnables': 'http://localhost:8000',
    },
  },
  build: {
    rollupOptions: {
      // Keep the packaged HTML name aligned with demo/app.py.
      input: {
        index: 'index.html',
      },
    },
  },
});
