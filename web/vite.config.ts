import { defineConfig } from 'vite';
import solidPlugin from 'vite-plugin-solid';

// Config for the frontend.
export default defineConfig({
  plugins: [solidPlugin()],
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
