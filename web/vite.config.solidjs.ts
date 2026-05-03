import { defineConfig } from 'vite';
import solidPlugin from 'vite-plugin-solid';

// Config for the SolidJS variant (src/main_solidjs.tsx + index.solidjs.html)
export default defineConfig({
  plugins: [solidPlugin()],
  server: {
    proxy: {
      '/sessions': 'http://localhost:8000',
    },
  },
  build: {
    rollupOptions: {
      // Keep output entry name as index.html for FastAPI StaticFiles(html=True).
      input: {
        index: 'index.solidjs.html',
      },
    },
  },
});
