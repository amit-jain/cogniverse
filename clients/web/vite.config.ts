import react from '@vitejs/plugin-react';
import { defineConfig } from 'vitest/config';

const serverPort = process.env.PORT ?? '4000';

export default defineConfig({
  plugins: [react()],
  build: { outDir: 'dist/client', emptyOutDir: true },
  server: {
    proxy: { '/api': `http://127.0.0.1:${serverPort}` },
  },
  test: {
    globals: true,
    include: ['tests/**/*.test.ts', 'tests/**/*.test.tsx'],
    // CopilotKit's entry imports its stylesheet, which Node cannot load.
    server: { deps: { inline: [/@copilotkit\/react-core/] } },
  },
});
