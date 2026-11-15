import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// base './' so the build works from file:// / Electron custom protocol.
export default defineConfig({
  base: './',
  plugins: [react()],
  test: { environment: 'node' },
} as any);
