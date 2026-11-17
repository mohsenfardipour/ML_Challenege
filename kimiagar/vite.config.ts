import { defineConfig, type Plugin } from 'vite';
import react from '@vitejs/plugin-react';
import { createHash } from 'node:crypto';

/**
 * Emits sw.js listing every built file, so the installed PWA (iPhone home screen) works fully offline.
 * The cache name changes whenever any file changes, which makes the phone pick up new versions.
 */
function offlineServiceWorker(): Plugin {
  return {
    name: 'kimiagar-offline-sw',
    apply: 'build',
    generateBundle(_opts, bundle) {
      const files = Object.keys(bundle).filter((f) => !f.endsWith('.map'));
      const extra = ['./', 'manifest.webmanifest', 'icons/icon-192.png', 'icons/icon-512.png', 'icons/apple-touch-icon.png'];
      const list = [...new Set([...extra, ...files])];
      const hash = createHash('sha256');
      for (const f of files) {
        const item = bundle[f] as { code?: string; source?: string | Uint8Array };
        hash.update(f).update(item.code ?? item.source ?? '');
      }
      const version = hash.digest('hex').slice(0, 12);
      this.emitFile({ type: 'asset', fileName: 'sw.js', source: SW_TEMPLATE(version, list) });
    },
  };
}

const SW_TEMPLATE = (version: string, files: string[]) => `// Generated at build time. Offline cache for Kimiagar.
const CACHE = 'kimiagar-${version}';
const FILES = ${JSON.stringify(files)};
self.addEventListener('install', (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(FILES)).then(() => self.skipWaiting()));
});
self.addEventListener('activate', (e) => {
  e.waitUntil(caches.keys()
    .then((keys) => Promise.all(keys.filter((k) => k.startsWith('kimiagar-') && k !== CACHE).map((k) => caches.delete(k))))
    .then(() => self.clients.claim()));
});
self.addEventListener('fetch', (e) => {
  const req = e.request;
  if (req.method !== 'GET' || new URL(req.url).origin !== location.origin) return;
  // App pages: always answer from the offline copy (the app never needs the network).
  if (req.mode === 'navigate') {
    e.respondWith(caches.match('./', { ignoreSearch: true }).then((r) => r || fetch(req)));
    return;
  }
  e.respondWith(caches.match(req, { ignoreSearch: true }).then((r) => r || fetch(req)));
});
`;

// base './' so the build works from file://, the Electron app:// scheme and any sub-folder (e.g. GitHub Pages).
export default defineConfig({
  base: './',
  plugins: [react(), offlineServiceWorker()],
  test: { environment: 'node' },
} as any);
