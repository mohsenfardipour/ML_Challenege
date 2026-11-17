// PWA helpers: offline service worker, iOS detection, and saving files on iPhone.

export const isIOS = () =>
  /iPad|iPhone|iPod/.test(navigator.userAgent) || (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);

/** Running from the home-screen icon (installed), not inside a Safari tab. */
export const isStandalone = () =>
  (navigator as any).standalone === true || window.matchMedia?.('(display-mode: standalone)').matches === true;

/** Register the offline cache. Only on real web hosting (https / localhost), never in Electron or dev. */
export function registerServiceWorker() {
  if (!('serviceWorker' in navigator) || import.meta.env.DEV) return;
  const ok = location.protocol === 'https:' || location.hostname === 'localhost' || location.hostname === '127.0.0.1';
  if (!ok) return;
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('./sw.js', { scope: './' }).catch(() => { /* app still works online */ });
  });
}

export type SaveResult = 'saved' | 'cancelled' | 'needs-tap';

/**
 * Save a file. On iPhone/iPad the share sheet is used ("Save to Files", Telegram, …), because plain downloads
 * are unreliable in home-screen apps. iOS only allows the share sheet right after a tap; if the tap "expired"
 * while the file was being prepared, returns 'needs-tap' so the UI can offer a second button.
 */
export async function saveFile(name: string, content: string, mime: string): Promise<SaveResult> {
  const file = new File([content], name, { type: mime });
  if (isIOS() && navigator.canShare?.({ files: [file] })) {
    try {
      await navigator.share({ files: [file], title: name });
      return 'saved';
    } catch (e) {
      const n = (e as DOMException).name;
      if (n === 'AbortError') return 'cancelled';
      if (n === 'NotAllowedError') return 'needs-tap';
      // fall through to a normal download
    }
  }
  const url = URL.createObjectURL(file);
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
  return 'saved';
}

/** Share plain text (e.g. the day summary to Telegram) when the device supports it. */
export const canShareText = () => typeof navigator.share === 'function';
export async function shareText(text: string) {
  try { await navigator.share({ text }); } catch { /* cancelled */ }
}
