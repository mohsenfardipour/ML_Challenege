import { useCallback, useEffect, useRef, useState } from 'react';
import { useLiveQuery } from 'dexie-react-hooks';
import { blankDay, db, saveDay } from './db';
import { DEFAULT_SETTINGS } from './defaults';
import type { Action, CommitmentAnswer, Day, Settings } from './types';
import type { ISODate } from './lib/dates';

/** Hash routing: "#/day/2026-11-15/plan" -> ["day","2026-11-15","plan"] */
export function useRoute(): string[] {
  const [hash, setHash] = useState(location.hash);
  useEffect(() => {
    const on = () => { setHash(location.hash); window.scrollTo(0, 0); };
    window.addEventListener('hashchange', on);
    return () => window.removeEventListener('hashchange', on);
  }, []);
  return hash.replace(/^#\/?/, '').split('/').filter(Boolean);
}

export const go = (path: string) => { location.hash = `#/${path}`; };

export function useSettings(): Settings {
  return useLiveQuery(() => db.settings.get('main'), []) ?? DEFAULT_SETTINGS;
}

export async function patchSettings(patch: Partial<Settings>) {
  const cur = (await db.settings.get('main')) ?? DEFAULT_SETTINGS;
  await db.settings.put({ ...cur, ...patch });
}

export function useDims() {
  return useLiveQuery(() => db.dimensions.orderBy('order').toArray(), []) ?? [];
}

export function useAllDays(): Day[] | undefined {
  return useLiveQuery(() => db.days.orderBy('date').toArray(), []);
}

const SAVE_DELAY = 400;

/**
 * Load one day and autosave every change (debounced). A day that does not exist yet
 * is only written once something is changed, so browsing never creates empty pages.
 */
export function useDay(date: ISODate) {
  const [day, setDay] = useState<Day | null>(null);
  const [saved, setSaved] = useState(true);
  const pending = useRef<Day | null>(null);
  const timer = useRef<number | undefined>(undefined);

  const flush = useCallback(async () => {
    window.clearTimeout(timer.current);
    const d = pending.current;
    if (!d) return;
    pending.current = null;
    await saveDay(d);
    if (!pending.current) setSaved(true);
  }, []);

  useEffect(() => {
    let alive = true;
    setDay(null);
    (async () => {
      const d = (await db.days.get(date)) ?? (await blankDay(date));
      if (alive) setDay(d);
    })();
    return () => { alive = false; void flush(); };
  }, [date, flush]);

  useEffect(() => {
    const onHide = () => { if (document.visibilityState === 'hidden') void flush(); };
    document.addEventListener('visibilitychange', onHide);
    window.addEventListener('pagehide', flush);
    window.addEventListener('beforeunload', flush);
    return () => {
      document.removeEventListener('visibilitychange', onHide);
      window.removeEventListener('pagehide', flush);
      window.removeEventListener('beforeunload', flush);
    };
  }, [flush]);

  const update = useCallback((fn: (d: Day) => Day) => {
    setDay((prev) => {
      if (!prev) return prev;
      const next = fn(prev);
      pending.current = next;
      return next;
    });
    setSaved(false);
    window.clearTimeout(timer.current);
    timer.current = window.setTimeout(() => void flush(), SAVE_DELAY);
  }, [flush]);

  const setAction = useCallback((i: number, patch: Partial<Action>) =>
    update((d) => ({ ...d, actions: d.actions.map((a, j) => (j === i ? { ...a, ...patch } : a)) })), [update]);

  const setCommit = useCallback((id: string, patch: Partial<CommitmentAnswer>) =>
    update((d) => ({ ...d, commitments: { ...d.commitments, [id]: { ...(d.commitments[id] ?? { answer: null }), ...patch } } })), [update]);

  return { day, update, setAction, setCommit, saved, flush };
}

/** The latest saved day strictly before `date` (for streaks and "same as yesterday" suggestions). */
export function usePrevDay(date: ISODate) {
  return useLiveQuery(() => db.days.where('date').below(date).last(), [date]);
}
