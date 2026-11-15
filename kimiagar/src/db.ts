import Dexie, { type Table } from 'dexie';
import type { CommitmentDef, Day, DimRating, Dimension, Experience, Settings, StarExercise } from './types';
import { DEFAULT_COMMITMENTS, DEFAULT_DIMENSIONS, DEFAULT_SETTINGS, emptyAction } from './defaults';
import { buildRules } from './lib/rules';
import { suggestDayNumber } from './lib/logic';
import type { ISODate } from './lib/dates';

export class KimiagarDB extends Dexie {
  settings!: Table<Settings, string>;
  commitmentDefs!: Table<CommitmentDef, string>;
  days!: Table<Day, string>;
  starExercises!: Table<StarExercise, string>;
  experiences!: Table<Experience, string>;
  dimensions!: Table<Dimension, string>;
  dimRatings!: Table<DimRating, string>;

  constructor(name = 'kimiagar') {
    super(name);
    this.version(1).stores({
      settings: 'id',
      commitmentDefs: 'id, order',
      days: 'date, dayNumber',
      starExercises: 'id, date, status',
      experiences: 'id, completedDate',
      dimensions: 'id, order',
      dimRatings: 'id, dimensionId, date',
    });
    this.on('populate', (tx) => {
      tx.table('settings').add(DEFAULT_SETTINGS);
      tx.table('commitmentDefs').bulkAdd(DEFAULT_COMMITMENTS);
      tx.table('dimensions').bulkAdd(DEFAULT_DIMENSIONS);
    });
  }
}

export const db = new KimiagarDB();

export const TABLES = ['settings', 'commitmentDefs', 'days', 'starExercises', 'experiences', 'dimensions', 'dimRatings'] as const;

export function uid(): string {
  return typeof crypto !== 'undefined' && 'randomUUID' in crypto
    ? crypto.randomUUID()
    : `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
}

export async function getSettings(): Promise<Settings> {
  return (await db.settings.get('main')) ?? DEFAULT_SETTINGS;
}

/** A blank (unsaved) day with today's rule snapshot and the suggested page number. */
export async function blankDay(date: ISODate): Promise<Day> {
  const [settings, defs, allDays] = await Promise.all([getSettings(), db.commitmentDefs.toArray(), db.days.toArray()]);
  return {
    date,
    dayNumber: suggestDayNumber(allDays, date),
    intention: '',
    actions: [0, 1, 2, 3].map(emptyAction),
    commitments: {},
    rules: buildRules(settings.cardMax, defs),
    experienceText: '',
    mood: null,
    updatedAt: 0,
  };
}

export async function saveDay(day: Day): Promise<void> {
  await db.days.put({ ...day, updatedAt: Date.now() });
}

/** Ask the browser not to evict our data under storage pressure. */
export async function requestPersistence(): Promise<boolean> {
  try {
    if (navigator.storage?.persist) return await navigator.storage.persist();
  } catch { /* ignore */ }
  return false;
}
