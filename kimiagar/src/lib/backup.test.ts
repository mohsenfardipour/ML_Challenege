import 'fake-indexeddb/auto';
import { describe, expect, it } from 'vitest';
import { db, blankDay, saveDay } from '../db';
import { exportAll, restoreAll, validateBackup } from './backup';
import { dayAsText, daysCsv } from './export';
import { DEFAULT_SETTINGS } from '../defaults';

describe('backup round-trip', () => {
  it('exports, wipes and restores identical data', async () => {
    await db.open();
    const d = await blankDay('2026-09-27');
    d.dayNumber = 16;
    d.intention = 'عشق و محبت';
    d.actions[0] = { ...d.actions[0], topic: 'خانواده', score: 10 };
    d.commitments.c01 = { answer: 'no', n: 1, note: 'غیبت' };
    await saveDay(d);
    await db.experiences.add({ id: 'e1', intent: 'کوهنوردی', why: '', targetDate: null, completed: false, completedDate: null, learned: '', rating: null, createdAt: 1 });

    const backup = JSON.parse(JSON.stringify(await exportAll()));
    expect(validateBackup(backup)).toMatchObject({ days: 1, experiences: 1, commitmentDefs: 13, dimensions: 13 });

    await db.days.clear();
    await db.experiences.clear();
    await db.days.put({ ...d, date: '2030-01-01' }); // junk that must disappear
    await restoreAll(backup);

    expect(await db.days.count()).toBe(1);
    const back = await db.days.get('2026-09-27');
    expect(back?.commitments.c01).toEqual({ answer: 'no', n: 1, note: 'غیبت' });
    expect(back?.intention).toBe('عشق و محبت');
    expect((await db.experiences.get('e1'))?.intent).toBe('کوهنوردی');

    const text = dayAsText(back!, DEFAULT_SETTINGS, []);
    expect(text).toContain('غیبت');
    expect(daysCsv([back!]).startsWith('﻿')).toBe(true);
  });

  it('rejects foreign or broken files', () => {
    expect(() => validateBackup({ hello: 1 })).toThrow();
    expect(() => validateBackup({ app: 'kimiagar', schema: 1, data: { days: [] } })).toThrow();
  });
});
