import { describe, expect, it } from 'vitest';
import { averageBetween, backupDue, daysUntil, nextExperienceDue, suggestDayNumber, suggestStreak, weekAverage } from './logic';
import { weekStart, jalaliToISO, isoToJalali, addDays } from './dates';
import { buildRules } from './rules';
import { DEFAULT_CARD_MAX, DEFAULT_COMMITMENTS, emptyAction } from '../defaults';
import type { Day, Experience } from '../types';

const rules = buildRules(DEFAULT_CARD_MAX, DEFAULT_COMMITMENTS);
function mkDay(date: string, scores: (number | null)[], dayNumber: number | null = null): Day {
  return {
    date, dayNumber, intention: '', rules, experienceText: '', mood: null, updatedAt: 0, commitments: {},
    actions: scores.map((score) => ({ ...emptyAction(), score })),
  };
}
const exp = (completedDate: string | null): Experience => ({
  id: Math.random().toString(), intent: '', why: '', targetDate: null, completed: !!completedDate,
  completedDate, learned: '', rating: null, createdAt: 0,
});

describe('Jalali dates', () => {
  it('converts known dates both ways', () => {
    expect(isoToJalali('2026-11-14')).toEqual({ jy: 1405, jm: 8, jd: 23 });
    expect(jalaliToISO(1405, 1, 1)).toBe('2026-03-21');
    expect(jalaliToISO(1405, 7, 5)).toBe('2026-09-27');
    expect(jalaliToISO(1405, 12, 31)).toBeNull(); // 1405 is not a leap year
  });
  it('week starts on Saturday', () => {
    expect(weekStart('2026-11-14')).toBe('2026-11-14'); // Saturday
    expect(weekStart('2026-11-20')).toBe('2026-11-14'); // Friday
    expect(weekStart('2026-11-21')).toBe('2026-11-21'); // next Saturday
  });
});

describe('averages', () => {
  const days = [mkDay('2026-11-14', [10, 10, 8, 7]), mkDay('2026-11-15', [10, 10, 0, 0]), mkDay('2026-11-16', [null, null, null, null])];
  it('ignores unscored (plan-only) days', () => {
    expect(averageBetween(days, '2026-11-01', '2026-11-30')).toBe(27.5);
  });
  it('weekly average', () => {
    expect(weekAverage(days, '2026-11-20')).toBe(27.5);
    expect(weekAverage(days, '2026-11-13')).toBeNull();
  });
});

describe('14-day experience cycle', () => {
  it('first due 14 days after course start', () => {
    expect(nextExperienceDue([], '2026-11-01', 14)).toBe('2026-11-15');
  });
  it('next due 14 days after the last completed one', () => {
    expect(nextExperienceDue([exp('2026-11-10'), exp('2026-11-03'), exp(null)], '2026-11-01', 14)).toBe('2026-11-24');
  });
  it('respects a custom interval and no start date', () => {
    expect(nextExperienceDue([exp('2026-11-10')], null, 10)).toBe('2026-11-20');
    expect(nextExperienceDue([], null, 14)).toBeNull();
  });
  it('days left / overdue', () => {
    expect(daysUntil('2026-11-20', '2026-11-15')).toBe(5);
    expect(daysUntil('2026-11-10', '2026-11-15')).toBe(-5);
  });
});

describe('suggestions', () => {
  it('streak = yesterday + 1 only if yesterday was Yes with a number', () => {
    expect(suggestStreak({ answer: 'yes', streak: 6 })).toBe(7);
    expect(suggestStreak({ answer: 'no', streak: 6 })).toBeNull();
    expect(suggestStreak({ answer: 'yes' })).toBeNull();
    expect(suggestStreak(undefined)).toBeNull();
  });
  it('day number follows the last recorded page, not the calendar', () => {
    const days = [mkDay('2026-10-01', [], 20), mkDay('2026-10-11', [], 23)];
    expect(suggestDayNumber(days, '2026-10-22')).toBe(24);
    expect(suggestDayNumber(days, '2026-10-05')).toBe(21);
    expect(suggestDayNumber([], '2026-10-05')).toBe(1);
  });
  it('weekly backup reminder', () => {
    const now = Date.now();
    expect(backupDue(null, now, true)).toBe(true);
    expect(backupDue(null, now, false)).toBe(false);
    expect(backupDue(now - 3 * 86400000, now, true)).toBe(false);
    expect(backupDue(now - 8 * 86400000, now, true)).toBe(true);
  });
  it('addDays crosses month boundaries', () => {
    expect(addDays('2026-11-30', 1)).toBe('2026-12-01');
  });
});
