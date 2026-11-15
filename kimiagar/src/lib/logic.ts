// Pure helpers: averages, experience cycle, suggestions. Tested in logic.test.ts.
import type { CommitmentAnswer, Day, Experience } from '../types';
import { addDays, diffDays, jalaliMonthRange, weekStart, type ISODate } from './dates';
import { scoreDay } from './scoring';

/** A day counts towards averages once anything has been scored/answered (pure plans don't). */
export function isScored(d: Day): boolean {
  return d.actions.some((a) => a.score !== null) || Object.values(d.commitments).some((c) => c.answer !== null);
}

export function dayTotal(d: Day): number {
  return scoreDay(d, d.rules).total;
}

/** Average total over scored days with from <= date <= to. null if none. */
export function averageBetween(days: Day[], from: ISODate, to: ISODate): number | null {
  const sel = days.filter((d) => d.date >= from && d.date <= to && isScored(d));
  if (!sel.length) return null;
  return Math.round((sel.reduce((s, d) => s + dayTotal(d), 0) / sel.length) * 10) / 10;
}

/** Week = Saturday..Friday containing iso. */
export function weekAverage(days: Day[], iso: ISODate) {
  const from = weekStart(iso);
  return averageBetween(days, from, addDays(from, 6));
}

/** Jalali month containing iso. */
export function monthAverage(days: Day[], iso: ISODate) {
  const [from, to] = jalaliMonthRange(iso);
  return averageBetween(days, from, to);
}

/** Next new-experience due date: last completed + interval, else course start + interval. */
export function nextExperienceDue(exps: Experience[], courseStart: ISODate | null, interval: number): ISODate | null {
  const done = exps.filter((e) => e.completed && e.completedDate).map((e) => e.completedDate!).sort();
  const base = done.length ? done[done.length - 1] : courseStart;
  return base ? addDays(base, interval) : null;
}

/** Positive = days left, 0 = due today, negative = days overdue. */
export function daysUntil(due: ISODate, today: ISODate): number {
  return diffDays(today, due);
}

/** Streak rows: yesterday Yes with N consecutive days → suggest N+1. */
export function suggestStreak(prev: CommitmentAnswer | undefined): number | null {
  return prev?.answer === 'yes' && typeof prev.streak === 'number' ? prev.streak + 1 : null;
}

/** "Day N" counts workbook pages: last recorded number before this date + 1. */
export function suggestDayNumber(days: Day[], date: ISODate): number {
  const nums = days.filter((d) => d.date < date && d.dayNumber).map((d) => d.dayNumber!);
  return nums.length ? Math.max(...nums) + 1 : 1;
}

export function backupDue(lastBackupAt: number | null, now: number, hasData: boolean): boolean {
  if (!hasData) return false;
  return lastBackupAt === null || now - lastBackupAt > 7 * 86400000;
}
