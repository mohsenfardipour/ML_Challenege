// All dates are stored as Gregorian ISO strings "YYYY-MM-DD" (local calendar day).
import { toJalaali, toGregorian, isValidJalaaliDate, jalaaliMonthLength } from 'jalaali-js';

export type ISODate = string;

const pad = (n: number) => String(n).padStart(2, '0');

export function toISO(d: Date): ISODate {
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
}

export function parseISO(iso: ISODate): Date {
  const [y, m, d] = iso.split('-').map(Number);
  return new Date(y, m - 1, d, 12); // noon avoids DST edge cases
}

export function todayISO(): ISODate {
  return toISO(new Date());
}

export function addDays(iso: ISODate, n: number): ISODate {
  const d = parseISO(iso);
  d.setDate(d.getDate() + n);
  return toISO(d);
}

/** Whole days from a to b (b - a). */
export function diffDays(a: ISODate, b: ISODate): number {
  return Math.round((parseISO(b).getTime() - parseISO(a).getTime()) / 86400000);
}

export function isoToJalali(iso: ISODate) {
  const [y, m, d] = iso.split('-').map(Number);
  return toJalaali(y, m, d);
}

export function jalaliToISO(jy: number, jm: number, jd: number): ISODate | null {
  if (!isValidJalaaliDate(jy, jm, jd)) return null;
  const g = toGregorian(jy, jm, jd);
  return `${g.gy}-${pad(g.gm)}-${pad(g.gd)}`;
}

export const monthLength = jalaaliMonthLength;

export const JALALI_MONTHS = [
  'فروردین', 'اردیبهشت', 'خرداد', 'تیر', 'مرداد', 'شهریور',
  'مهر', 'آبان', 'آذر', 'دی', 'بهمن', 'اسفند',
];

// Index by JS getDay(): 0 = Sunday
export const WEEKDAYS = ['یکشنبه', 'دوشنبه', 'سه‌شنبه', 'چهارشنبه', 'پنجشنبه', 'جمعه', 'شنبه'];

/** Saturday that starts the week containing iso (week = Saturday..Friday). */
export function weekStart(iso: ISODate): ISODate {
  const dow = parseISO(iso).getDay(); // 0 Sun .. 6 Sat
  const back = (dow + 1) % 7; // Sat->0, Sun->1, ... Fri->6
  return addDays(iso, -back);
}

/** First and last ISO date of the Jalali month containing iso. */
export function jalaliMonthRange(iso: ISODate): [ISODate, ISODate] {
  const { jy, jm } = isoToJalali(iso);
  return [jalaliToISO(jy, jm, 1)!, jalaliToISO(jy, jm, jalaaliMonthLength(jy, jm))!];
}
