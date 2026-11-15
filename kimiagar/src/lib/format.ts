import { isoToJalali, JALALI_MONTHS, parseISO, WEEKDAYS, type ISODate } from './dates';

export type DigitMode = 'fa' | 'latin';
let mode: DigitMode = 'fa';
export const setDigitMode = (m: DigitMode) => { mode = m; };

const FA = '۰۱۲۳۴۵۶۷۸۹';

/** Render digits in the user's chosen style. Minus sign kept as ASCII "-" so it reads correctly. */
export function num(v: number | string | null | undefined): string {
  if (v === null || v === undefined) return '';
  const s = String(v);
  return mode === 'fa' ? s.replace(/[0-9]/g, (d) => FA[+d]) : s;
}

/** Signed number, e.g. +10 / -10. */
export function signed(v: number): string {
  return v > 0 ? `+${num(v)}` : num(v);
}

/** Parse user-typed digits (Persian, Arabic or Latin) to an integer, or null. */
export function parseIntFa(s: string): number | null {
  const latin = s
    .replace(/[۰-۹]/g, (d) => String(d.charCodeAt(0) - 0x06f0))
    .replace(/[٠-٩]/g, (d) => String(d.charCodeAt(0) - 0x0660))
    .replace(/[−–]/g, '-')
    .trim();
  if (!/^-?\d+$/.test(latin)) return null;
  return parseInt(latin, 10);
}

export function fmtJalali(iso: ISODate): string {
  const { jy, jm, jd } = isoToJalali(iso);
  return num(`${jy}/${String(jm).padStart(2, '0')}/${String(jd).padStart(2, '0')}`);
}

export function fmtJalaliLong(iso: ISODate): string {
  const { jy, jm, jd } = isoToJalali(iso);
  return `${WEEKDAYS[parseISO(iso).getDay()]} ${num(jd)} ${JALALI_MONTHS[jm - 1]} ${num(jy)}`;
}
