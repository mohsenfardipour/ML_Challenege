// CSV export and "copy day as Persian text".
import type { Day, Dimension, Settings } from '../types';
import { COMFORT_LABELS, MOOD_TITLES, STATUS_LABELS, WALK_BLANK } from '../defaults';
import { scoreDay, cardScore } from './scoring';
import { fmtJalali, num, signed } from './format';
import { isoToJalali } from './dates';
import { commitmentPoints } from './scoring';

const cell = (v: unknown) => {
  const s = v === null || v === undefined ? '' : String(v);
  return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
};
const csv = (rows: unknown[][]) => '﻿' + rows.map((r) => r.map(cell).join(',')).join('\r\n');
const jal = (iso: string) => { const { jy, jm, jd } = isoToJalali(iso); return `${jy}/${jm}/${jd}`; };

export function labelOf(label: string, settings: Settings) {
  return settings.walkBlank ? label.replace(WALK_BLANK, settings.walkBlank) : label;
}

export function daysCsv(days: Day[]) {
  return csv([
    ['date', 'jalali', 'day_number', 'intention', 'actions_score', 'commitments_score', 'total', 'mood', 'experience'],
    ...days.map((d) => {
      const s = scoreDay(d, d.rules);
      return [d.date, jal(d.date), d.dayNumber, d.intention, s.actions, s.commitments, s.total, d.mood, d.experienceText];
    }),
  ]);
}

export function actionsCsv(days: Day[], dims: Dimension[]) {
  const dn = (id: string | null) => dims.find((x) => x.id === id)?.name ?? '';
  return csv([
    ['date', 'jalali', 'card', 'topic', 'dimension', 'description', 'reflection', 'need', 'value', 'ability', 'comfort_zone', 'score', 'max', 'status'],
    ...days.flatMap((d) => d.actions.map((a, i) => [
      d.date, jal(d.date), i + 1, a.topic, dn(a.dimensionId), a.description, a.reflection, a.need, a.value, a.ability,
      a.comfort ? COMFORT_LABELS[a.comfort - 1] : '', a.score, d.rules.cardMax[i], a.status,
    ])),
  ]);
}

export function commitmentsCsv(days: Day[]) {
  return csv([
    ['date', 'jalali', 'row', 'label', 'answer', 'points', 'count', 'note', 'behavior', 'streak'],
    ...days.flatMap((d) => d.rules.defs.map((def, i) => {
      const a = d.commitments[def.id];
      return [d.date, jal(d.date), i + 1, def.label, a?.answer ?? '', commitmentPoints(def, a), a?.n ?? '', a?.note ?? '', a?.behavior ?? '', a?.streak ?? ''];
    })),
  ]);
}

const MOOD_EMOJI = ['😀', '🙂', '😐', '🙁', '😧'];

export function dayAsText(d: Day, settings: Settings, dims: Dimension[]): string {
  const s = scoreDay(d, d.rules);
  const L: string[] = [];
  L.push(`فرم انتخاب، ایجاد و بررسی کیمیاگر${settings.name ? `: ${settings.name}` : ''}`);
  L.push(`روز ${num(d.dayNumber ?? '…')} از دوره ${num(settings.courseLength)} روزه‌ام — تاریخ: ${fmtJalali(d.date)}`);
  if (d.intention) L.push(`قصد امروز جاری کردن: ${d.intention}`);
  d.actions.forEach((a, i) => {
    L.push('');
    const dim = dims.find((x) => x.id === a.dimensionId);
    L.push(`${num(i + 1)}) موضوع اقدام: ${a.topic || '—'}${dim ? ` (${dim.name})` : ''}`);
    if (a.description) L.push(`شرح اقدام: ${a.description}`);
    if (a.reflection) L.push(`موانع و کشف: ${a.reflection}`);
    const nva = [a.need && `نیاز: ${a.need}`, a.value && `ارزش: ${a.value}`, a.ability && `توانایی: ${a.ability}`].filter(Boolean);
    if (nva.length) L.push(nva.join(' | '));
    const meta = [
      a.comfort && `خروج از ناحیه امن: ${COMFORT_LABELS[a.comfort - 1]}`,
      a.status !== 'todo' && STATUS_LABELS[a.status],
      `امتیاز: ${num(cardScore(a.score, d.rules.cardMax[i]))} از ${num(d.rules.cardMax[i])}`,
    ].filter(Boolean);
    L.push(meta.join(' | '));
  });
  L.push('', 'تعهدات:');
  d.rules.defs.forEach((def) => {
    const a = d.commitments[def.id];
    const mark = a?.answer === 'yes' ? '✅' : a?.answer === 'no' ? '❌' : '⬜';
    let line = `${mark} ${labelOf(def.label, settings)}`;
    if (a?.answer) line += ` (${signed(commitmentPoints(def, a))})`;
    if (a?.answer === 'no' && def.kind === 'multiplier') line += ` — ${num(a.n ?? 1)} مورد${a.note ? `: ${a.note}` : ''}`;
    if (def.kind === 'streak' && (a?.behavior || a?.streak)) line += ` — ${a.behavior ?? ''}${a.streak ? ` برای ${num(a.streak)} روز متوالی` : ''}`;
    L.push(line);
  });
  if (d.experienceText || d.mood) {
    L.push('');
    if (d.experienceText) L.push(`تجربه امروز: ${d.experienceText}`);
    if (d.mood) L.push(`حال امروز: ${MOOD_EMOJI[d.mood - 1]} ${MOOD_TITLES[d.mood - 1]}`);
  }
  L.push('', `امتیاز امروز: ${num(s.total)} (اقدام‌ها ${num(s.actions)} + تعهدات ${num(s.commitments)})`);
  return L.join('\n');
}
