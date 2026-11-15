import { useMemo, useState } from 'react';
import { go, useAllDays, useDay, useDims, usePrevDay, useSettings } from '../hooks';
import { ActionCard, Commitments, ScorePanel, type Mode, type Suggestions } from '../components/dayParts';
import { IntInput, MoodPicker, Segmented } from '../components/ui';
import { addDays, todayISO, type ISODate } from '../lib/dates';
import { fmtJalali, fmtJalaliLong, num } from '../lib/format';
import { dayAsText } from '../lib/export';
import { buildRules } from '../lib/rules';
import { db } from '../db';
import type { Day } from '../types';

/** Most frequent values first (case/space-trimmed), from most recent days. */
function frequent(values: string[], limit = 30): string[] {
  const count = new Map<string, number>();
  for (const v of values.flatMap((x) => x.split(/[،,/]/)).map((x) => x.trim()).filter(Boolean)) count.set(v, (count.get(v) ?? 0) + 1);
  return [...count.entries()].sort((a, b) => b[1] - a[1]).map(([v]) => v).slice(0, limit);
}

export function useSuggestions(days: Day[] | undefined, exclude: ISODate): Suggestions {
  return useMemo(() => {
    const ds = (days ?? []).filter((d) => d.date !== exclude);
    const acts = ds.flatMap((d) => d.actions);
    return {
      topics: [...new Set(acts.map((a) => a.topic.trim()).filter(Boolean).reverse())],
      need: frequent(acts.map((a) => a.need)),
      value: frequent(acts.map((a) => a.value)),
      ability: frequent(acts.map((a) => a.ability)),
    };
  }, [days, exclude]);
}

export function DayPage({ date, initialMode }: { date: ISODate; initialMode?: Mode }) {
  const settings = useSettings();
  const dims = useDims();
  const days = useAllDays();
  const prev = usePrevDay(date);
  const { day, update, setAction, setCommit, saved } = useDay(date);
  const today = todayISO();
  const [mode, setMode] = useState<Mode>(initialMode ?? (date > today ? 'plan' : 'review'));
  const [copied, setCopied] = useState(false);
  const sugg = useSuggestions(days, date);

  if (!day) return <p className="muted pad">در حال بارگذاری…</p>;

  const copyText = async () => {
    const text = dayAsText(day, settings, dims);
    try { await navigator.clipboard.writeText(text); } catch {
      const ta = document.createElement('textarea'); ta.value = text; document.body.appendChild(ta); ta.select(); document.execCommand('copy'); ta.remove();
    }
    setCopied(true); setTimeout(() => setCopied(false), 2000);
  };

  const applyCurrentRules = async () => {
    if (!confirm('قوانین امتیازدهی فعلی (از تنظیمات) روی این روز اعمال شود؟ امتیاز این روز ممکن است تغییر کند.')) return;
    const [defs] = await Promise.all([db.commitmentDefs.toArray()]);
    update((d) => ({ ...d, rules: buildRules(settings.cardMax, defs) }));
  };

  const rel = date === today ? 'امروز' : date === addDays(today, 1) ? 'فردا' : date === addDays(today, -1) ? 'دیروز' : '';

  return (
    <div className="page day-page">
      <div className="day-nav">
        <button className="ghost" onClick={() => go(`day/${addDays(date, -1)}`)} aria-label="روز قبل">→</button>
        <div className="center">
          <div className="title">{rel && <span className="rel">{rel} · </span>}{fmtJalaliLong(date)}</div>
          <span className={`saved ${saved ? '' : 'pending'}`}>{saved ? 'ذخیره شد ✓' : 'در حال ذخیره…'}</span>
        </div>
        <button className="ghost" onClick={() => go(`day/${addDays(date, 1)}`)} aria-label="روز بعد">←</button>
      </div>

      <Segmented className="mode-tabs" clearable={false} value={mode} onChange={(m) => m && setMode(m)}
        options={[{ value: 'plan', label: 'برنامه‌ریزی (شب قبل)' }, { value: 'review', label: 'انجام و بررسی' }]} />

      <header className="form-head card">
        <div className="fh-row">
          <span>فرم انتخاب، ایجاد و بررسی کیمیاگر: <b>{settings.name || <a href="#/settings">(نام در تنظیمات)</a>}</b></span>
        </div>
        <div className="fh-row">
          <span>روز <IntInput value={day.dayNumber} min={1} max={999} ariaLabel="شماره روز" onChange={(dayNumber) => update((d) => ({ ...d, dayNumber }))} /> از دوره {num(settings.courseLength)} روزه‌ام.</span>
          <span>تاریخ: <b>{fmtJalali(date)}</b></span>
        </div>
        <label className="field-inline">
          <b>قصد امروز جاری کردن:</b>
          <input value={day.intention} onChange={(e) => update((d) => ({ ...d, intention: e.target.value }))} />
        </label>
      </header>

      {day.actions.map((a, i) => (
        <ActionCard key={i} i={i} a={a} max={day.rules.cardMax[i]} mode={mode} dims={dims} sugg={sugg}
          prevTopic={prev?.actions[i]?.topic} onChange={(p) => setAction(i, p)} />
      ))}

      {mode === 'review' && (
        <>
          <section className="card">
            <Commitments day={day} settings={settings} prev={prev} onChange={setCommit} />
          </section>
          <section className="card">
            <label className="field-block">
              <b>تجربه امروز:</b>
              <textarea rows={3} value={day.experienceText} onChange={(e) => update((d) => ({ ...d, experienceText: e.target.value }))} />
            </label>
            <MoodPicker value={day.mood} onChange={(mood) => update((d) => ({ ...d, mood }))} />
          </section>
          <ScorePanel day={day} />
        </>
      )}

      <div className="actions-row">
        {mode === 'plan'
          ? <button onClick={() => setMode('review')}>رفتن به انجام و بررسی</button>
          : <button onClick={() => go(`day/${addDays(date, 1)}/plan`)}>برنامه‌ریزی فردا</button>}
        <button className="ghost" onClick={copyText}>{copied ? 'کپی شد ✓' : 'کپی متن روز'}</button>
        <button className="ghost small" onClick={applyCurrentRules} title="این روز با قوانین امتیازی زمان ساخت خودش ذخیره شده است">اعمال قوانین فعلی</button>
      </div>
    </div>
  );
}
