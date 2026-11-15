import { useState } from 'react';
import { useLiveQuery } from 'dexie-react-hooks';
import { db, uid } from '../db';
import { useAllDays, useDims } from '../hooks';
import { Card, Segmented } from '../components/ui';
import { addDays, todayISO } from '../lib/dates';
import { fmtJalali, num } from '../lib/format';
import type { DimRating } from '../types';

function HBars({ rows }: { rows: { name: string; value: number }[] }) {
  const max = Math.max(1, ...rows.map((r) => r.value));
  return (
    <div className="hbars">
      {rows.map((r) => (
        <div className="hbar" key={r.name}>
          <span className="hb-name">{r.name}</span>
          <svg viewBox="0 0 100 10" preserveAspectRatio="none" className="hb-svg">
            <rect x={100 - (r.value / max) * 100} y="1" width={(r.value / max) * 100} height="8" rx="1.5" className="bar" />
          </svg>
          <b>{num(r.value)}</b>
        </div>
      ))}
    </div>
  );
}

export function Balance() {
  const dims = useDims();
  const days = useAllDays() ?? [];
  const ratings = useLiveQuery(() => db.dimRatings.orderBy('date').toArray(), []) ?? [];
  const [win, setWin] = useState<14 | 30>(14);
  const [dimId, setDimId] = useState('');
  const [rating, setRating] = useState<number | null>(null);
  const [note, setNote] = useState('');
  const today = todayISO();
  const since = addDays(today, -(win - 1));

  const counts = new Map<string, number>();
  days.filter((d) => d.date >= since && d.date <= today)
    .forEach((d) => d.actions.forEach((a) => a.dimensionId && counts.set(a.dimensionId, (counts.get(a.dimensionId) ?? 0) + 1)));
  const lifeDims = dims.filter((d) => d.kind === 'dimension' && d.active);
  const cats = dims.filter((d) => d.kind === 'category' && d.active);
  const latest = new Map<string, DimRating>();
  ratings.forEach((r) => latest.set(r.dimensionId, r)); // sorted by date → last wins

  const save = async () => {
    if (!dimId || !rating) return;
    await db.dimRatings.add({ id: uid(), dimensionId: dimId, date: today, rating, note });
    setRating(null); setNote('');
  };

  return (
    <div className="page">
      <h1>تعادل زندگی</h1>
      <p className="muted">برای آگاهی است، نه تکلیف روزانه.</p>
      <Card>
        <div className="row-wrap">
          <h2>تعداد اقدام‌ها در هر بُعد</h2>
          <Segmented clearable={false} value={win} onChange={(v) => v && setWin(v)}
            options={[{ value: 14, label: `${num(14)} روز` }, { value: 30, label: `${num(30)} روز` }]} />
        </div>
        <HBars rows={lifeDims.map((d) => ({ name: d.name, value: counts.get(d.id) ?? 0 }))} />
        {cats.length > 0 && (
          <>
            <h3 className="muted">دسته‌های جدا</h3>
            <HBars rows={cats.map((d) => ({ name: d.name, value: counts.get(d.id) ?? 0 }))} />
          </>
        )}
      </Card>

      <Card>
        <h2>رضایت از هر بُعد (۱ تا ۱۰)</h2>
        <ul className="ratings">
          {lifeDims.map((d) => {
            const r = latest.get(d.id);
            return (
              <li key={d.id}>
                <span>{d.name}</span>
                {r ? <><b>{num(r.rating)}</b><small className="muted">{fmtJalali(r.date)}{r.note ? ` · ${r.note}` : ''}</small></> : <span className="muted">—</span>}
              </li>
            );
          })}
        </ul>
      </Card>

      <Card>
        <h2>ثبت رضایت</h2>
        <select value={dimId} onChange={(e) => setDimId(e.target.value)}>
          <option value="">انتخاب بُعد…</option>
          {lifeDims.map((d) => <option key={d.id} value={d.id}>{d.name}</option>)}
        </select>
        <Segmented value={rating} onChange={setRating} options={Array.from({ length: 10 }, (_, k) => ({ value: k + 1, label: num(k + 1) }))} />
        <input placeholder="یادداشت (اختیاری)" value={note} onChange={(e) => setNote(e.target.value)} />
        <button disabled={!dimId || !rating} onClick={save}>ثبت</button>
      </Card>
    </div>
  );
}
