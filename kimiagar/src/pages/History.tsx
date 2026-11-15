import { useState } from 'react';
import { go, useAllDays, useDims } from '../hooks';
import { Card, Face, JalaliDateInput } from '../components/ui';
import { JALALI_MONTHS, addDays, isoToJalali, jalaliToISO, monthLength, parseISO, todayISO, type ISODate } from '../lib/dates';
import { fmtJalali, num } from '../lib/format';
import { averageBetween, dayTotal, isScored, monthAverage, weekAverage } from '../lib/logic';
import type { Day } from '../types';

function dayState(d: Day): { label: string; cls: string } {
  if (isScored(d)) return { label: 'بررسی شد', cls: 'st-done' };
  const planned = d.intention || d.actions.some((a) => a.topic || a.description);
  return planned ? { label: 'برنامه‌ریزی شد', cls: 'st-plan' } : { label: 'خالی', cls: 'st-empty' };
}

export function History() {
  const days = useAllDays();
  const dims = useDims();
  const today = todayISO();
  const tj = isoToJalali(today);
  const [ym, setYm] = useState({ jy: tj.jy, jm: tj.jm });
  const [from, setFrom] = useState<ISODate | null>(null);
  const [to, setTo] = useState<ISODate | null>(null);
  const [dim, setDim] = useState('');
  if (!days) return <p className="muted pad">در حال بارگذاری…</p>;

  const byDate = new Map(days.map((d) => [d.date, d]));

  // Month grid (weeks start on Saturday)
  const first = jalaliToISO(ym.jy, ym.jm, 1)!;
  const len = monthLength(ym.jy, ym.jm);
  const lead = (parseISO(first).getDay() + 1) % 7;
  const cells: (ISODate | null)[] = [...Array(lead).fill(null), ...Array.from({ length: len }, (_, i) => addDays(first, i))];
  const shiftMonth = (n: number) => {
    let m = ym.jm + n, y = ym.jy;
    if (m < 1) { m = 12; y--; } if (m > 12) { m = 1; y++; }
    setYm({ jy: y, jm: m });
  };

  const list = days
    .filter((d) => (!from || d.date >= from) && (!to || d.date <= to))
    .filter((d) => !dim || d.actions.some((a) => a.dimensionId === dim))
    .slice().reverse();
  const periodAvg = averageBetween(list, '0000-01-01', '9999-12-31');
  const monthAvg = monthAverage(days, first);
  const weekAvg = weekAverage(days, today);

  return (
    <div className="page history">
      <h1>تاریخچه</h1>
      <Card>
        <div className="cal-nav">
          <button className="ghost" onClick={() => shiftMonth(-1)} aria-label="ماه قبل">→</button>
          <b>{JALALI_MONTHS[ym.jm - 1]} {num(ym.jy)}</b>
          <button className="ghost" onClick={() => shiftMonth(1)} aria-label="ماه بعد">←</button>
        </div>
        <div className="cal">
          {['ش', 'ی', 'د', 'س', 'چ', 'پ', 'ج'].map((w) => <span key={w} className="wd">{w}</span>)}
          {cells.map((iso, i) => {
            if (!iso) return <span key={i} />;
            const d = byDate.get(iso);
            const st = d ? dayState(d) : null;
            return (
              <button key={iso} className={`cal-cell ${st?.cls ?? ''} ${iso === today ? 'today' : ''}`} onClick={() => go(`day/${iso}`)}>
                <span className="dn">{num(isoToJalali(iso).jd)}</span>
                {d && isScored(d) && <b>{num(dayTotal(d))}</b>}
                {d?.dayNumber && <small>روز {num(d.dayNumber)}</small>}
              </button>
            );
          })}
        </div>
        <div className="avgs">
          <span>میانگین این هفته: <b>{weekAvg === null ? '—' : num(weekAvg)}</b></span>
          <span>میانگین این ماه: <b>{monthAvg === null ? '—' : num(monthAvg)}</b></span>
        </div>
      </Card>

      <Card>
        <div className="filters">
          <label>از <JalaliDateInput value={from} onChange={setFrom} allowEmpty /></label>
          <label>تا <JalaliDateInput value={to} onChange={setTo} allowEmpty /></label>
          <label>بُعد
            <select value={dim} onChange={(e) => setDim(e.target.value)}>
              <option value="">همه</option>
              {dims.map((d) => <option key={d.id} value={d.id}>{d.name}</option>)}
            </select>
          </label>
        </div>
        <p>میانگین بازهٔ انتخاب‌شده: <b>{periodAvg === null ? '—' : num(periodAvg)}</b> <span className="muted">({num(list.filter(isScored).length)} روز امتیازدار)</span></p>
        <ul className="day-list">
          {list.length === 0 && <li className="muted">روزی ثبت نشده.</li>}
          {list.map((d) => {
            const st = dayState(d);
            return (
              <li key={d.date} onClick={() => go(`day/${d.date}`)}>
                <span className="dl-num">{d.dayNumber ? `روز ${num(d.dayNumber)}` : ''}</span>
                <span className="dl-date">{fmtJalali(d.date)}</span>
                <span className={`dl-st ${st.cls}`}>{st.label}</span>
                <span className="dl-mood">{d.mood ? <Face mood={d.mood} size={22} /> : ''}</span>
                <b className="dl-total">{isScored(d) ? num(dayTotal(d)) : '—'}</b>
                <span className="dl-int muted">{d.intention}</span>
              </li>
            );
          })}
        </ul>
      </Card>
    </div>
  );
}
