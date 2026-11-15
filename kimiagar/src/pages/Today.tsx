import { useLiveQuery } from 'dexie-react-hooks';
import { go, useAllDays, useDay, useDims, usePrevDay, useSettings } from '../hooks';
import { Commitments, ScorePanel, StatusPicker } from '../components/dayParts';
import { Banner, BarChart, BarLabels, Card } from '../components/ui';
import { addDays, isoToJalali, todayISO } from '../lib/dates';
import { fmtJalali, fmtJalaliLong, num } from '../lib/format';
import { backupDue, dayTotal, daysUntil, isScored, nextExperienceDue, weekAverage } from '../lib/logic';
import { db } from '../db';

export function Today() {
  const today = todayISO();
  const settings = useSettings();
  const dims = useDims();
  const days = useAllDays();
  const prev = usePrevDay(today);
  const exps = useLiveQuery(() => db.experiences.toArray(), []) ?? [];
  const stars = useLiveQuery(() => db.starExercises.toArray(), []) ?? [];
  const { day, setAction, setCommit, saved } = useDay(today);
  if (!day || !days) return <p className="muted pad">در حال بارگذاری…</p>;

  const byDate = new Map(days.map((d) => [d.date, d]));
  // Oldest first; the chart draws the first item on the right (RTL), so time reads right → left.
  const chrono = Array.from({ length: 14 }, (_, k) => addDays(today, -13 + k));

  const due = nextExperienceDue(exps, settings.courseStart, settings.expIntervalDays);
  const left = due ? daysUntil(due, today) : null;
  const drafts = stars.filter((s) => s.status === 'draft');
  const lastDone = stars.filter((s) => s.status === 'done').sort((a, b) => (b.doneAt ?? 0) - (a.doneAt ?? 0))[0];

  // mini balance: actions per dimension, last 14 days
  const since = addDays(today, -13);
  const counts = new Map<string, number>();
  days.filter((d) => d.date >= since).forEach((d) => d.actions.forEach((a) => a.dimensionId && counts.set(a.dimensionId, (counts.get(a.dimensionId) ?? 0) + 1)));
  const topDims = [...counts.entries()].sort((a, b) => b[1] - a[1]).slice(0, 5);
  const wAvg = weekAverage(days, today);

  return (
    <div className="page today">
      <div className="today-head">
        <div>
          <h1>{fmtJalaliLong(today)}</h1>
          {day.dayNumber && <span className="muted">روز {num(day.dayNumber)} از دوره {num(settings.courseLength)} روزه</span>}
        </div>
        <span className={`saved ${saved ? '' : 'pending'}`}>{saved ? '✓' : '…'}</span>
      </div>

      {backupDue(settings.lastBackupAt, Date.now(), days.length > 0) && (
        <Banner tone="warn" onClick={() => go('settings')}>یک هفته از آخرین پشتیبان گذشته است. برای پشتیبان‌گیری اینجا بزنید.</Banner>
      )}
      {left !== null && (
        <Banner tone={left < 0 ? 'warn' : 'info'} onClick={() => go('experiences')}>
          تجربهٔ جدید: {left > 0 ? `${num(left)} روز مانده (${fmtJalali(due!)})` : left === 0 ? 'موعدش امروز است' : `${num(-left)} روز از موعد گذشته`}
        </Banner>
      )}

      <Card>
        <div className="label-row"><b>قصد امروز جاری کردن:</b> {day.intention || <span className="muted">— (شب قبل در «برنامه‌ریزی فردا» بنویسید)</span>}</div>
        <ol className="today-actions">
          {day.actions.map((a, i) => (
            <li key={i}>
              <div className="ta-topic">
                <b>{a.topic || <span className="muted">اقدام {num(i + 1)}</span>}</b>
                {a.description && <span className="muted"> — {a.description}</span>}
              </div>
              <StatusPicker value={a.status} onChange={(status) => setAction(i, { status })} />
            </li>
          ))}
        </ol>
        <div className="actions-row">
          <button onClick={() => go(`day/${today}`)}>صفحهٔ کامل امروز (امتیاز و بازتاب)</button>
          <button className="ghost" onClick={() => go(`day/${addDays(today, 1)}/plan`)}>برنامه‌ریزی فردا</button>
        </div>
      </Card>

      <Card>
        <h2>تعهدات روزانه</h2>
        <Commitments day={day} settings={settings} prev={prev} onChange={setCommit} compact />
      </Card>

      <ScorePanel day={day} />

      <Card>
        <h2>۱۴ روز اخیر {wAvg !== null && <small className="muted">· میانگین این هفته {num(wAvg)}</small>}</h2>
        <BarChart data={chrono.map((iso) => {
          const d = byDate.get(iso);
          return { label: iso, value: d && isScored(d) ? dayTotal(d) : null, onClick: () => go(`day/${iso}`), title: `${fmtJalali(iso)}: ${d && isScored(d) ? num(dayTotal(d)) : '—'}` };
        })} />
        <BarLabels labels={chrono.map((iso) => {
          const d = byDate.get(iso);
          return { top: d && isScored(d) ? num(dayTotal(d)) : '', bottom: num(isoToJalali(iso).jd) };
        })} />
      </Card>

      <div className="grid2">
        <Card>
          <h2>ستاره بازنویسی</h2>
          {drafts.length > 0 && <p>{num(drafts.length)} تمرین نیمه‌کاره</p>}
          {lastDone && <p className="muted">آخرین تکمیل: {fmtJalali(lastDone.date)}</p>}
          {!drafts.length && !lastDone && <p className="muted">هنوز تمرینی ثبت نشده.</p>}
          <button className="ghost" onClick={() => go('star')}>{drafts.length ? 'ادامه' : 'شروع'}</button>
        </Card>
        <Card>
          <h2>تعادل (۱۴ روز)</h2>
          {topDims.length ? (
            <ul className="mini-balance">
              {topDims.map(([id, n]) => <li key={id}>{dims.find((x) => x.id === id)?.name ?? '?'} <b>{num(n)}</b></li>)}
            </ul>
          ) : <p className="muted">اقدامی به ابعاد زندگی وصل نشده.</p>}
          <button className="ghost" onClick={() => go('balance')}>جزئیات</button>
        </Card>
      </div>
    </div>
  );
}
