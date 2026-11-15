import { useLiveQuery } from 'dexie-react-hooks';
import { db, uid } from '../db';
import { useSettings } from '../hooks';
import { Card, JalaliDateInput, Segmented } from '../components/ui';
import { todayISO } from '../lib/dates';
import { fmtJalali, num } from '../lib/format';
import { daysUntil, nextExperienceDue } from '../lib/logic';
import type { Experience } from '../types';

function ExpCard({ e }: { e: Experience }) {
  const set = (patch: Partial<Experience>) => db.experiences.put({ ...e, ...patch });
  return (
    <Card className={e.completed ? 'exp done' : 'exp'}>
      <label className="field-block"><b>چه چیزی را می‌خواهم امتحان کنم؟</b>
        <input value={e.intent} onChange={(x) => set({ intent: x.target.value })} /></label>
      <label className="field-block"><b>چرا؟</b>
        <textarea rows={2} value={e.why} onChange={(x) => set({ why: x.target.value })} /></label>
      <div className="row-wrap">
        <label>تاریخ هدف: <JalaliDateInput value={e.targetDate} onChange={(targetDate) => set({ targetDate })} allowEmpty /></label>
        <label className="check">
          <input type="checkbox" checked={e.completed}
            onChange={(x) => set({ completed: x.target.checked, completedDate: x.target.checked ? (e.completedDate ?? todayISO()) : null })} />
          انجام شد
        </label>
        {e.completed && <label>تاریخ انجام: <JalaliDateInput value={e.completedDate} onChange={(completedDate) => set({ completedDate })} /></label>}
      </div>
      {e.completed && (
        <>
          <label className="field-block"><b>چه چیزی یاد گرفتم؟</b>
            <textarea rows={2} value={e.learned} onChange={(x) => set({ learned: x.target.value })} /></label>
          <div className="row-wrap"><span>چقدر معنادار/لذت‌بخش بود؟</span>
            <Segmented value={e.rating} onChange={(rating) => set({ rating })}
              options={[1, 2, 3, 4, 5].map((v) => ({ value: v, label: num(v) }))} /></div>
        </>
      )}
      <button className="ghost small" onClick={() => confirm('حذف شود؟') && db.experiences.delete(e.id)}>حذف</button>
    </Card>
  );
}

export function Experiences() {
  const settings = useSettings();
  const list = useLiveQuery(() => db.experiences.toArray(), []) ?? [];
  const today = todayISO();
  const due = nextExperienceDue(list, settings.courseStart, settings.expIntervalDays);
  const left = due ? daysUntil(due, today) : null;
  const open = list.filter((e) => !e.completed).sort((a, b) => b.createdAt - a.createdAt);
  const done = list.filter((e) => e.completed).sort((a, b) => (b.completedDate ?? '').localeCompare(a.completedDate ?? ''));

  const add = () => db.experiences.add({
    id: uid(), intent: '', why: '', targetDate: due, completed: false, completedDate: null, learned: '', rating: null, createdAt: Date.now(),
  });

  return (
    <div className="page">
      <h1>تجربهٔ جدید هر {num(settings.expIntervalDays)} روز</h1>
      <Card>
        {due ? (
          <p>موعد تجربهٔ بعدی: <b>{fmtJalali(due)}</b> — {left! > 0 ? `${num(left!)} روز مانده` : left === 0 ? 'امروز' : `${num(-left!)} روز گذشته`}</p>
        ) : <p className="muted">برای محاسبهٔ موعد، تاریخ شروع دوره را در تنظیمات وارد کنید.</p>}
        <button onClick={add}>افزودن تجربه</button>
      </Card>
      {open.map((e) => <ExpCard key={e.id} e={e} />)}
      {done.length > 0 && <h2>انجام‌شده</h2>}
      {done.map((e) => <ExpCard key={e.id} e={e} />)}
    </div>
  );
}
