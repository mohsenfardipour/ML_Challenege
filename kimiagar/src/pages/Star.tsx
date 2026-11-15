import { useEffect, useRef, useState } from 'react';
import { useLiveQuery } from 'dexie-react-hooks';
import STAR from '../config/star.json';
import { blankDay, db, saveDay, uid } from '../db';
import { go } from '../hooks';
import { Card } from '../components/ui';
import { todayISO } from '../lib/dates';
import { fmtJalali, num } from '../lib/format';
import type { StarExercise } from '../types';

type Field = { id: string; label: string };
type Point = (typeof STAR.points)[number];
const STEPS = [STAR.start, ...STAR.points] as ({ id: string; title?: string; question: string; instruction: string; fields: Field[] } & Partial<Point>)[];
const SUMMARY = STEPS.length; // index after the last point

/** The six-pointed star, clockwise from the top, with the active point highlighted. */
export function StarFigure({ active, size = 220 }: { active?: number; size?: number }) {
  const P = (r: number, deg: number) => {
    const a = (deg * Math.PI) / 180;
    return `${(100 + r * Math.cos(a)).toFixed(2)},${(100 + r * Math.sin(a)).toFixed(2)}`;
  };
  return (
    <svg width={size} height={size} viewBox="0 0 200 200" className="star-fig" aria-hidden="true">
      <circle cx="100" cy="100" r="92" fill="none" className="ring" />
      {STAR.points.map((p, k) => {
        const deg = -90 + 60 * k;
        const on = active === undefined || active === k;
        return (
          <polygon key={p.id} points={`100,100 ${P(24, deg - 30)} ${P(90, deg)} ${P(24, deg + 30)}`}
            fill={p.color} opacity={on ? 1 : 0.18} />
        );
      })}
      <circle cx="100" cy="100" r="26" className="core" />
    </svg>
  );
}

function fieldLabel(step: (typeof STEPS)[number], f: Field) {
  if (f.label) return <span>{f.label}</span>;
  if (f.id === 'red' && step.red) return <span className="red-l">{step.red}</span>;
  if (f.id === 'black' && step.black) return <span className="black-l">{step.black}</span>;
  return null;
}

function Runner({ id }: { id: string }) {
  const [ex, setEx] = useState<StarExercise | null>(null);
  const timer = useRef<number | undefined>(undefined);
  const latest = useRef<StarExercise | null>(null);

  useEffect(() => {
    db.starExercises.get(id).then((e) => { setEx(e ?? null); latest.current = e ?? null; });
    return () => { window.clearTimeout(timer.current); if (latest.current) void db.starExercises.put(latest.current); };
  }, [id]);

  if (!ex) return <p className="muted pad">در حال بارگذاری…</p>;

  const save = (next: StarExercise, now = false) => {
    setEx(next); latest.current = next;
    window.clearTimeout(timer.current);
    if (now) void db.starExercises.put(next);
    else timer.current = window.setTimeout(() => db.starExercises.put(next), 400);
  };
  const setText = (stepId: string, fieldId: string, v: string) =>
    save({ ...ex, answers: { ...ex.answers, [stepId]: { ...ex.answers[stepId], [fieldId]: v } } });
  const goStep = (i: number) => save({ ...ex, current: i }, true);

  const finish = () => save({ ...ex, current: SUMMARY, status: 'done', doneAt: ex.doneAt ?? Date.now() }, true);

  const markCommitment = async () => {
    const day = (await db.days.get(ex.date)) ?? (await blankDay(ex.date));
    const def = day.rules.defs.find((d) => d.kind === 'star');
    if (!def) return alert('در قوانین این روز ردیف «ستاره بازنویسی» فعال نیست.');
    await saveDay({ ...day, commitments: { ...day.commitments, [def.id]: { ...day.commitments[def.id], answer: 'yes' } } });
    alert(`برای ${fmtJalali(ex.date)} «بله» ثبت شد.`);
  };

  if (ex.current >= SUMMARY) {
    return (
      <div className="page star-run">
        <h1>{STAR.title} · {fmtJalali(ex.date)}</h1>
        <StarFigure />
        {STEPS.map((s, i) => (
          <Card key={s.id}>
            <h3 style={{ color: s.color }}>{s.title ?? <>{s.red} ← <span className="black-l">{s.black}</span></>}</h3>
            {s.fields.map((f) => ex.answers[s.id]?.[f.id] ? <p key={f.id}>{fieldLabel(s, f)}{fieldLabel(s, f) && ': '}{ex.answers[s.id][f.id]}</p> : null)}
            <button className="ghost small" onClick={() => goStep(i)}>ویرایش</button>
          </Card>
        ))}
        <div className="actions-row">
          <button onClick={markCommitment}>ثبت «بله» در تعهد ستاره برای این روز</button>
          <button className="ghost" onClick={() => go('star')}>بازگشت به فهرست</button>
        </div>
      </div>
    );
  }

  const i = ex.current;
  const step = STEPS[i];
  const pointIdx = i - 1;
  return (
    <div className="page star-run calm">
      <div className="star-progress">
        {i === 0 ? <b>{step.title}</b> : <b>{num(pointIdx + 1)} / {num(STAR.points.length)}</b>}
        <span className="muted">{fmtJalali(ex.date)}{ex.status === 'done' ? ' · تکمیل‌شده' : ' · ذخیرهٔ خودکار'}</span>
      </div>
      <StarFigure active={i === 0 ? undefined : pointIdx} size={170} />
      {i === 0 ? (
        <div className="center-labels"><span className="red-l">{STAR.center.red}</span><span className="black-l">{STAR.center.black}</span></div>
      ) : (
        <h2 className="point-title">
          <span className="red-l">{step.red}</span>
          {step.redDetail && <small className="red-l"> {step.redDetail}</small>}
          <span className="arrow"> ← </span>
          <span className="black-l">{step.black}</span>
        </h2>
      )}
      {STAR.intro && i === 0 && <p className="instr">{STAR.intro}</p>}
      {step.instruction && <p className="instr">{step.instruction}</p>}
      {step.question && <p className="question">{step.question}</p>}
      {step.fields.map((f) => (
        <label key={f.id} className="field-block">
          {fieldLabel(step, f)}
          <textarea rows={5} value={ex.answers[step.id]?.[f.id] ?? ''} onChange={(e) => setText(step.id, f.id, e.target.value)} />
        </label>
      ))}
      <div className="actions-row">
        {i > 0 && <button className="ghost" onClick={() => goStep(i - 1)}>قبلی</button>}
        {i < STEPS.length - 1 ? <button onClick={() => goStep(i + 1)}>بعدی</button> : <button onClick={finish}>پایان تمرین</button>}
        <button className="ghost small" onClick={() => go('star')}>ذخیره و خروج</button>
      </div>
    </div>
  );
}

export function Star({ id }: { id?: string }) {
  const list = useLiveQuery(() => db.starExercises.orderBy('date').reverse().toArray(), []) ?? [];
  if (id) return <Runner key={id} id={id} />;

  const start = async () => {
    const ex: StarExercise = { id: uid(), date: todayISO(), answers: {}, current: 0, status: 'draft', createdAt: Date.now(), doneAt: null };
    await db.starExercises.add(ex);
    go(`star/${ex.id}`);
  };
  const del = async (e: StarExercise) => { if (confirm('این تمرین حذف شود؟')) await db.starExercises.delete(e.id); };
  const drafts = list.filter((e) => e.status === 'draft');
  const done = list.filter((e) => e.status === 'done');
  const preview = (e: StarExercise) => e.answers.current?.text?.slice(0, 80) || '';

  return (
    <div className="page star">
      <h1>{STAR.title}</h1>
      <StarFigure />
      <div className="actions-row"><button onClick={start}>شروع تمرین جدید</button></div>
      {drafts.length > 0 && (
        <Card>
          <h2>نیمه‌کاره</h2>
          <ul className="simple-list">
            {drafts.map((e) => (
              <li key={e.id}>
                <a href={`#/star/${e.id}`}>{fmtJalali(e.date)} · مرحلهٔ {e.current === 0 ? STAR.start.title : `${num(e.current)} / ${num(STAR.points.length)}`}</a>
                <span className="muted"> {preview(e)}</span>
                <button className="ghost small" onClick={() => del(e)}>حذف</button>
              </li>
            ))}
          </ul>
        </Card>
      )}
      <Card>
        <h2>تمرین‌های تکمیل‌شده</h2>
        {done.length === 0 && <p className="muted">هنوز تمرینی تکمیل نشده.</p>}
        <ul className="simple-list">
          {done.map((e) => (
            <li key={e.id}>
              <a href={`#/star/${e.id}`}>{fmtJalali(e.date)}</a>
              <span className="muted"> {preview(e)}</span>
              <button className="ghost small" onClick={() => del(e)}>حذف</button>
            </li>
          ))}
        </ul>
      </Card>
    </div>
  );
}
