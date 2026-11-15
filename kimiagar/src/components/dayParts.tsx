import type { Action, CommitmentAnswer, Day, Dimension, Settings } from '../types';
import { COMFORT_LABELS, STATUS_LABELS } from '../defaults';
import { num, signed } from '../lib/format';
import { commitmentPoints, scoreDay } from '../lib/scoring';
import { suggestStreak } from '../lib/logic';
import { labelOf } from '../lib/export';
import { IntInput, Segmented, SuggestInput } from './ui';
import { go } from '../hooks';

export const LEAD_IN = 'برای اینکه به هدفم برسم، امروز باید چه قدمی بردارم؟';
export const REFLECTION_Q = 'چگونه از قصدم برای انجام این اقدام استفاده کردم؟';

export interface Suggestions {
  topics: string[];
  need: string[];
  value: string[];
  ability: string[];
}

export type Mode = 'plan' | 'review';

export function ActionCard({ i, a, max, mode, dims, sugg, prevTopic, onChange }: {
  i: number; a: Action; max: number; mode: Mode; dims: Dimension[]; sugg: Suggestions;
  prevTopic?: string; onChange: (patch: Partial<Action>) => void;
}) {
  const review = mode === 'review';
  return (
    <article className={`action-card status-${a.status}`}>
      <div className="ac-head">
        <span className="ac-num">{num(i + 1)}</span>
        <div className="ac-topic">
          {i < 2 && <div className="lead">{LEAD_IN}</div>}
          <label className="field-inline">
            <b>موضوع اقدام:</b>
            <SuggestInput id={`topic${i}`} value={a.topic} onChange={(topic) => onChange({ topic })}
              suggestions={sugg.topics} chips={!a.topic && prevTopic ? [prevTopic] : []} />
          </label>
        </div>
      </div>

      <label className="field-inline">
        <b>شرح اقدام:</b>
        <textarea rows={2} value={a.description} onChange={(e) => onChange({ description: e.target.value })} />
      </label>

      <div className="row-wrap">
        <label className="field-inline grow">
          <span className="muted">بُعد زندگی:</span>
          <select value={a.dimensionId ?? ''} onChange={(e) => onChange({ dimensionId: e.target.value || null })}>
            <option value="">—</option>
            {dims.filter((d) => d.active || d.id === a.dimensionId).map((d) => (
              <option key={d.id} value={d.id}>{d.name}{d.kind === 'category' ? ' (دسته جدا)' : ''}</option>
            ))}
          </select>
        </label>
        <div className="comfort">
          <span className="muted">میزان خروج از ناحیه امنم</span>
          <Segmented className="circles" value={a.comfort}
            onChange={(v) => onChange({ comfort: v as Action['comfort'] })}
            options={COMFORT_LABELS.map((l, k) => ({ value: (k + 1) as 1, label: <><i className="dot" />{l}</> }))} />
        </div>
      </div>

      {review && (
        <label className="field-block">
          <span><b>موانع و کشف:</b> <span className="q">{REFLECTION_Q}</span></span>
          <textarea rows={3} value={a.reflection} onChange={(e) => onChange({ reflection: e.target.value })} />
        </label>
      )}

      <div className="nva">
        <label><b>نیاز:</b><SuggestInput id={`need${i}`} value={a.need} onChange={(need) => onChange({ need })} chips={sugg.need} /></label>
        <label><b>ارزش:</b><SuggestInput id={`value${i}`} value={a.value} onChange={(value) => onChange({ value })} chips={sugg.value} /></label>
        <label><b>توانایی:</b><SuggestInput id={`ability${i}`} value={a.ability} onChange={(ability) => onChange({ ability })} chips={sugg.ability} /></label>
      </div>

      {review && (
        <div className="row-wrap ac-foot">
          <div className="score-pick">
            <b>امتیاز ({num(0)} تا {num(max)}):</b>
            <Segmented value={a.score} onChange={(score) => onChange({ score })}
              options={Array.from({ length: max + 1 }, (_, k) => ({ value: k, label: num(k) }))} />
          </div>
          <StatusPicker value={a.status} onChange={(status) => onChange({ status })} />
        </div>
      )}
    </article>
  );
}

export function StatusPicker({ value, onChange }: { value: Action['status']; onChange: (s: Action['status']) => void }) {
  return (
    <Segmented className="status" value={value} clearable={false} onChange={(v) => onChange((v ?? 'todo') as Action['status'])}
      options={(['todo', 'done', 'partly', 'skipped'] as const).map((s) => ({ value: s, label: STATUS_LABELS[s] }))} />
  );
}

export function Commitments({ day, settings, prev, onChange, compact }: {
  day: Day; settings: Settings; prev?: Day; compact?: boolean;
  onChange: (id: string, patch: Partial<CommitmentAnswer>) => void;
}) {
  return (
    <div className={`commitments ${compact ? 'compact' : ''}`}>
      <div className="cm-head"><span /><span>بله</span><span>خیر</span></div>
      {day.rules.defs.map((def) => {
        const a = day.commitments[def.id] ?? { answer: null };
        const pa = prev?.commitments[def.id];
        const setAnswer = (ans: 'yes' | 'no') => {
          const answer = a.answer === ans ? null : ans;
          const patch: Partial<CommitmentAnswer> = { answer };
          if (def.kind === 'multiplier' && answer === 'no' && !a.n) patch.n = 1;
          if (def.kind === 'streak' && answer === 'yes') {
            if (a.streak == null) patch.streak = suggestStreak(pa);
            if (!a.behavior && pa?.behavior) patch.behavior = pa.behavior;
          }
          onChange(def.id, patch);
        };
        const noLabel = def.kind === 'multiplier' ? `${num(def.noPts)} × ${a.answer === 'no' ? num(a.n ?? 1) : '…'}` : num(def.noPts);
        return (
          <div className="cm-row" key={def.id}>
            <div className="cm-label">
              {labelOf(def.label, settings)}
              {def.kind === 'star' && <a className="link" href="#/star" onClick={(e) => { e.preventDefault(); go('star'); }}> باز کردن ستاره ←</a>}
              {def.kind === 'multiplier' && a.answer === 'no' && (
                <span className="cm-extra">
                  تعداد: <IntInput value={a.n ?? 1} min={1} max={99} onChange={(n) => onChange(def.id, { n: n ?? 1 })} ariaLabel="تعداد عدم تعهد" />
                  <input placeholder="چه چیزی را شکستم؟" value={a.note ?? ''} onChange={(e) => onChange(def.id, { note: e.target.value })} />
                </span>
              )}
              {def.kind === 'streak' && !compact && (
                <span className="cm-extra">
                  <input placeholder="رفتار" value={a.behavior ?? ''} onChange={(e) => onChange(def.id, { behavior: e.target.value })} />
                  برای <IntInput value={a.streak} min={0} max={9999} onChange={(streak) => onChange(def.id, { streak })} ariaLabel="روز متوالی" /> روز متوالی.
                </span>
              )}
            </div>
            <button type="button" className={`yn yes ${a.answer === 'yes' ? 'on' : ''}`} aria-pressed={a.answer === 'yes'} onClick={() => setAnswer('yes')}>
              {signed(def.yesPts)}
            </button>
            <button type="button" className={`yn no ${a.answer === 'no' ? 'on' : ''}`} aria-pressed={a.answer === 'no'} onClick={() => setAnswer('no')}>
              {noLabel}
            </button>
          </div>
        );
      })}
    </div>
  );
}

export function ScorePanel({ day }: { day: Day }) {
  const s = scoreDay(day, day.rules);
  return (
    <div className="score-panel">
      <div className="total"><span>امتیاز امروز</span><strong className={s.total < 0 ? 'neg' : ''}>{num(s.total)}</strong></div>
      <div className="breakdown">
        <span>اقدام‌ها <b>{num(s.actions)}</b></span>
        <span>تعهدات <b>{num(s.commitments)}</b></span>
        <span className="lost">کسر از {num(s.max)}: <b>{num(-s.lost)}</b></span>
      </div>
    </div>
  );
}

/** Commitment points (for small inline display). */
export const pointsOf = (day: Day, id: string) => {
  const def = day.rules.defs.find((d) => d.id === id);
  return def ? commitmentPoints(def, day.commitments[id]) : 0;
};
