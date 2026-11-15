import { useEffect, useState, type ReactNode } from 'react';
import { JALALI_MONTHS, isoToJalali, jalaliToISO, monthLength, todayISO, type ISODate } from '../lib/dates';
import { num, parseIntFa } from '../lib/format';
import { MOOD_TITLES } from '../defaults';

export function Card({ children, className = '' }: { children: ReactNode; className?: string }) {
  return <section className={`card ${className}`}>{children}</section>;
}

export function Banner({ tone = 'info', children, onClick }: { tone?: 'info' | 'warn'; children: ReactNode; onClick?: () => void }) {
  return <div className={`banner ${tone}`} onClick={onClick} role={onClick ? 'button' : undefined}>{children}</div>;
}

/** Integer text input that accepts Persian digits. Empty -> null. */
export function IntInput({ value, onChange, min, max, className = '', placeholder, ariaLabel }: {
  value: number | null | undefined; onChange: (v: number | null) => void; min?: number; max?: number;
  className?: string; placeholder?: string; ariaLabel?: string;
}) {
  const [text, setText] = useState(value === null || value === undefined ? '' : num(value));
  useEffect(() => { setText(value === null || value === undefined ? '' : num(value)); }, [value]);
  const bad = text.trim() !== '' && (() => {
    const v = parseIntFa(text);
    return v === null || (min !== undefined && v < min) || (max !== undefined && v > max);
  })();
  return (
    <input
      className={`int ${bad ? 'bad' : ''} ${className}`}
      inputMode="numeric"
      value={text}
      placeholder={placeholder}
      aria-label={ariaLabel}
      onChange={(e) => {
        setText(e.target.value);
        if (e.target.value.trim() === '') return onChange(null);
        const v = parseIntFa(e.target.value);
        if (v !== null && (min === undefined || v >= min) && (max === undefined || v <= max)) onChange(v);
      }}
    />
  );
}

/** Jalali date picker: day / month / year. Emits ISO (Gregorian) or null. */
export function JalaliDateInput({ value, onChange, allowEmpty = false }: {
  value: ISODate | null; onChange: (v: ISODate | null) => void; allowEmpty?: boolean;
}) {
  const base = isoToJalali(value ?? todayISO());
  const [jy, setJy] = useState(base.jy);
  const [jm, setJm] = useState(base.jm);
  const [jd, setJd] = useState(base.jd);
  const [empty, setEmpty] = useState(!value);
  useEffect(() => {
    if (!value) { setEmpty(true); return; }
    const j = isoToJalali(value);
    setJy(j.jy); setJm(j.jm); setJd(j.jd); setEmpty(false);
  }, [value]);
  const emit = (y: number, m: number, d: number) => {
    const dd = Math.min(d, monthLength(y, m));
    setJy(y); setJm(m); setJd(dd); setEmpty(false);
    onChange(jalaliToISO(y, m, dd));
  };
  if (allowEmpty && empty) {
    return <button type="button" className="ghost small" onClick={() => emit(jy, jm, jd)}>انتخاب تاریخ</button>;
  }
  return (
    <span className="jdate">
      <select aria-label="روز" value={jd} onChange={(e) => emit(jy, jm, +e.target.value)}>
        {Array.from({ length: monthLength(jy, jm) }, (_, i) => <option key={i} value={i + 1}>{num(i + 1)}</option>)}
      </select>
      <select aria-label="ماه" value={jm} onChange={(e) => emit(jy, +e.target.value, jd)}>
        {JALALI_MONTHS.map((m, i) => <option key={m} value={i + 1}>{m}</option>)}
      </select>
      <IntInput className="year" value={jy} min={1300} max={1500} ariaLabel="سال" onChange={(y) => y && emit(y, jm, jd)} />
      {allowEmpty && <button type="button" className="ghost small" onClick={() => { setEmpty(true); onChange(null); }}>×</button>}
    </span>
  );
}

/** Text input with autocomplete (datalist) and quick-pick chips from previous entries. */
export function SuggestInput({ value, onChange, suggestions, chips = [], placeholder, id, multiline }: {
  value: string; onChange: (v: string) => void; suggestions?: string[]; chips?: string[];
  placeholder?: string; id: string; multiline?: boolean;
}) {
  const listId = `dl-${id}`;
  const shown = chips.filter((c) => c && !value.split(/[،,/]/).map((s) => s.trim()).includes(c)).slice(0, 6);
  return (
    <span className="suggest">
      {multiline
        ? <textarea value={value} placeholder={placeholder} onChange={(e) => onChange(e.target.value)} rows={2} />
        : <input value={value} placeholder={placeholder} list={suggestions?.length ? listId : undefined} onChange={(e) => onChange(e.target.value)} />}
      {suggestions && suggestions.length > 0 && (
        <datalist id={listId}>{suggestions.map((s) => <option key={s} value={s} />)}</datalist>
      )}
      {shown.length > 0 && (
        <span className="chips">
          {shown.map((c) => (
            <button type="button" key={c} className="chip" onClick={() => onChange(value.trim() ? `${value.trim()}، ${c}` : c)}>{c}</button>
          ))}
        </span>
      )}
    </span>
  );
}

/** Segmented control. Clicking the selected option again clears it (when clearable). */
export function Segmented<T extends string | number>({ options, value, onChange, clearable = true, className = '' }: {
  options: { value: T; label: ReactNode; title?: string }[]; value: T | null; onChange: (v: T | null) => void;
  clearable?: boolean; className?: string;
}) {
  return (
    <span className={`seg ${className}`} role="radiogroup">
      {options.map((o) => (
        <button
          type="button" key={String(o.value)} role="radio" aria-checked={value === o.value} title={o.title}
          className={value === o.value ? 'on' : ''}
          onClick={() => onChange(value === o.value && clearable ? null : o.value)}
        >{o.label}</button>
      ))}
    </span>
  );
}

/** The five faces from the workbook: 1 big smile … 5 shocked/distressed. */
export function Face({ mood, size = 30 }: { mood: number; size?: number }) {
  const mouths: Record<number, ReactNode> = {
    1: <path d="M10 19 Q16 27 22 19 Z" fill="currentColor" />,
    2: <path d="M10 19 Q16 24 22 19" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />,
    3: <path d="M11 21 H21" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />,
    4: <path d="M10 23 Q16 18 22 23" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />,
    5: <ellipse cx="16" cy="22" rx="3" ry="3.6" fill="currentColor" />,
  };
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" aria-hidden="true">
      <circle cx="16" cy="16" r="14" fill="none" stroke="currentColor" strokeWidth="1.6" />
      <circle cx="11.5" cy="12.5" r="1.7" fill="currentColor" />
      <circle cx="20.5" cy="12.5" r="1.7" fill="currentColor" />
      {mouths[mood]}
    </svg>
  );
}

export function MoodPicker({ value, onChange }: { value: number | null; onChange: (v: 1 | 2 | 3 | 4 | 5 | null) => void }) {
  return (
    <span className="moods" role="radiogroup" aria-label="حال امروز">
      {[1, 2, 3, 4, 5].map((m) => (
        <button type="button" key={m} role="radio" aria-checked={value === m} title={MOOD_TITLES[m - 1]}
          className={value === m ? 'on' : ''} onClick={() => onChange(value === m ? null : (m as 1))}>
          <Face mood={m} />
        </button>
      ))}
    </span>
  );
}

/** Simple vertical bar chart (plain SVG). Bars ordered right-to-left (RTL). */
export function BarChart({ data, max = 100, height = 120 }: {
  data: { label: string; value: number | null; onClick?: () => void; title?: string }[]; max?: number; height?: number;
}) {
  const w = 100 / Math.max(1, data.length);
  const top = 14, bottom = 18, h = height - top - bottom;
  return (
    <svg className="barchart" viewBox={`0 0 100 ${height}`} preserveAspectRatio="none" role="img" aria-label="نمودار امتیاز روزانه">
      <line x1="0" x2="100" y1={top + h} y2={top + h} className="axis" vectorEffect="non-scaling-stroke" />
      {data.map((d, i) => {
        const x = 100 - (i + 1) * w; // RTL: first item on the right
        const v = d.value === null ? 0 : Math.max(0, Math.min(max, d.value));
        const bh = (v / max) * h;
        return (
          <g key={i} onClick={d.onClick} style={{ cursor: d.onClick ? 'pointer' : undefined }}>
            <title>{d.title ?? `${d.label}: ${d.value ?? '—'}`}</title>
            <rect x={x} y={0} width={w} height={height} fill="transparent" />
            {d.value !== null && <rect x={x + w * 0.18} y={top + h - bh} width={w * 0.64} height={Math.max(bh, 0.6)} rx="0.8" className="bar" />}
          </g>
        );
      })}
    </svg>
  );
}

/** Labels under the bar chart (HTML so text is not stretched). */
export function BarLabels({ labels }: { labels: { top: string; bottom: string }[] }) {
  return (
    <div className="barlabels" style={{ gridTemplateColumns: `repeat(${labels.length}, 1fr)` }}>
      {labels.map((l, i) => <span key={i}><b>{l.top}</b><small>{l.bottom}</small></span>)}
    </div>
  );
}
