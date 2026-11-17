import { useRef, useState } from 'react';
import { useLiveQuery } from 'dexie-react-hooks';
import { db, uid } from '../db';
import { patchSettings, useAllDays, useDims, useSettings } from '../hooks';
import { Card, IntInput, JalaliDateInput, Segmented } from '../components/ui';
import { exportAll, restoreAll, validateBackup, type BackupFile } from '../lib/backup';
import { saveFile, type SaveResult } from '../pwa';
import { actionsCsv, commitmentsCsv, daysCsv } from '../lib/export';
import { todayISO } from '../lib/dates';
import { num } from '../lib/format';
import type { CommitmentDef, CommitmentKind, Dimension } from '../types';
import { WALK_BLANK } from '../defaults';

const KIND_LABEL: Record<CommitmentKind, string> = {
  simple: 'ساده', multiplier: 'ضرب در تعداد (خیر)', streak: 'روزهای متوالی', star: 'ستاره بازنویسی',
};

/** Swap `order` of item i with its neighbour (dir -1 up, +1 down) and save. */
async function move<T extends { id: string; order: number }>(table: 'commitmentDefs' | 'dimensions', list: T[], i: number, dir: number) {
  const j = i + dir;
  if (j < 0 || j >= list.length) return;
  const a = list[i], b = list[j];
  await db.transaction('rw', db.table(table), async () => {
    await db.table(table).update(a.id, { order: b.order });
    await db.table(table).update(b.id, { order: a.order });
  });
}

function CommitmentEditor() {
  const defs = useLiveQuery(() => db.commitmentDefs.orderBy('order').toArray(), []) ?? [];
  const put = (d: CommitmentDef, patch: Partial<CommitmentDef>) => db.commitmentDefs.put({ ...d, ...patch });
  const max = defs.filter((d) => d.active).reduce((s, d) => s + Math.max(0, d.yesPts), 0);
  return (
    <Card>
      <h2>تعهدات روزانه</h2>
      <p className="muted">
        تغییرات فقط روی روزهای تازه اعمال می‌شود. هر روز قوانین زمان ساخت خودش را نگه می‌دارد
        (در صفحهٔ روز، «اعمال قوانین فعلی» را بزنید). جمع «بله»ها: <b>{num(max)}</b>
      </p>
      <ol className="editor-list">
        {defs.map((d, i) => (
          <li key={d.id} className={d.active ? '' : 'inactive'}>
            <textarea rows={2} value={d.label} onChange={(e) => put(d, { label: e.target.value })} />
            <div className="row-wrap">
              <label>بله <IntInput value={d.yesPts} min={-100} max={100} onChange={(v) => v !== null && put(d, { yesPts: v })} /></label>
              <label>خیر <IntInput value={d.noPts} min={-100} max={100} onChange={(v) => v !== null && put(d, { noPts: v })} /></label>
              <select value={d.kind} onChange={(e) => put(d, { kind: e.target.value as CommitmentKind })}>
                {Object.entries(KIND_LABEL).map(([k, l]) => <option key={k} value={k}>{l}</option>)}
              </select>
              <label className="check"><input type="checkbox" checked={d.active} onChange={(e) => put(d, { active: e.target.checked })} /> فعال</label>
              <button className="ghost small" onClick={() => move('commitmentDefs', defs, i, -1)} aria-label="بالا">▲</button>
              <button className="ghost small" onClick={() => move('commitmentDefs', defs, i, 1)} aria-label="پایین">▼</button>
            </div>
          </li>
        ))}
      </ol>
      <button className="ghost" onClick={() => db.commitmentDefs.add({
        id: uid(), order: (defs.at(-1)?.order ?? 0) + 1, label: 'تعهد جدید', yesPts: 5, noPts: 0, kind: 'simple', active: true,
      })}>افزودن تعهد</button>
    </Card>
  );
}

function DimensionEditor() {
  const dims = useDims();
  const put = (d: Dimension, patch: Partial<Dimension>) => db.dimensions.put({ ...d, ...patch });
  return (
    <Card>
      <h2>ابعاد زندگی و دسته‌ها</h2>
      <ol className="editor-list">
        {dims.map((d, i) => (
          <li key={d.id} className={d.active ? '' : 'inactive'}>
            <div className="row-wrap">
              <input value={d.name} onChange={(e) => put(d, { name: e.target.value })} />
              <select value={d.kind} onChange={(e) => put(d, { kind: e.target.value as Dimension['kind'] })}>
                <option value="dimension">بُعد زندگی</option>
                <option value="category">دستهٔ جدا</option>
              </select>
              <label className="check"><input type="checkbox" checked={d.active} onChange={(e) => put(d, { active: e.target.checked })} /> فعال</label>
              <button className="ghost small" onClick={() => move('dimensions', dims, i, -1)} aria-label="بالا">▲</button>
              <button className="ghost small" onClick={() => move('dimensions', dims, i, 1)} aria-label="پایین">▼</button>
            </div>
          </li>
        ))}
      </ol>
      <button className="ghost" onClick={() => db.dimensions.add({ id: uid(), name: 'جدید', order: (dims.at(-1)?.order ?? 0) + 1, active: true, kind: 'dimension' })}>افزودن</button>
    </Card>
  );
}

function BackupSection() {
  const settings = useSettings();
  const days = useAllDays() ?? [];
  const dims = useDims();
  const fileRef = useRef<HTMLInputElement>(null);
  const [msg, setMsg] = useState('');
  const [retry, setRetry] = useState<null | { name: string; content: string; mime: string; isBackup: boolean }>(null);
  const stamp = todayISO();

  const done = async (r: SaveResult, f: { name: string; content: string; mime: string; isBackup: boolean }) => {
    if (r === 'needs-tap') { setRetry(f); setMsg(''); return; }
    setRetry(null);
    if (r === 'cancelled') return setMsg('ذخیره لغو شد.');
    if (f.isBackup) await patchSettings({ lastBackupAt: Date.now() });
    setMsg(f.isBackup ? 'فایل پشتیبان ساخته شد. آن را جای امنی نگه دارید (مثلاً Files یا Saved Messages تلگرام).' : 'فایل ساخته شد.');
  };
  const save = async (name: string, content: string, mime: string, isBackup = false) => {
    const f = { name, content, mime, isBackup };
    await done(await saveFile(name, content, mime), f);
  };
  const backup = async () => {
    const b = await exportAll();
    await save(`kimiagar-backup-${stamp}.json`, JSON.stringify(b, null, 1), 'application/json', true);
  };

  const restore = async (file: File) => {
    try {
      const b = JSON.parse(await file.text()) as BackupFile;
      const c = validateBackup(b);
      const ok = confirm(
        `بازگردانی همهٔ اطلاعات فعلی را با این فایل جایگزین می‌کند.\n` +
        `روزها: ${c.days} · تمرین ستاره: ${c.starExercises} · تجربه‌ها: ${c.experiences} · رضایت‌ها: ${c.dimRatings}\n` +
        `تاریخ پشتیبان: ${b.exportedAt}\n\nادامه می‌دهید؟`);
      if (!ok) return;
      await restoreAll(b);
      setMsg('بازگردانی انجام شد ✓');
    } catch (e) {
      setMsg(`خطا: ${(e as Error).message}`);
    } finally {
      if (fileRef.current) fileRef.current.value = '';
    }
  };

  return (
    <Card>
      <h2>پشتیبان و خروجی</h2>
      <p className="muted">
        همهٔ اطلاعات فقط روی همین دستگاه است. گوشی و کامپیوتر همگام نمی‌شوند؛ برای انتقال، پشتیبان بگیرید و در دستگاه دیگر بازگردانی کنید.
        {settings.lastBackupAt && <> آخرین پشتیبان: {new Date(settings.lastBackupAt).toLocaleDateString('fa-IR')}</>}
      </p>
      <div className="actions-row">
        <button onClick={backup}>پشتیبان‌گیری (JSON)</button>
        <button className="ghost" onClick={() => fileRef.current?.click()}>بازگردانی از فایل…</button>
        <input ref={fileRef} type="file" accept="application/json,.json,text/plain" hidden onChange={(e) => e.target.files?.[0] && restore(e.target.files[0])} />
      </div>
      <div className="actions-row">
        <button className="ghost small" onClick={() => save(`kimiagar-days-${stamp}.csv`, daysCsv(days), 'text/csv')}>CSV روزها</button>
        <button className="ghost small" onClick={() => save(`kimiagar-actions-${stamp}.csv`, actionsCsv(days, dims), 'text/csv')}>CSV اقدام‌ها</button>
        <button className="ghost small" onClick={() => save(`kimiagar-commitments-${stamp}.csv`, commitmentsCsv(days), 'text/csv')}>CSV تعهدات</button>
      </div>
      {retry && (
        <p className="msg">
          فایل آماده است. <button onClick={async () => done(await saveFile(retry.name, retry.content, retry.mime), retry)}>ذخیره / ارسال فایل</button>
        </p>
      )}
      {msg && <p className="msg">{msg}</p>}
    </Card>
  );
}

export function Settings() {
  const s = useSettings();
  return (
    <div className="page settings">
      <h1>تنظیمات</h1>
      <Card>
        <label className="field-inline"><b>نام کیمیاگر:</b><input value={s.name} onChange={(e) => patchSettings({ name: e.target.value })} /></label>
        <label className="field-inline"><b>تاریخ شروع دوره:</b><JalaliDateInput value={s.courseStart} onChange={(courseStart) => patchSettings({ courseStart })} allowEmpty /></label>
        <label className="field-inline"><b>طول دوره (روز):</b><IntInput value={s.courseLength} min={1} max={999} onChange={(v) => v && patchSettings({ courseLength: v })} /></label>
        <div className="field-inline"><b>سقف امتیاز کارت‌ها:</b>
          {s.cardMax.map((m, i) => (
            <IntInput key={i} value={m} min={0} max={100} ariaLabel={`سقف کارت ${i + 1}`}
              onChange={(v) => v !== null && patchSettings({ cardMax: s.cardMax.map((x, j) => (j === i ? v : x)) })} />
          ))}
        </div>
        <label className="field-inline"><b>جای خالی ردیف پیاده‌روی («{WALK_BLANK}»):</b><input value={s.walkBlank} onChange={(e) => patchSettings({ walkBlank: e.target.value })} /></label>
        <label className="field-inline"><b>فاصلهٔ تجربهٔ جدید (روز):</b><IntInput value={s.expIntervalDays} min={1} max={365} onChange={(v) => v && patchSettings({ expIntervalDays: v })} /></label>
        <div className="field-inline"><b>ارقام:</b>
          <Segmented clearable={false} value={s.digits} onChange={(digits) => digits && patchSettings({ digits })}
            options={[{ value: 'fa', label: '۱۲۳' }, { value: 'latin', label: '123' }]} />
        </div>
      </Card>
      <BackupSection />
      <CommitmentEditor />
      <DimensionEditor />
    </div>
  );
}
