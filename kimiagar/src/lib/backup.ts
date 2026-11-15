import { db, TABLES } from '../db';

export const BACKUP_SCHEMA = 1;

export interface BackupFile {
  app: 'kimiagar';
  schema: number;
  exportedAt: string;
  data: Record<(typeof TABLES)[number], unknown[]>;
}

export async function exportAll(): Promise<BackupFile> {
  const data = {} as BackupFile['data'];
  for (const t of TABLES) data[t] = await db.table(t).toArray();
  return { app: 'kimiagar', schema: BACKUP_SCHEMA, exportedAt: new Date().toISOString(), data };
}

/** Validate a parsed backup; returns record counts or throws a Persian error message. */
export function validateBackup(obj: unknown): Record<string, number> {
  const b = obj as BackupFile;
  if (!b || b.app !== 'kimiagar' || typeof b.data !== 'object') throw new Error('این فایل پشتیبان کیمیاگر نیست.');
  if (b.schema > BACKUP_SCHEMA) throw new Error('این پشتیبان با نسخهٔ جدیدتری از برنامه ساخته شده است.');
  const counts: Record<string, number> = {};
  for (const t of TABLES) {
    const rows = b.data[t];
    if (!Array.isArray(rows)) throw new Error(`بخش «${t}» در فایل پشتیبان نیست یا خراب است.`);
    counts[t] = rows.length;
  }
  for (const d of b.data.days as any[]) {
    if (typeof d?.date !== 'string' || !Array.isArray(d.actions) || !d.rules) throw new Error('یکی از روزها در فایل پشتیبان خراب است.');
  }
  return counts;
}

/** Replace ALL data with the backup, atomically (all or nothing). */
export async function restoreAll(b: BackupFile): Promise<void> {
  validateBackup(b);
  await db.transaction('rw', TABLES.map((t) => db.table(t)), async () => {
    for (const t of TABLES) {
      await db.table(t).clear();
      await db.table(t).bulkPut(b.data[t] as any[]);
    }
  });
}

export function downloadFile(name: string, content: string, mime: string) {
  const blob = new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
