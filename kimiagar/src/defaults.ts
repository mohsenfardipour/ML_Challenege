import type { Action, CommitmentDef, Dimension, Settings } from './types';

export const DEFAULT_CARD_MAX = [10, 10, 8, 7];

export const DEFAULT_SETTINGS: Settings = {
  id: 'main',
  name: '',
  courseStart: null,
  courseLength: 45,
  cardMax: DEFAULT_CARD_MAX,
  digits: 'fa',
  walkBlank: '',
  expIntervalDays: 14,
  lastBackupAt: null,
};

/** Placeholder in row 3 that is replaced by settings.walkBlank when shown. */
export const WALK_BLANK = '........';

// Labels exactly as printed on the workbook form.
// Per the owner's decision, row 5 (affirmation) is replaced by Star Rewriting; the original row is kept inactive.
export const DEFAULT_COMMITMENTS: CommitmentDef[] = [
  { id: 'c01', order: 1, kind: 'multiplier', yesPts: 10, noPts: -10, active: true, label: 'متعهد بوده‌ام؟ (هر عدم تعهد ۱۰- امتیاز)' },
  { id: 'c02', order: 2, kind: 'simple', yesPts: 5, noPts: -10, active: true, label: 'به موقع بیدار شده‌ام؟' },
  { id: 'c03', order: 3, kind: 'simple', yesPts: 5, noPts: -10, active: true, label: `۱۵ دقیقه پیاده‌روی در روز با موزیک مثبت و ${WALK_BLANK} داشته‌ام؟` },
  { id: 'c04', order: 4, kind: 'simple', yesPts: 5, noPts: 0, active: true, label: 'یک عمل محبت‌آمیز یا دو مورد تحسین دیگران داشته‌ام؟' },
  { id: 'c13', order: 5, kind: 'star', yesPts: 5, noPts: 0, active: true, label: 'ستاره بازنویسی را تکمیل کرده‌ام؟' },
  { id: 'c06', order: 6, kind: 'simple', yesPts: 5, noPts: -5, active: true, label: 'ملاقات صبح‌گاهی داشته‌ام WOOP + تلقین' },
  { id: 'c07', order: 7, kind: 'simple', yesPts: 5, noPts: 0, active: true, label: 'حداقل برای ۱۰ مورد شکرگزاری روزانه کرده‌ام؟' },
  { id: 'c08', order: 8, kind: 'simple', yesPts: 5, noPts: -2, active: true, label: 'حداقل ۳ صفحه کتابم را مطالعه کرده‌ام؟' },
  { id: 'c09', order: 9, kind: 'simple', yesPts: 5, noPts: 0, active: true, label: 'حداقل ۲ دقیقه خودم را تحسین کرده‌ام؟' },
  { id: 'c10', order: 10, kind: 'streak', yesPts: 5, noPts: 0, active: true, label: 'رفتار غیرسازنده‌ای را حذف کرده‌ام؟' },
  { id: 'c11', order: 11, kind: 'streak', yesPts: 5, noPts: 0, active: true, label: 'رفتار سازنده‌ام را انجام داده‌ام؟' },
  { id: 'c12', order: 12, kind: 'simple', yesPts: 5, noPts: 0, active: true, label: 'با ارسال این فرم کامل شده‌ام؟' },
  { id: 'c05', order: 13, kind: 'simple', yesPts: 5, noPts: 0, active: false, label: 'قصد یا عبارت سازنده تأکیدی‌ام را تکرار کرده‌ام؟' },
];

export const DEFAULT_DIMENSIONS: Dimension[] = [
  'خانواده', 'دوستان', 'رابطه عاطفی', 'سلامت جسم', 'سلامت روان', 'شغل و حرفه',
  'آموزش و یادگیری', 'سلامت مالی', 'معنویت', 'روابط اجتماعی', 'تفریح و اوقات فراغت', 'رشد فردی و خودآگاهی',
].map((name, i): Dimension => ({ id: `d${String(i + 1).padStart(2, '0')}`, name, order: i + 1, active: true, kind: 'dimension' }))
  .concat([{ id: 'k01', name: 'کودک', order: 13, active: true, kind: 'category' }]);

export const COMFORT_LABELS = ['کم', 'متوسط', 'زیاد', 'خیلی زیاد'];

export const STATUS_LABELS: Record<Action['status'], string> = {
  todo: 'انجام نشده', done: 'انجام شد', partly: 'نیمه‌کاره', skipped: 'رها شد',
};

export const MOOD_TITLES = ['خیلی خوب', 'خوب', 'معمولی', 'ناراحت', 'آشفته'];

export function emptyAction(): Action {
  return { topic: '', dimensionId: null, description: '', reflection: '', need: '', value: '', ability: '', comfort: null, score: null, status: 'todo' };
}
