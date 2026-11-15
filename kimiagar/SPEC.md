# Kimiagar (کیمیاگر) — MVP Spec

Source of truth: the photos in `reference/`. Where the brief and the photos disagree, the photos win; each case is marked **[PHOTO]**.

## 1. What the reference pages show

| File | Content |
|---|---|
| `example_digital_day17.jpg` | Typed form (Zohreh), day 17, 1405/08/23, everything filled, total 100 |
| `rewrite_star.jpg` | Star diagram only (no steps page) |
| `handwritten_dayNN_plan.jpg` | Pages for days 17, 18, 19, 20, 23 and 24, filled the night before |
| `handwritten_dayNN_done.jpg` | The same pages finished at the end of the day: days 16, 17, 18, 19, 20 and 23 |

There's no separate blank form. The printed form is fully readable on every page, so it serves as the blank form. There are no pages for the 14-day experience, the 12 dimensions or the Star steps.

### Fixed form elements (exact text)
- Header: «فرم انتخاب، ایجاد و بررسی کیمیاگر: [name]», «قصد امروز جاری کردن:», «روز … از دوره ۴۵ روزه‌ام.», «تاریخ:», and a logo.
- Cards 1–2 begin with «برای اینکه به هدفم برسم، امروز باید چه قدمی بردارم؟ موضوع اقدام:». Cards 3–4 begin with «موضوع اقدام:» only.
- Each card has, in this order: موضوع اقدام · شرح اقدام · «موانع و کشف: چگونه از قصدم برای انجام این اقدام استفاده کردم؟» · bottom row «نیاز: / ارزش: / توانایی:» · «امتیاز (۰ تا ۱۰|۱۰|۸|۷):».
- «میزان خروج از ناحیه امنم» uses 4 circles. In RTL order from the right they are کم · متوسط · زیاد · خیلی زیاد. Only one can be chosen.
- Footer: commitments table with columns «بله» and «خیر», the 12 rows exactly as in the brief, «تجربه امروز:» free text, 5 face icons, a trophy icon and «امتیاز امروز:».
- Row 6 has no question mark: «ملاقات صبح‌گاهی داشته‌ام WOOP + تلقین». Row 1's No cell is printed as «۱۰- × …».

### What the photos teach (different from or added to the brief)
1. **[PHOTO] The comfort-zone level is chosen while planning.** On every plan page the level is already ticked, but the reflection and score are empty. So it's a plan field that I'll keep editable later. The brief calls it end-of-day.
2. **[PHOTO] نیاز / ارزش / توانایی are filled at either time.** Days 17 and 18 fill them while planning; day 19 fills them at the end of the day. So they belong to both modes.
3. **[PHOTO] The day number isn't calendar-based.** Day 20 is dated 7/9, day 23 is 7/19 and day 24 is 7/30. So «روز N» counts filled pages, not days since the start. The app suggests N = (last day number in the course) + 1, and I can edit it.
4. **[PHOTO] The «موانع و کشف» field holds obstacles and discoveries together.** Sometimes it starts with «موانع:» or «مانع:». One multi-line field is right.
5. **[PHOTO] Points lost are tallied by hand** in the margin (for example «۳۵-» on day 23). The breakdown will show actions / commitments / total and "lost out of 100".
6. Row 1 No also gets a short note on paper (day 16: «غیبت», day 23: «غر زدم»). This matches the brief.
7. Rows 10–11 «برای … روز متوالی» are filled on Zohreh's page (6 and 8). They're empty on the handwritten pages, so they stay optional.
8. Topics repeat: card 1 is mostly «خانواده» for me, card 2 is «پروژه Agg», card 3 «کودک» and card 4 «معنویت». Zohreh's topics differ («تمرین», «پیگیری ایده»). I'll suggest yesterday's topic per card; nothing is auto-filled.
9. Values and needs repeat a lot (عشق و محبت، هویت مستقل، معنویت، لذت؛ محبت، دیده شدن، تایید شدن، تفریح). That makes chips useful.

### Scoring checked against the paper (actions + commitments)
| Day | Actions | Commitment answers | My calc | Paper |
|---|---|---|---|---|
| 17 typed | 10+10+8+7=35 | all Yes = 65 | 100 | 100 ✓ |
| 16 | 10+10+8+4=32 | R1 No n=1 → 45 | 77 | 77 ✓ (test B) |
| 17 | 35 | R2 No (−10) → 50 | 85 | 85 ✓ |
| 19 | 35 | R8 No (−2), R12 No (0) → 53 | 88 | 88 ✓ |
| 23 | 35 | R1 No n=1, R3 No → 30 | 65 | 65 ✓ |
| 18 | 10+10+5+7=32 | R2 No, R3 No → 35 | 67 | **62** ✗ (open question Q1) |
| 20 | 8?+10+8+2 | R1 No n=1 → 45 | 73? | **70** (Q2) |

Rule (fits every legible page): total = Σ card scores + Σ over rows of (Yes → yesPts, No → noPts, or noPts×n for row 1, unanswered → 0). Days 17, 19 and 23 become extra unit tests along with A and B.

### Star diagram check
Your reading matches the image, clockwise from the top. The arrows on the circle also run clockwise. The red labels have extra text you left out:
- 2 red: «فقدان ارزش (در ترس، سود کاذب و … گیر کردم)»
- 3 red: «احساس گیرافتادن و بی‌عملی (در یک بی‌عملی یا یک عمل غیرمؤثر گیر افتادم)»
- 6 red is on two lines: «اجتناب از تجربه» / «پاسخ چالشی». Black: «تمایل/پذیرش».

Each point has its own color: red, magenta, violet, blue, teal and yellow-green. I'll reuse these colors in the workflow.

## 2. Screens
1. **Today (home):** intention, the 4 action cards (topic + status), quick Yes/No commitments, a big live total with its breakdown, a 14-day score bar chart (SVG), the experience banner, Star status, a mini balance view and a weekly backup reminder.
2. **Day page:** mirrors the form; it's used for today, past days and planning. It has a "Plan" and "Review" tab.
   - **Plan** shows the intention, the 4 topics with optional dimension, descriptions, comfort zone and need/value/ability.
   - **Review** adds reflection, score, status, commitments, experience text and mood. Nothing is required.
3. **Plan tomorrow** («برنامه‌ریزی فردا»): a button that opens tomorrow's Day page on the Plan tab.
4. **History:** a list or calendar of course days (Jalali) showing day number, date, total, mood and status. Filters: date range and dimension. Totals also show weekly, monthly and custom-range averages.
5. **Star Rewriting:** start → situation → 6 steps (one per point, n/6) → summary. Drafts auto-save; there's a list of drafts and finished exercises; a finished one opens read-only, with an edit button.
6. **New experiences:** a list and form for each experience, plus next-due logic.
7. **Life balance:** for the last 14 or 30 days, actions per dimension as SVG bars; latest satisfaction (1–10) per dimension; a quick "rate now" form with a note.
8. **Settings:** name, course start, card maxima, digits (Persian/Latin), the commitments editor (add, edit, deactivate, reorder, points), the row 3 blank, the dimensions editor, the experience interval, backup/restore/CSV.

## 3. Data model (IndexedDB via Dexie; dates as `YYYY-MM-DD` Gregorian)
```
settings     {id:'main', name, courseStart, courseLength:45, cardMax:[10,10,8,7],
              digits:'fa'|'latin', walkBlank, expIntervalDays:14, lastBackupAt}
commitmentDefs {id, order, label, yesPts, noPts, kind:'simple'|'multiplier'|'streak'|'star', active}
days         {date (PK), dayNumber, intention,
              actions:[4]{topic, dimensionId?, description, reflection, need, value, ability,
                          comfort:1..4|null, score:int|null, status:'todo'|'done'|'partly'|'skipped'},
              commitments:{[defId]:{answer:'yes'|'no'|null, n?, note?, behavior?, streak?}},
              rules:{cardMax, defs:[{id,label,yesPts,noPts,kind}]},   // snapshot
              experienceText, mood:1..5|null, updatedAt}
starExercises {id, date, situation, steps:{[pointId]:{stuck, rewrite}}, current, status:'draft'|'done', doneAt}
experiences  {id, intent, why, targetDate, completed, completedDate, learned, rating:1..5|null}
dimensions   {id, name, order, active}
dimRatings   {id, dimensionId, date, rating:1..10, note}
```
- A **rules snapshot** is saved on each day the first time it's created. Later changes in Settings then don't silently change old scores. A "use current rules" button can re-apply them on purpose.
- Star step texts and point labels live in `src/config/star.json`, so they can be edited without touching code.

## 4. Scoring engine
`scoreDay(day, rules) → {actions, commitments, total, max, lost}` is a pure function.
- Card score: clamped to 0..cardMax[i], blank = 0. The input rejects values that are out of range or not whole numbers.
- `simple` and `streak`: Yes → yesPts, No → noPts. `multiplier`: No → noPts × max(1, n). Unanswered → 0.
- `star` (row 13): plain Yes/No, starting at +5/0. It's **inactive by default** so max stays 100 (Q4).
- Averages ignore days with no page. Weekly is Saturday–Friday (Q6).
- Streak pre-fill: if yesterday's row was Yes with streak k, suggest k+1; otherwise leave it empty.
- Tests (Vitest): A=100, B=77, plus pages 17→85, 19→88 and 23→65.

## 5. Other rules
- **Experience cycle:** due date = last completed date (or course start) + interval. Banner shows «N روز مانده» or «N روز گذشته».
- **Backup:** one JSON file for everything (with a schema version). Restore checks the file, shows counts and asks before replacing. A banner appears if the last backup was more than 7 days ago. CSV export covers days, actions and commitments (UTF-8 with BOM so Excel shows Persian).
- **Copy as text:** the day in form order, in Persian, to paste into Telegram.
- **Autosave:** about 400 ms after each change, plus on blur and when the page is hidden. A small "saved" indicator shows. The app asks the browser for persistent storage.

## 6. Tech decision
React 18 + TypeScript + Vite. Dexie (IndexedDB). `jalaali-js` (tiny, no dependencies) for Jalali dates. Vazirmatn font bundled from npm. Vitest for tests. Routing by hash without a library. Plain CSS with RTL logical properties. Charts are hand-written SVG. No network calls at runtime.
M7: Electron + electron-builder (NSIS). Building a Windows `.exe` from Linux needs Wine, so I'll set up the config and give you one command to run on your Windows PC (Q7).

## 7. Open questions
1. **Day 18** (7/7): I compute 67 (32 + 35), but the paper says **62** and «۲۸-». Is one of the rows 4–12 actually No or unanswered, or is it an arithmetic slip?
2. **Day 20** (7/9): card 1 looks like «۸», with «۵» written outside cards 1 and 4. Total 70 only works if card 1 = 5. What do the outside marks mean?
3. **Mood scale:** I read 1 = big smile, 2 = smile, 3 = neutral, 4 = sad, 5 = shocked/distressed. Do you want short labels under the faces, or faces only?
4. **Row 13 (Star):** should it be active by default? If it is, the daily max becomes 105. Keep +5/0?
5. **Star workflow:** there's no steps page. Which point does the sequence start from (top, clockwise)? What are the real questions for each step? Until you tell me, each step asks: «کجا در [red label] گیر کرده‌ام؟» and «بازنویسی به سمت [black label]».
6. **Week start:** Saturday?
7. **Windows build:** is it OK to run one command on your PC (`npm run dist:win`) to make the `.exe`?
8. **"تجربه مشترک سوم/چهارم با خانواده"** (days 17 and 20, three days apart): is this the 14-day new experience or something separate?
9. **Day number:** confirm «روز N» counts filled pages, not calendar days since the start, as the dates suggest.
10. **Year:** your handwritten dates are only month/day (7/5 …). I'll assume the current Jalali year and you can override it. OK?
11. **Dimension list:** confirm the Persian names: خانواده، دوستان، رابطه عاطفی، سلامت جسم، سلامت روان، شغل و حرفه، آموزش و یادگیری، سلامت مالی، معنویت، روابط اجتماعی، تفریح و اوقات فراغت، رشد فردی و خودآگاهی. Should «کودک» (card 3, your inner child) map to تفریح, or should it be its own dimension?

## 8. Out of scope (MVP)
Per-dimension goal and trend pages, push notifications, pixel-perfect PDF/PNG, sync, multiple users, PWA install.
