# Architecture (short)

**Stack:** React 18 + TypeScript + Vite. IndexedDB through Dexie. Jalali dates via `jalaali-js`. Vazirmatn font bundled locally. Electron for the Windows app. No backend and no runtime network calls (the CSP blocks them, and Electron cancels every http(s) request).

**Layers**
- `src/lib/`: pure logic with no React and no DB. `scoring.ts` is the scoring engine (`scoreDay(day, rules)`). `logic.ts` has averages, the experience cycle and the streak/day-number suggestions. `dates.ts`/`format.ts` handle Jalali conversion, Saturday weeks and Persian digits. `export.ts` builds the CSV files and the "copy as text". `backup.ts` does JSON export, validation and an atomic restore. Covered by Vitest (`*.test.ts`), including a backup round-trip on fake-indexeddb.
- `src/db.ts`: the Dexie schema (settings, commitmentDefs, days, starExercises, experiences, dimensions, dimRatings) and the first-run seed from `src/defaults.ts`.
- `src/hooks.ts`: hash routing, live queries, and `useDay()`. That hook loads a day, keeps edits in memory and **autosaves 400 ms after the last change**, plus on unmount, when the tab is hidden and before closing. A day is written only after the first edit, so browsing never creates empty pages.
- `src/pages/` + `src/components/`: the screens. `dayParts.tsx` mirrors the paper form (action card, commitments table, score panel).

**Data safety rules**
- Dates are stored as Gregorian `YYYY-MM-DD` and shown as Jalali.
- Each day stores a **rules snapshot** (card maxima + commitment labels/points) from when it was created. Editing commitments in Settings never changes old scores unless you press "اعمال قوانین فعلی" on that day.
- Card scores are validated (whole numbers, 0..max) in the UI and clamped again in the engine.
- The app requests persistent storage. In Electron the app is served from `app://kimiagar`, a stable origin, so the database survives app updates. It lives in `%APPDATA%/Kimiagar`.

**Config without code:** the Star Rewriting texts are in `src/config/star.json`. Commitments, points, dimensions, card maxima and the experience interval are editable in Settings.

## Android app (`android/`)
- Native **Kotlin + Jetpack Compose**, minSdk 26, no INTERNET permission, `allowBackup=false` (nothing goes to Google cloud backup).
- `core/`: a plain-Kotlin port of the same rules (`Scoring`, `Logic`, `Jalali`, `Export`, `Backup`). Models use the same field names as the desktop app, so **the JSON backup file is identical** on both platforms. The Jalali port is checked against 60 dates produced by `jalaali-js`, and a real desktop backup is restored in a test.
- `data/Repo`: all data is one JSON document in private app storage, written with `AtomicFile` (crash-safe), 400 ms after the last change and always when the app goes to background. An unreadable file is never overwritten; a copy is kept.
- `ui/`: Compose screens mirroring the paper form. Star texts are read from the shared `src/config/star.json`.
- Tests: `./gradlew testDebugUnitTest` runs the 20 core tests plus 2 full UI flows (Robolectric).
- Signing: `kimiagar.keystore` is one fixed key for every build, so new versions always install over old ones without losing data.
