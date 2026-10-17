# VibeShift Dashboard UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A local React + shadcn dashboard with two pages. Emotions has the mockup chart, moving averages, and filters for people, channels and time. Backfill shows and extends history depth and tracks jobs.

**Architecture:** A Vite SPA in `dashboard/` that reads PocketBase directly with the JS SDK. The chart is computed client-side from `daily_rollups` (one row per guild/channel/author/day with `count` and `prob_sums`), never from raw messages. All aggregation lives in pure, unit-tested TS modules (`src/lib/`), and components only render. Filter state lives in the URL query string.

**Tech Stack:** Node 24 (pixi), Vite, React 19, TypeScript, Tailwind v4, shadcn/ui (Sidebar, Card, Chart/Recharts, Toggle, Badge, Select, Popover, Calendar, Command, Dialog, Table, Collapsible, Button), react-router 7, `pocketbase` JS SDK, vitest, tsx.

**Spec:** [docs/superpowers/specs/2026-10-17-emotion-dashboard-design.md](../specs/2026-10-17-emotion-dashboard-design.md). Visual target: [assets/emotion-chart-mockup.png](../specs/assets/emotion-chart-mockup.png).

**Depends on:** Tasks 1–2 of the [backend plan](2026-10-17-backend-architecture.md) (pixi with Node, `pixi run pb`, and the PocketBase schema table in its Task 2, which is the data contract). Nothing here needs the bot; Task 2 seeds fake data.

## Global Constraints

- PocketBase URL is `import.meta.env.VITE_POCKETBASE_URL ?? "http://127.0.0.1:8090"`. There is no auth.
- **Discord ids are strings everywhere: never `Number()` them.** This covers URL params, filters and keys.
- The 13 labels in canonical order: `anger, sarcasm, surprise, love, desire, shame, fear, sadness, disgust, confusion, neutral, guilt, happiness`. Display them capitalized.
- Dark theme by default (`class="dark"` on `<html>`), matching the mockup: page `#141414`, card `#1c1c1c`, subtle `1px` borders, a large light-weight sans title, and **monospace** axis ticks, stat values and the italic "Peak:" line.
- Emotion colors are fixed (`src/lib/emotions.ts`). The mockup's four map as love=`#93c5fd`, anger=`#a5b4fc`, happiness=`#4ade80`, fear=`#d9f99d`. The rest: surprise=`#fbbf24`, sarcasm=`#f472b6`, desire=`#fb7185`, shame=`#c084fc`, sadness=`#60a5fa`, disgust=`#2dd4bf`, confusion=`#f97316`, neutral=`#a1a1aa`, guilt=`#fdba74`.
- Defaults: range 30 days, moving average 7 days (options: raw=1, 7, 14, 21, 30), whole server, and the 4 most prevalent emotions in view switched on.
- Chart metric: per day, `100 * sum(prob_sums[label]) / sum(count)` over the trailing window, **weighted by message count**. Days with zero messages in the window are `null`, which shows as a gap, never 0.
- History extension only moves earlier than the current depth. There are no arbitrary ranges.
- Use shadcn components; don't hand-roll equivalents.

## Review Focus

1. **Sparse data:** a moving-average window wider than the data, or days with no messages, must show gaps and must not use zeros to drag the average down (Task 3 `gaps are null, not zero` and `window only averages days with data`).
2. **A filtered-to-nothing view** (people with no messages in range) shows an empty state, not NaN% or a crash (Task 3 `all-null series gives null current/peak`, Task 6 empty state).
3. **A bad or stale URL** (`?ma=5&range=yesterday&people=`) falls back to defaults rather than breaking the page (Task 4 `invalid params fall back to defaults`).
4. **Extending to a date that isn't earlier than the current depth** keeps the submit button disabled (Task 7 `canExtend`).
5. **19-digit snowflake ids** must survive the URL and the filter string exactly (Task 4 `snowflakes stay strings`, Task 5 `rollupFilter keeps 19-digit ids`).

---

### Task 1: Scaffold the app with shadcn and dark theme

**Files:**
- Create: `dashboard/` (Vite react-ts template), `dashboard/components.json`, `dashboard/src/index.css`, `dashboard/src/lib/emotions.ts`, `dashboard/src/lib/emotions.test.ts`
- Modify: `pyproject.toml` (pixi tasks: `dash-install = "npm --prefix dashboard install"`, `dash = "npm --prefix dashboard run dev"`, `dash-test = "npm --prefix dashboard test -- --run"`)

**Interfaces:**
- Produces: `LABELS: readonly Label[]` (canonical order), `type Label`, `EMOTION_COLORS: Record<Label, string>`, `labelName(l: Label): string` (capitalized).

- [ ] **Step 1: Scaffold.** Run `pixi run npm create vite@latest dashboard -- --template react-ts`, add Tailwind v4 (`@tailwindcss/vite`), and run `npx shadcn@latest init` (base color neutral, CSS variables). Add the components: `sidebar card chart toggle badge select popover calendar command dialog table collapsible button progress`. Add `react-router`, `pocketbase`, `vitest`, `tsx`.
- [ ] **Step 2: Write the failing test** `emotions.test.ts`: `LABELS.length === 13`, the order matches Global Constraints, every label has a 7-char hex color, the colors are unique, and `labelName("happiness") === "Happiness"`.
- [ ] **Step 3: Run** `pixi run dash-test`. Expected: FAIL.
- [ ] **Step 4: Implement** `emotions.ts`, and set the dark theme tokens in `index.css` to the page and card colors above. Use a monospace font variable for numeric UI.
- [ ] **Step 5: Run** `pixi run dash-test` and `npm --prefix dashboard run build`. Expected: PASS, and the build succeeds.
- [ ] **Step 6: Commit** `feat(ui): scaffold dashboard with shadcn dark theme`.

### Task 2: Dev seed data

**Files:**
- Create: `dashboard/scripts/seed.ts`; add `"seed": "tsx scripts/seed.ts"` to `package.json`

**Interfaces:**
- Produces: one guild `guild_id="100000000000000001"` named "Seed Server" (`history_depth` 120 days ago), 4 channels, 6 members, 120 days of `daily_rollups` for every channel×member (some days empty, smooth waves per emotion like the mockup, using a seeded PRNG), 3 `backfill_jobs` (done/running/failed), and 25 pending `messages`. It's idempotent: it deletes this guild's rows first.

- [ ] **Step 1: Implement** the seed against a running `pixi run pb`.
- [ ] **Step 2: Verify.** Run `npm --prefix dashboard run seed` twice. In the PocketBase admin, `daily_rollups` holds the same row count after both runs (no duplicates), and ids are stored as exact strings.
- [ ] **Step 3: Commit** `feat(ui): deterministic dev seed`.

### Task 3: Series math (the chart's brain)

**Files:**
- Create: `dashboard/src/lib/series.ts`, `dashboard/src/lib/series.test.ts`

**Interfaces:**
- Consumes: `Label`, `LABELS`.
- Produces:
  - `type Rollup = { day: string; count: number; prob_sums: Partial<Record<Label, number>> }`
  - `type SeriesPoint = { day: string; count: number; values: Record<Label, number | null> }` (values are percentages 0–100)
  - `buildSeries(rows: Rollup[], from: string, to: string, window: number): SeriesPoint[]` returns one point per calendar day from `from` to `to` inclusive. It first sums rows by day (rows from several channels or authors on the same day combine), then each day's value is the trailing `window`-day count-weighted mean. Callers pass rows starting `window - 1` days before `from`.
  - `currentAndPeak(series: SeriesPoint[], label: Label): { current: number | null; peak: number | null }`. `current` is the last non-null value, and `peak` is the max non-null value.
  - `topEmotions(rows: Rollup[], n: number): Label[]` ranks by total `prob_sums / count` across the rows, breaking ties by canonical order.
  - `shiftDay(day: string, delta: number): string`

- [ ] **Step 1: Write the failing tests:**
  - `combines rows on the same day`: two rows for one day (count 1 and 3) give count 4 and a weighted value
  - `gaps are null, not zero`: window 1 with a missing middle day → that day's values are all `null`
  - `window only averages days with data`: window 7 where only day 1 has data (count 2, anger sum 1.0) → day 3's anger is `50`
  - `weights by message count`: day A (count 1, anger 1.0) and day B (count 3, anger 0.0) with window 2 → `25`, not `50`
  - `all-null series gives null current/peak`
  - `topEmotions returns the 4 highest by mean`
  - `shiftDay("2026-03-01", -1) === "2026-02-28"`
- [ ] **Step 2: Run** `pixi run dash-test`. Expected: FAIL.
- [ ] **Step 3: Implement.** Do date math on `YYYY-MM-DD` strings in UTC, with no local-time `Date` parsing.
- [ ] **Step 4: Run** the tests. Expected: PASS.
- [ ] **Step 5: Commit** `feat(ui): count-weighted moving-average series`.

### Task 4: URL filter state

**Files:**
- Create: `dashboard/src/lib/filters.ts`, `dashboard/src/lib/filters.test.ts`

**Interfaces:**
- Produces:
  - `type RangeKey = "7d" | "30d" | "60d" | "90d" | "all" | "custom"`
  - `type Filters = { guild: string | null; range: RangeKey; from: string | null; to: string | null; ma: 1 | 7 | 14 | 21 | 30; people: string[]; channels: string[]; emotions: Label[] | null }`. `emotions: null` means "use the top 4".
  - `parseFilters(q: URLSearchParams): Filters`, `toSearch(f: Filters): URLSearchParams` (omits defaults), and `resolveRange(f: Filters, today: string, earliest: string | null): { from: string; to: string }`. `"all"` resolves to `earliest ?? today`; `"custom"` needs both `from` and `to`, otherwise it falls back to 30 days.
  - `useFilters(): [Filters, (patch: Partial<Filters>) => void]`, a react-router `useSearchParams` wrapper.
- URL keys: `guild, range, from, to, ma, people` (comma list), `channels` (comma list), `emo` (comma list).

- [ ] **Step 1: Write the failing tests:**
  - `defaults when empty`: range `30d`, ma `7`, empty people/channels, emotions `null`
  - `round trips`
  - `invalid params fall back to defaults`: `?ma=5&range=yesterday&people=&emo=joy` gives ma 7, range 30d, people `[]`, and emotions `null` (unknown labels are dropped, and an empty result becomes null)
  - `snowflakes stay strings`: `people=123456789012345678901` round-trips exactly
  - `resolveRange 30d`: from today−29 to today
- [ ] **Step 2: Run** the tests. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** them again. Expected: PASS.
- [ ] **Step 5: Commit** `feat(ui): URL-backed filter state`.

### Task 5: PocketBase data layer

**Files:**
- Create: `dashboard/src/lib/pb.ts`, `dashboard/src/lib/data.ts`, `dashboard/src/lib/data.test.ts`

**Interfaces:**
- Consumes: the schema contract (backend Task 2) and `Rollup`.
- Produces:
  - `pb` (a PocketBase client singleton)
  - `rollupFilter(p: { guild: string; from: string; to: string; channels: string[]; people: string[] }): string`. It's pure and builds the PocketBase filter: `guild_id = "…" && day >= "…" && day <= "…"`, plus `(channel_id = "a" || channel_id = "b")` and the same for `author_id` when non-empty. Values are quoted and escaped.
  - Fetchers:
    - `fetchGuilds(): Promise<Guild[]>`
    - `fetchChannels(guild): Promise<Channel[]>`
    - `fetchMembers(guild): Promise<Member[]>`
    - `fetchRollups(p): Promise<Rollup[]>` (`getFullList` with `fields=day,count,prob_sums`)
    - `fetchPendingCount(guild): Promise<number>` (`perPage=1`, `totalItems`)
    - `fetchJobs(guild): Promise<Job[]>`
    - `createExtendJob(guild: string, target: string): Promise<void>`
  - `Guild`, `Channel`, `Member` and `Job` mirror the schema fields.

- [ ] **Step 1: Write the failing tests** for `rollupFilter`:
  - guild and day bounds only
  - with 2 channels → contains an OR group
  - `rollupFilter keeps 19-digit ids`: `"1234567890123456789"` appears verbatim
  - escapes a `"` in a value
- [ ] **Step 2: Run** the tests. Expected: FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** them again. Expected: PASS. Then, with the seed loaded, run `npx tsx -e` with `fetchRollups` for the seed guild over the last 30 days and check it returns rows.
- [ ] **Step 5: Commit** `feat(ui): pocketbase data layer`.

### Task 6: App shell and the Emotions page

**Files:**
- Create: `dashboard/src/App.tsx` (router), `dashboard/src/components/app-sidebar.tsx`, `dashboard/src/components/guild-switcher.tsx`, `dashboard/src/components/filters-panel.tsx`, `dashboard/src/components/emotion-chart-card.tsx`, `dashboard/src/components/emotion-stat-cards.tsx`, `dashboard/src/pages/emotions.tsx`

**Interfaces:**
- Consumes: Tasks 3–5.
- Routes: `/` → Emotions and `/backfill` → Backfill (Task 7), both inside a shadcn `SidebarProvider` layout.
- **Sidebar** (Statbot-style): `GuildSwitcher` in the header (guild icon and name, with a dropdown of `fetchGuilds()` that sets `guild`, defaulting to the first guild), then a "Stats" group with Emotions and Backfill. Impact and Auto-roles appear disabled with a "Soon" badge.
- **FiltersPanel:** a shadcn `Collapsible` titled "Filters". Each row reads "**Time** — Last 30 days", "**Average** — 7 days", "**People** — Everyone / N selected", "**Channels** — All / N selected", with a pencil button that opens a `Popover`:
  - Time: range presets plus a `Calendar` range for custom
  - Average: `Select` 1/7/14/21/30, where 1 is labeled "Raw"
  - People and Channels: `Command` multi-select comboboxes with search
- **EmotionChartCard:** matches the mockup.
  - Title "Emotion Prevalence History". Subtitle `${days}-day ${ma === 1 ? "daily" : `${ma}-day average`} probability scores extracted from text history`.
  - A row of 13 pill `Toggle`s, each with a colored dot and name, for the labels in `emotions ?? topEmotions(rows, 4)`. Toggling writes `emo`.
  - A Recharts `LineChart` inside shadcn `ChartContainer`: `type="monotone"`, `connectNulls={false}`, `dot` drawn as a small hollow circle in the line color, Y axis 0/50/100 with a `%` suffix, monospace ticks, X ticks every 4 days. Tooltip shows `day`, the message count, and each visible value to 0 decimals.
  - An empty state when every value is null: "No scored messages in this view yet."
- **EmotionStatCards:** a responsive grid, one card per visible emotion. "{Label} (Current)", a big monospace `{current}%`, and an italic monospace `Peak: {peak}%`. Show "—" for null.
- **Emotions page:** when `pending > 0`, a small muted line above the chart: "{pending} messages waiting to be scored".

- [ ] **Step 1: Implement** the shell, the panel and the page.
- [ ] **Step 2: Verify visually.** With `pixi run pb` and the seed loaded, run `pixi run dash`. Open the page and compare it to `docs/superpowers/specs/assets/emotion-chart-mockup.png`. Check:
  - 4 lines are on by default
  - toggling a pill updates the URL and the chart
  - changing Average to Raw shows gaps on the seed's empty days
  - selecting one person changes the lines
  - reloading the page keeps the filters
  - selecting a channel with no data shows the empty state
  - at 375px wide, nothing scrolls sideways
- [ ] **Step 3: Run** `pixi run dash-test` and `npm --prefix dashboard run build`. Expected: PASS.
- [ ] **Step 4: Commit** `feat(ui): sidebar shell and emotion chart page`.

### Task 7: Backfill page

**Files:**
- Create: `dashboard/src/pages/backfill.tsx`, `dashboard/src/components/extend-history-dialog.tsx`, `dashboard/src/lib/backfill.ts`, `dashboard/src/lib/backfill.test.ts`

**Interfaces:**
- Consumes: `fetchGuilds`, `fetchJobs`, `fetchPendingCount`, `createExtendJob`, and `shiftDay`.
- Produces: `canExtend(currentDepth: string | null, target: string, today: string): boolean`. It's true only when `target < (currentDepth ?? today)`.
- **Page:**
  - A Card shows "History goes back to **{date}**", or "Live only (no history fetched)" when there's no depth, plus the pending count. It has an "Extend history" button.
  - The dialog offers presets (30/60/90/180/365 days back from today) and a `Calendar`. Presets that aren't earlier than the current depth are disabled, and submit is disabled unless `canExtend`. Submitting creates the job and closes the dialog.
  - A `Table` of jobs, newest first: Origin (a badge, "Downtime" or "Extend"), Target, Status (a badge), Scanned, Created, Error (truncated, with the full text in a tooltip).
  - It polls jobs and the pending count every 5 s while the page is open.

- [ ] **Step 1: Write the failing tests:** `canExtend(null, today−1, today)` is true; `canExtend("2026-09-01", "2026-09-01", today)` is false; `canExtend("2026-09-01", "2026-08-31", today)` is true.
- [ ] **Step 2: Run** the tests. Expected: FAIL.
- [ ] **Step 3: Implement** `canExtend`, then the page and dialog.
- [ ] **Step 4: Run** the tests. Expected: PASS.
- [ ] **Step 5: Verify visually** with the seed. The 3 seeded jobs show with correct badges. Extending to 180 days creates a `pending` extend job, which appears within 5 s. The 30-day preset is disabled (the seed depth is 120 days).
- [ ] **Step 6: Commit** `feat(ui): backfill page with history-depth extension`.
