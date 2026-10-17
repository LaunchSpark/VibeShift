import { LABELS, type Label } from './emotions'

export type Rollup = { day: string; count: number; prob_sums: Partial<Record<Label, number>> }
export type SeriesPoint = { day: string; count: number; values: Record<Label, number | null> }

const DAY_MS = 86_400_000

function parseDay(day: string): number {
  const [y, m, d] = day.split('-').map(Number)
  return Date.UTC(y, m - 1, d)
}

/** Shift a YYYY-MM-DD string by `delta` days (UTC, no local-time parsing). */
export function shiftDay(day: string, delta: number): string {
  return new Date(parseDay(day) + delta * DAY_MS).toISOString().slice(0, 10)
}

type DayTotals = { count: number; sums: Partial<Record<Label, number>> }

function sumByDay(rows: Rollup[]): Map<string, DayTotals> {
  const byDay = new Map<string, DayTotals>()
  for (const row of rows) {
    const t = byDay.get(row.day) ?? { count: 0, sums: {} }
    t.count += row.count
    for (const l of LABELS) {
      const v = row.prob_sums[l]
      if (v) t.sums[l] = (t.sums[l] ?? 0) + v
    }
    byDay.set(row.day, t)
  }
  return byDay
}

export function buildSeries(rows: Rollup[], from: string, to: string, window: number): SeriesPoint[] {
  const byDay = sumByDay(rows)
  const n = Math.max(1, Math.floor(window))
  const out: SeriesPoint[] = []
  const days = Math.round((parseDay(to) - parseDay(from)) / DAY_MS)
  for (let i = 0; i <= days; i++) {
    const day = shiftDay(from, i)
    let wCount = 0
    const wSums: Partial<Record<Label, number>> = {}
    for (let k = 0; k < n; k++) {
      const t = byDay.get(shiftDay(day, -k))
      if (!t) continue
      wCount += t.count
      for (const l of LABELS) wSums[l] = (wSums[l] ?? 0) + (t.sums[l] ?? 0)
    }
    const values = {} as Record<Label, number | null>
    for (const l of LABELS) values[l] = wCount > 0 ? (100 * (wSums[l] ?? 0)) / wCount : null
    out.push({ day, count: byDay.get(day)?.count ?? 0, values })
  }
  return out
}

export function currentAndPeak(series: SeriesPoint[], label: Label): { current: number | null; peak: number | null } {
  let current: number | null = null
  let peak: number | null = null
  for (const p of series) {
    const v = p.values[label]
    if (v === null) continue
    current = v
    if (peak === null || v > peak) peak = v
  }
  return { current, peak }
}

export function topEmotions(rows: Rollup[], n: number): Label[] {
  let total = 0
  const sums: Record<string, number> = {}
  for (const row of rows) {
    total += row.count
    for (const l of LABELS) sums[l] = (sums[l] ?? 0) + (row.prob_sums[l] ?? 0)
  }
  const mean = (l: Label) => (total > 0 ? (sums[l] ?? 0) / total : 0)
  return LABELS.map((l, i) => ({ l, i, m: mean(l) }))
    .sort((a, b) => b.m - a.m || a.i - b.i)
    .slice(0, n)
    .map((x) => x.l)
}
