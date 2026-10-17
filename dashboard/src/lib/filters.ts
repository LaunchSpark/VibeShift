import { useCallback } from 'react'
import { useSearchParams } from 'react-router'
import { LABELS, type Label } from './emotions'
import { shiftDay } from './series'

export type RangeKey = '7d' | '30d' | '60d' | '90d' | 'all' | 'custom'
export type MovingAverage = 1 | 7 | 14 | 21 | 30

export type Filters = {
  guild: string | null
  range: RangeKey
  from: string | null
  to: string | null
  ma: MovingAverage
  people: string[]
  channels: string[]
  /** null means "use the top 4" */
  emotions: Label[] | null
}

const RANGES: RangeKey[] = ['7d', '30d', '60d', '90d', 'all', 'custom']
const MAS: MovingAverage[] = [1, 7, 14, 21, 30]
const DEFAULT_RANGE: RangeKey = '30d'
const DEFAULT_MA: MovingAverage = 7
const DAY_RE = /^\d{4}-\d{2}-\d{2}$/

function list(v: string | null): string[] {
  return v ? v.split(',').filter(Boolean) : []
}

function day(v: string | null): string | null {
  return v && DAY_RE.test(v) && !Number.isNaN(Date.parse(v)) ? v : null
}

export function parseFilters(q: URLSearchParams): Filters {
  const range = q.get('range') as RangeKey | null
  const ma = Number(q.get('ma'))
  const known = new Set<string>(LABELS)
  const emo = list(q.get('emo')).filter((l): l is Label => known.has(l))
  return {
    guild: q.get('guild') || null,
    range: range && RANGES.includes(range) ? range : DEFAULT_RANGE,
    from: day(q.get('from')),
    to: day(q.get('to')),
    ma: (MAS as number[]).includes(ma) ? (ma as MovingAverage) : DEFAULT_MA,
    people: list(q.get('people')),
    channels: list(q.get('channels')),
    emotions: emo.length ? emo : null,
  }
}

export function toSearch(f: Filters): URLSearchParams {
  const q = new URLSearchParams()
  if (f.guild) q.set('guild', f.guild)
  if (f.range !== DEFAULT_RANGE) q.set('range', f.range)
  if (f.range === 'custom') {
    if (f.from) q.set('from', f.from)
    if (f.to) q.set('to', f.to)
  }
  if (f.ma !== DEFAULT_MA) q.set('ma', String(f.ma))
  if (f.people.length) q.set('people', f.people.join(','))
  if (f.channels.length) q.set('channels', f.channels.join(','))
  if (f.emotions?.length) q.set('emo', f.emotions.join(','))
  return q
}

/** The earliest of the known history bounds (YYYY-MM-DD strings), ignoring missing ones. */
export function earliestDay(...days: (string | null | undefined)[]): string | null {
  const known = days.filter((d): d is string => !!d)
  return known.length ? known.reduce((a, b) => (b < a ? b : a)) : null
}

export function resolveRange(
  f: Filters,
  today: string,
  earliest: string | null,
): { from: string; to: string } {
  const last = (n: number) => ({ from: shiftDay(today, -(n - 1)), to: today })
  const { from, to } = rawRange(f, today, earliest, last)
  // Never chart the future (covers hand-edited URLs too).
  const cTo = to > today ? today : to
  const cFrom = from > cTo ? cTo : from
  return { from: cFrom, to: cTo }
}

function rawRange(
  f: Filters,
  today: string,
  earliest: string | null,
  last: (n: number) => { from: string; to: string },
): { from: string; to: string } {
  switch (f.range) {
    case '7d': return last(7)
    case '60d': return last(60)
    case '90d': return last(90)
    case 'all': return { from: earliest ?? today, to: today }
    case 'custom': return f.from && f.to ? { from: f.from, to: f.to } : last(30)
    default: return last(30)
  }
}

export function useFilters(): [Filters, (patch: Partial<Filters>) => void] {
  const [params, setParams] = useSearchParams()
  const filters = parseFilters(params)
  const update = useCallback(
    (patch: Partial<Filters>) => {
      setParams((prev) => toSearch({ ...parseFilters(prev), ...patch }))
    },
    [setParams],
  )
  return [filters, update]
}
