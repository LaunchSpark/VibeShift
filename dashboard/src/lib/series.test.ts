import { describe, expect, it } from 'vitest'
import { buildSeries, currentAndPeak, shiftDay, topEmotions, type Rollup } from './series'

const r = (day: string, count: number, p: Rollup['prob_sums'] = {}): Rollup => ({ day, count, prob_sums: p })

describe('series', () => {
  it('combines rows on the same day', () => {
    const s = buildSeries([r('2026-03-01', 1, { anger: 1 }), r('2026-03-01', 3, { anger: 0 })], '2026-03-01', '2026-03-01', 1)
    expect(s).toHaveLength(1)
    expect(s[0].count).toBe(4)
    expect(s[0].values.anger).toBe(25)
  })
  it('gaps are null, not zero', () => {
    const s = buildSeries([r('2026-03-01', 1, { anger: 1 }), r('2026-03-03', 1, { anger: 1 })], '2026-03-01', '2026-03-03', 1)
    expect(s[1].count).toBe(0)
    for (const v of Object.values(s[1].values)) expect(v).toBeNull()
  })
  it('window only averages days with data', () => {
    const s = buildSeries([r('2026-03-01', 2, { anger: 1 })], '2026-03-01', '2026-03-03', 7)
    expect(s[2].values.anger).toBe(50)
  })
  it('weights by message count', () => {
    const s = buildSeries([r('2026-03-01', 1, { anger: 1 }), r('2026-03-02', 3, { anger: 0 })], '2026-03-01', '2026-03-02', 2)
    expect(s[1].values.anger).toBe(25)
  })
  it('all-null series gives null current/peak', () => {
    const s = buildSeries([], '2026-03-01', '2026-03-03', 1)
    expect(currentAndPeak(s, 'anger')).toEqual({ current: null, peak: null })
  })
  it('current is last non-null, peak is max', () => {
    const s = buildSeries([r('2026-03-01', 1, { anger: 0.8 }), r('2026-03-02', 1, { anger: 0.2 })], '2026-03-01', '2026-03-03', 1)
    expect(currentAndPeak(s, 'anger')).toEqual({ current: 20, peak: 80 })
  })
  it('topEmotions returns the 4 highest by mean', () => {
    const rows = [r('2026-03-01', 2, { fear: 2, love: 1.5, neutral: 1, guilt: 0.8, anger: 0.1 })]
    expect(topEmotions(rows, 4)).toEqual(['fear', 'love', 'neutral', 'guilt'])
  })
  it('topEmotions breaks ties by canonical order', () => {
    expect(topEmotions([], 2)).toEqual(['anger', 'sarcasm'])
  })
  it('shiftDay crosses month boundaries', () => {
    expect(shiftDay('2026-03-01', -1)).toBe('2026-02-28')
    expect(shiftDay('2026-02-28', 2)).toBe('2026-03-02')
  })
})
