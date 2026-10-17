import { describe, expect, it } from 'vitest'
import { parseFilters, resolveRange, toSearch, type Filters } from './filters'

const q = (s: string) => new URLSearchParams(s)

describe('filters', () => {
  it('defaults when empty', () => {
    const f = parseFilters(q(''))
    expect(f).toEqual({
      guild: null, range: '30d', from: null, to: null, ma: 7,
      people: [], channels: [], emotions: null,
    })
    expect(toSearch(f).toString()).toBe('')
  })

  it('round trips', () => {
    const f: Filters = {
      guild: '111', range: 'custom', from: '2026-01-01', to: '2026-02-01', ma: 14,
      people: ['1', '2'], channels: ['3'], emotions: ['love', 'anger'],
    }
    expect(parseFilters(toSearch(f))).toEqual(f)
  })

  it('invalid params fall back to defaults', () => {
    const f = parseFilters(q('?ma=5&range=yesterday&people=&emo=joy'))
    expect(f.ma).toBe(7)
    expect(f.range).toBe('30d')
    expect(f.people).toEqual([])
    expect(f.emotions).toBeNull()
  })

  it('snowflakes stay strings', () => {
    const id = '123456789012345678901'
    const f = parseFilters(q(`people=${id}`))
    expect(f.people).toEqual([id])
    expect(toSearch(f).get('people')).toBe(id)
  })

  it('resolveRange 30d', () => {
    const f = parseFilters(q(''))
    expect(resolveRange(f, '2026-03-31', null)).toEqual({ from: '2026-03-02', to: '2026-03-31' })
  })

  it('resolveRange all and custom', () => {
    const all = { ...parseFilters(q('range=all')) }
    expect(resolveRange(all, '2026-03-31', '2026-01-05')).toEqual({ from: '2026-01-05', to: '2026-03-31' })
    expect(resolveRange(all, '2026-03-31', null).from).toBe('2026-03-31')
    const c = parseFilters(q('range=custom&from=2026-01-01&to=2026-01-10'))
    expect(resolveRange(c, '2026-03-31', null)).toEqual({ from: '2026-01-01', to: '2026-01-10' })
    const bad = parseFilters(q('range=custom&from=2026-01-01'))
    expect(resolveRange(bad, '2026-03-31', null)).toEqual({ from: '2026-03-02', to: '2026-03-31' })
  })

  it('resolveRange clamps future dates to today', () => {
    const c = parseFilters(q('range=custom&from=2026-03-20&to=2026-04-10'))
    expect(resolveRange(c, '2026-03-31', null)).toEqual({ from: '2026-03-20', to: '2026-03-31' })
    const allFuture = parseFilters(q('range=custom&from=2026-05-01&to=2026-05-10'))
    expect(resolveRange(allFuture, '2026-03-31', null)).toEqual({ from: '2026-03-31', to: '2026-03-31' })
  })
})
