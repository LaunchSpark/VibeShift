import { describe, expect, it } from 'vitest'
import { firstGap, normalizeLabeler, windowRange } from './labeling'

describe('normalizeLabeler', () => {
  it('turns a typed name into one stable slug', () => {
    expect(normalizeLabeler('  Lucas S. ')).toBe('lucas-s')
    expect(normalizeLabeler('Ana_María')).toBe('ana-mar-a')
    expect(normalizeLabeler('lucas')).toBe('lucas')
  })
  it('returns an empty string when nothing usable remains', () => {
    expect(normalizeLabeler('!!!')).toBe('')
    expect(normalizeLabeler('   ')).toBe('')
  })
})

describe('windowRange', () => {
  it('clips at the start of the conversation', () => {
    expect(windowRange(2, 5)).toEqual({ from: 1, to: 2 })
  })
  it('ends on the target and spans the window size', () => {
    expect(windowRange(10, 3)).toEqual({ from: 8, to: 10 })
    expect(windowRange(4, 1)).toEqual({ from: 4, to: 4 })
  })
})

describe('firstGap', () => {
  it('finds the first unlabeled queue position', () => {
    expect(firstGap([])).toBe(0)
    expect(firstGap([0, 1, 2])).toBe(3)
    expect(firstGap([2, 0, 1, 4])).toBe(3)
  })
})
