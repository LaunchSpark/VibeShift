import { describe, expect, it } from 'vitest'
import { describeError, firstGap, normalizeLabeler, parseWindowSize, windowRange } from './labeling'

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

describe('parseWindowSize', () => {
  it('accepts whole numbers from 1 to the max', () => {
    expect(parseWindowSize('1')).toBe(1)
    expect(parseWindowSize(' 25 ')).toBe(25)
  })
  it('rejects anything else', () => {
    expect(parseWindowSize('0')).toBeNull()
    expect(parseWindowSize('26')).toBeNull()
    expect(parseWindowSize('2.5')).toBeNull()
    expect(parseWindowSize('')).toBeNull()
    expect(parseWindowSize('abc')).toBeNull()
  })
})

describe('describeError', () => {
  const base = 'http://127.0.0.1:8090'
  it('explains an unreachable PocketBase', () => {
    expect(describeError({ status: 0, message: 'Something went wrong.' }, base)).toBe(
      "Can't reach PocketBase at http://127.0.0.1:8090. Start it with `pixi run pb`.",
    )
  })
  it('explains missing label collections', () => {
    const e = { status: 404, url: `${base}/api/collections/label_messages/records`, message: 'Missing collection context.' }
    expect(describeError(e, base)).toBe(
      'The labeling collections are missing. Restart `pixi run pb` to apply migrations, then run `pixi run label-import`.',
    )
  })
  it('passes other errors through', () => {
    expect(describeError(new Error('boom'), base)).toBe('boom')
    expect(describeError('plain', base)).toBe('plain')
  })
})
