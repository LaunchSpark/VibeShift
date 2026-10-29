import { describe, expect, it } from 'vitest'
import { shuffle } from './labeling'

describe('shuffle', () => {
  it('returns a permutation without mutating the input', () => {
    const input = ['a', 'b', 'c', 'd', 'e']
    const out = shuffle(input)
    expect(input).toEqual(['a', 'b', 'c', 'd', 'e'])
    expect([...out].sort()).toEqual(input)
  })
  it('is a Fisher-Yates shuffle driven by the given random source', () => {
    // random() = 0 always swaps position i with 0, walking down from the end.
    expect(shuffle(['a', 'b', 'c', 'd'], () => 0)).toEqual(['b', 'c', 'd', 'a'])
  })
  it('handles empty and single-item lists', () => {
    expect(shuffle([])).toEqual([])
    expect(shuffle(['only'])).toEqual(['only'])
  })
})
