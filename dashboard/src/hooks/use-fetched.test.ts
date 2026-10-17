import { describe, expect, it } from 'vitest'
import { selectFetched } from './use-fetched'

describe('selectFetched', () => {
  it('is loading before any result', () => {
    expect(selectFetched(null, 'a')).toEqual({ data: null, error: null, loading: true })
  })
  it('returns data only for the matching key', () => {
    const raw = { key: 'a', data: [1], error: null }
    expect(selectFetched(raw, 'a')).toEqual({ data: [1], error: null, loading: false })
    expect(selectFetched(raw, 'b')).toEqual({ data: null, error: null, loading: true })
  })
  it('does not leak a stale error to a new key', () => {
    const raw = { key: 'a', data: null, error: 'boom' }
    expect(selectFetched(raw, 'a').error).toBe('boom')
    expect(selectFetched(raw, 'b')).toEqual({ data: null, error: null, loading: true })
  })
})
