import { describe, expect, it } from 'vitest'
import { rollupFilter } from './data'

const base = { guild: '100', from: '2026-01-01', to: '2026-01-31', channels: [], people: [] }

describe('rollupFilter', () => {
  it('builds guild and day bounds only', () => {
    expect(rollupFilter(base)).toBe('guild_id = "100" && day >= "2026-01-01" && day <= "2026-01-31"')
  })
  it('adds OR groups for channels and people', () => {
    const f = rollupFilter({ ...base, channels: ['a', 'b'], people: ['p'] })
    expect(f).toContain('(channel_id = "a" || channel_id = "b")')
    expect(f).toContain('(author_id = "p")')
  })
  it('keeps 19-digit ids', () => {
    expect(rollupFilter({ ...base, guild: '1234567890123456789' })).toContain('"1234567890123456789"')
  })
  it('escapes quotes and backslashes', () => {
    expect(rollupFilter({ ...base, channels: ['a"b'] })).toContain('channel_id = "a\\"b"')
  })
})
