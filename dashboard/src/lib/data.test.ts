import { describe, expect, it } from 'vitest'
import { earliestRollupQuery, guildFilter, mapGuild, rollupFilter } from './data'

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

describe('mapGuild / guildFilter', () => {
  it('maps a record and trims history_depth to a day', () => {
    const g = mapGuild({ id: 'a', guild_id: 'g1', name: 'N', icon_url: '', history_depth: '2026-06-19 00:00:00.000Z' })
    expect(g.history_depth).toBe('2026-06-19')
    expect(mapGuild({ id: 'a', guild_id: 'g1', name: 'N', icon_url: '', history_depth: '' }).history_depth).toBeNull()
  })
  it('escapes the guild id in the filter', () => {
    expect(guildFilter('a"b')).toBe('guild_id = "a\\"b"')
  })
})

describe('earliestRollupQuery', () => {
  it('asks for the single oldest day of the guild', () => {
    expect(earliestRollupQuery('a"b')).toEqual({ filter: 'guild_id = "a\\"b"', sort: 'day', fields: 'day', skipTotal: true })
  })
})
