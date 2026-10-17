import { describe, expect, it } from 'vitest'
import { canExtend } from './backfill'

const today = '2026-10-17'

describe('canExtend', () => {
  it('allows any past day when there is no depth', () => {
    expect(canExtend(null, '2026-10-16', today)).toBe(true)
  })
  it('rejects today or later when there is no depth', () => {
    expect(canExtend(null, today, today)).toBe(false)
    expect(canExtend(null, '2026-10-18', today)).toBe(false)
  })
  it('rejects a target equal to the current depth', () => {
    expect(canExtend('2026-09-01', '2026-09-01', today)).toBe(false)
  })
  it('rejects a target later than the current depth', () => {
    expect(canExtend('2026-09-01', '2026-09-02', today)).toBe(false)
  })
  it('allows a target earlier than the current depth', () => {
    expect(canExtend('2026-09-01', '2026-08-31', today)).toBe(true)
  })
})
