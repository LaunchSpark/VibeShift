import { describe, expect, it } from 'vitest'
import { EMOTION_COLORS, LABELS, labelName } from './emotions'

describe('emotions', () => {
  it('has the 7 labels in schema order', () => {
    expect(LABELS.length).toBe(7)
    expect([...LABELS]).toEqual([
      'joy', 'sadness', 'anger', 'fear', 'disgust', 'surprise', 'neutral',
    ])
  })
  it('gives every label a 7-char hex color', () => {
    for (const l of LABELS) expect(EMOTION_COLORS[l]).toMatch(/^#[0-9a-f]{6}$/i)
  })
  it('has unique colors', () => {
    expect(new Set(Object.values(EMOTION_COLORS)).size).toBe(7)
  })
  it('capitalizes label names', () => {
    expect(labelName('joy')).toBe('Joy')
  })
})
