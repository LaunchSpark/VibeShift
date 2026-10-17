import { describe, expect, it } from 'vitest'
import { EMOTION_COLORS, LABELS, labelName } from './emotions'

describe('emotions', () => {
  it('has 13 labels in canonical order', () => {
    expect(LABELS.length).toBe(13)
    expect([...LABELS]).toEqual([
      'anger', 'sarcasm', 'surprise', 'love', 'desire', 'shame', 'fear',
      'sadness', 'disgust', 'confusion', 'neutral', 'guilt', 'happiness',
    ])
  })
  it('gives every label a 7-char hex color', () => {
    for (const l of LABELS) expect(EMOTION_COLORS[l]).toMatch(/^#[0-9a-f]{6}$/i)
  })
  it('has unique colors', () => {
    expect(new Set(Object.values(EMOTION_COLORS)).size).toBe(13)
  })
  it('capitalizes label names', () => {
    expect(labelName('happiness')).toBe('Happiness')
  })
})
