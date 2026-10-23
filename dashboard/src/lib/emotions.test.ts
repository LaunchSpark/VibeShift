import { describe, expect, it } from 'vitest'
import { EMOTION_COLORS, LABELS, labelName } from './emotions'

describe('emotions', () => {
  it('has 7 labels in canonical order', () => {
    expect([...LABELS]).toEqual(['joy', 'sadness', 'anger', 'fear', 'disgust', 'surprise', 'neutral'])
  })
  it('gives every label a 7-char hex color', () => {
    for (const l of LABELS) expect(EMOTION_COLORS[l]).toMatch(/^#[0-9a-f]{6}$/i)
  })
  it('has unique colors', () => {
    expect(new Set(Object.values(EMOTION_COLORS)).size).toBe(LABELS.length)
  })
  it('capitalizes label names', () => {
    expect(labelName('surprise')).toBe('Surprise')
  })
})
