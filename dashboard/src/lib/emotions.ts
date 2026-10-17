export const LABELS = [
  'anger', 'sarcasm', 'surprise', 'love', 'desire', 'shame', 'fear',
  'sadness', 'disgust', 'confusion', 'neutral', 'guilt', 'happiness',
] as const

export type Label = (typeof LABELS)[number]

export const EMOTION_COLORS: Record<Label, string> = {
  love: '#93c5fd',
  anger: '#a5b4fc',
  happiness: '#4ade80',
  fear: '#d9f99d',
  surprise: '#fbbf24',
  sarcasm: '#f472b6',
  desire: '#fb7185',
  shame: '#c084fc',
  sadness: '#60a5fa',
  disgust: '#2dd4bf',
  confusion: '#f97316',
  neutral: '#a1a1aa',
  guilt: '#fdba74',
}

export function labelName(l: Label): string {
  return l.charAt(0).toUpperCase() + l.slice(1)
}
