export const LABELS = ['joy', 'sadness', 'anger', 'fear', 'disgust', 'surprise', 'neutral'] as const

export type Label = (typeof LABELS)[number]

export const EMOTION_COLORS: Record<Label, string> = {
  joy: '#4ade80',
  sadness: '#60a5fa',
  anger: '#a5b4fc',
  fear: '#d9f99d',
  disgust: '#2dd4bf',
  surprise: '#fbbf24',
  neutral: '#a1a1aa',
}

export function labelName(l: Label): string {
  return l.charAt(0).toUpperCase() + l.slice(1)
}
