import type { Label } from './emotions'

/** One message to label: the last message of its window. Mirrors the label_messages collection. */
export type LabelTarget = {
  message_id: string
  conversation_key: string
  message_number: number
  author: string
  content: string
  queue_position: number
  algorithm_version: number
}

/** One message shown in a window, oldest to newest; the last one is the target. */
export type WindowMessage = { message_id: string; author: string; content: string; isTarget: boolean }
type LocalData = { messages: LabelTarget[]; labels: Array<{ id: string; labeler: string; window_size: number; label: Label; queue_position: number }> }
export type Dataset = string

export const labelerStorageKey = 'vibeshift.labeler'
export const windowSizeStorageKey = 'vibeshift.labelWindow'
export const DEFAULT_WINDOW_SIZE = 5
export const MAX_WINDOW_SIZE = 25

/** A typed name as a stable slug ("Lucas S." -> "lucas-s"); used in PocketBase rows and export filenames. */
export function normalizeLabeler(name: string): string {
  return name
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
}

/** Message numbers (1-based, inclusive) of the window of `size` that ends on `messageNumber`. */
export function windowRange(messageNumber: number, size: number): { from: number; to: number } {
  return { from: Math.max(1, messageNumber - size + 1), to: messageNumber }
}

/** A typed window size, or null unless it is a whole number from 1 to MAX_WINDOW_SIZE. */
export function parseWindowSize(raw: string): number | null {
  const t = raw.trim()
  if (!/^\d+$/.test(t)) return null
  const n = Number(t)
  return n >= 1 && n <= MAX_WINDOW_SIZE ? n : null
}

/** A labeler-facing message for a failed PocketBase call. */
export function describeError(e: unknown, baseUrl: string): string {
  const err = e as { status?: number; url?: string; message?: string } | null
  if (err && typeof err === 'object' && err.status === 0) {
    return `Can't reach PocketBase at ${baseUrl}. Start it with \`pixi run pb\`.`
  }
  if (err && typeof err === 'object' && err.status === 404 && /\/collections\/(label_messages|labels)\//.test(err.url ?? '')) {
    return 'The labeling collections are missing. Restart `pixi run pb` to apply migrations, then run `pixi run label-import`.'
  }
  return String((err && typeof err === 'object' && err.message) || e)
}

export async function fetchDatasets(): Promise<Dataset[]> {
  const res = await fetch('/api/local-label-datasets')
  if (!res.ok) throw new Error((await res.text()) || 'Could not list local datasets')
  return res.json()
}

async function localData(dataset: Dataset): Promise<LocalData> {
  const res = await fetch(`/api/local-label-data?dataset=${encodeURIComponent(dataset)}`)
  if (!res.ok) throw new Error((await res.text()) || 'Could not load local label data')
  return res.json()
}

/** Smallest non-negative integer not in `positions`: where a labeler resumes. */
export function firstGap(positions: number[]): number {
  const seen = new Set(positions)
  let i = 0
  while (seen.has(i)) i++
  return i
}

/** Number of messages in the queue; 0 until `pixi run label-import` has run. */
export async function fetchQueueSize(dataset: Dataset): Promise<number> {
  return (await localData(dataset)).messages.length
}

export async function fetchTarget(dataset: Dataset, position: number): Promise<LabelTarget | null> {
  return (await localData(dataset)).messages.find((m) => m.queue_position === Math.trunc(position)) ?? null
}

export async function fetchWindow(dataset: Dataset, target: LabelTarget, size: number): Promise<WindowMessage[]> {
  const { from, to } = windowRange(target.message_number, size)
  const rows = (await localData(dataset)).messages
    .filter((r) => r.conversation_key === target.conversation_key && r.message_number >= from && r.message_number <= to)
    .sort((a, b) => a.message_number - b.message_number)
  return rows.map((r) => ({
    message_id: r.message_id,
    author: r.author,
    content: r.content,
    isTarget: r.message_id === target.message_id,
  }))
}

export async function fetchLabeledPositions(dataset: Dataset, labeler: string, size: number): Promise<number[]> {
  return (await localData(dataset)).labels.filter((r) => r.labeler === labeler && r.window_size === Math.trunc(size)).map((r) => r.queue_position)
}

async function findLabel(dataset: Dataset, labeler: string, size: number, messageId: string) {
  return (await localData(dataset)).labels.find((r) => r.labeler === labeler && r.window_size === Math.trunc(size) && r.id === messageId) ?? null
}

export async function fetchLabel(dataset: Dataset, labeler: string, size: number, messageId: string): Promise<Label | null> {
  return ((await findLabel(dataset, labeler, size, messageId))?.label as Label | undefined) ?? null
}

/** Save (or revise) this labeler's label for the target at this window size. */
export async function saveLabel(dataset: Dataset, labeler: string, size: number, target: LabelTarget, label: Label): Promise<void> {
  const res = await fetch(`/api/local-label-data?dataset=${encodeURIComponent(dataset)}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
    message_id: target.message_id,
    labeler,
    window_size: size,
    label,
    queue_position: target.queue_position,
    algorithm_version: target.algorithm_version,
    }),
  })
  if (!res.ok) throw new Error((await res.text()) || 'Could not save label')
}
