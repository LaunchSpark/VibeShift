import { pb } from './pb'
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

/** Smallest non-negative integer not in `positions`: where a labeler resumes. */
export function firstGap(positions: number[]): number {
  const seen = new Set(positions)
  let i = 0
  while (seen.has(i)) i++
  return i
}

const q = (v: string) => `"${v.replace(/\\/g, '\\\\').replace(/"/g, '\\"')}"`

function toTarget(r: Record<string, any>): LabelTarget {
  return {
    message_id: r.message_id,
    conversation_key: r.conversation_key,
    message_number: r.message_number,
    author: r.author,
    content: r.content,
    queue_position: r.queue_position,
    algorithm_version: r.algorithm_version,
  }
}

/** Number of messages in the queue; 0 until `pixi run label-import` has run. */
export async function fetchQueueSize(): Promise<number> {
  const res = await pb.collection('label_messages').getList(1, 1, { fields: 'id' })
  return res.totalItems
}

export async function fetchTarget(position: number): Promise<LabelTarget | null> {
  const res = await pb
    .collection('label_messages')
    .getList(1, 1, { filter: `queue_position = ${Math.trunc(position)}`, skipTotal: true })
  return res.items[0] ? toTarget(res.items[0]) : null
}

export async function fetchWindow(target: LabelTarget, size: number): Promise<WindowMessage[]> {
  const { from, to } = windowRange(target.message_number, size)
  const rows = await pb.collection('label_messages').getFullList({
    filter: `conversation_key = ${q(target.conversation_key)} && message_number >= ${from} && message_number <= ${to}`,
    sort: 'message_number',
    fields: 'message_id,author,content',
  })
  return rows.map((r) => ({
    message_id: r.message_id,
    author: r.author,
    content: r.content,
    isTarget: r.message_id === target.message_id,
  }))
}

const labelKey = (labeler: string, size: number) => `labeler = ${q(labeler)} && window_size = ${Math.trunc(size)}`

export async function fetchLabeledPositions(labeler: string, size: number): Promise<number[]> {
  const rows = await pb
    .collection('labels')
    .getFullList({ filter: labelKey(labeler, size), fields: 'queue_position', batch: 1000 })
  return rows.map((r) => r.queue_position as number)
}

async function findLabel(labeler: string, size: number, messageId: string) {
  const res = await pb.collection('labels').getList(1, 1, {
    filter: `${labelKey(labeler, size)} && message_id = ${q(messageId)}`,
    skipTotal: true,
  })
  return res.items[0] ?? null
}

export async function fetchLabel(labeler: string, size: number, messageId: string): Promise<Label | null> {
  return ((await findLabel(labeler, size, messageId))?.label as Label | undefined) ?? null
}

/** Save (or revise) this labeler's label for the target at this window size. */
export async function saveLabel(labeler: string, size: number, target: LabelTarget, label: Label): Promise<void> {
  const existing = await findLabel(labeler, size, target.message_id)
  if (existing) {
    await pb.collection('labels').update(existing.id, { label })
    return
  }
  await pb.collection('labels').create({
    message_id: target.message_id,
    labeler,
    window_size: size,
    label,
    queue_position: target.queue_position,
    algorithm_version: target.algorithm_version,
  })
}
