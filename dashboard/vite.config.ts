import fs from 'node:fs'
import fsp from 'node:fs/promises'
import path from 'node:path'
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vitest/config'
import type { Plugin } from 'vite'

const ROOT = path.resolve(__dirname, '..')
const DATA_DIR = path.join(ROOT, 'data')
const LABEL_DIR = path.join(DATA_DIR, 'annotations', 'labels')
const LABELS = new Set(['joy', 'sadness', 'anger', 'fear', 'disgust', 'surprise', 'neutral'])

type LocalMessage = {
  message_id: string
  conversation_key: string
  message_number: number
  author: string
  content: string
  queue_position: number
  algorithm_version: number
}

type LocalLabel = { id: string; labeler: string; window_size: number; label: string; queue_position: number; algorithm_version: number }
type LabelPayload = LocalLabel & { message_id: string }

function parseCsv(text: string): string[][] {
  const rows: string[][] = []
  let row: string[] = []
  let field = ''
  let quoted = false
  for (let i = 0; i < text.length; i++) {
    const ch = text[i]
    if (quoted) {
      if (ch === '"' && text[i + 1] === '"') {
        field += '"'
        i++
      } else if (ch === '"') quoted = false
      else field += ch
    } else if (ch === '"' && field.length === 0) quoted = true
    else if (ch === ',') {
      row.push(field)
      field = ''
    } else if (ch === '\n') {
      row.push(field.replace(/\r$/, ''))
      rows.push(row)
      row = []
      field = ''
    } else field += ch
  }
  if (field || row.length) {
    row.push(field.replace(/\r$/, ''))
    rows.push(row)
  }
  return rows
}

function datasetPath(dataset: string): string {
  if (!/^[a-zA-Z0-9][a-zA-Z0-9._-]*\.csv$/.test(dataset)) throw new Error('Invalid dataset name')
  const resolved = path.resolve(DATA_DIR, dataset)
  if (path.dirname(resolved) !== DATA_DIR) throw new Error('Invalid dataset path')
  return resolved
}

async function listDatasets(): Promise<string[]> {
  if (!fs.existsSync(DATA_DIR)) return []
  return (await fsp.readdir(DATA_DIR, { withFileTypes: true }))
    .filter((entry) => entry.isFile() && entry.name.toLowerCase().endsWith('.csv'))
    .map((entry) => entry.name)
    .sort((a, b) => (a === 'discord_messages.csv' ? -1 : b === 'discord_messages.csv' ? 1 : a.localeCompare(b)))
}

async function loadMessagesFrom(dataset: string): Promise<LocalMessage[]> {
  const csvPath = datasetPath(dataset)
  if (!fs.existsSync(csvPath)) return []
  const rows = parseCsv(await fsp.readFile(csvPath, 'utf8'))
  if (!rows.length) return []
  const headers = new Map(rows[0].map((h, i) => [h.trim(), i]))
  const required = ['conversation_key', 'conversation_message_number', 'author', 'message']
  if (required.some((h) => !headers.has(h))) throw new Error(`data/discord_messages.csv is missing a required column (need ${required.join(', ')})`)
  const messages = rows.slice(1).filter((r) => r.length > 1).map((r) => {
    const conversation_key = r[headers.get('conversation_key')!]
    const message_number = Number(r[headers.get('conversation_message_number')!])
    const content = (r[headers.get('message')!] ?? '').replaceAll('\\n', '\n')
    return {
      message_id: `${conversation_key}:${message_number}`,
      conversation_key,
      message_number,
      author: r[headers.get('author')!] ?? '',
      content,
      queue_position: 0,
      algorithm_version: 1,
    }
  })
  // Shuffle deterministically so the queue feels random but resumes consistently
  // after a refresh or restart. Conversation order itself is preserved below
  // when the context window is built.
  const score = (message: LocalMessage): number => {
    let hash = 2166136261
    for (const char of `${dataset}:${message.message_id}`) {
      hash ^= char.charCodeAt(0)
      hash = Math.imul(hash, 16777619)
    }
    return hash >>> 0
  }
  messages.sort((a, b) => score(a) - score(b) || a.message_id.localeCompare(b.message_id))
  return messages.map((message, queue_position) => ({ ...message, queue_position }))
}

async function loadLabels(dataset: string): Promise<LocalLabel[]> {
  if (!fs.existsSync(LABEL_DIR)) return []
  const stem = path.basename(dataset, '.csv').replace(/[^a-zA-Z0-9_-]+/g, '_')
  const files = (await fsp.readdir(LABEL_DIR)).filter((name) => name.startsWith(`${stem}_`) && name.endsWith('.jsonl'))
  const out: LocalLabel[] = []
  for (const file of files) {
    const text = await fsp.readFile(path.join(LABEL_DIR, file), 'utf8')
    for (const line of text.split(/\r?\n/).filter(Boolean)) out.push(JSON.parse(line) as LocalLabel)
  }
  return out
}

function localLabelApi(): Plugin {
  return {
    name: 'vibeshift-local-label-data',
    configureServer(server) {
      server.middlewares.use('/api/local-label-datasets', async (_req, res) => {
        res.setHeader('Content-Type', 'application/json')
        res.end(JSON.stringify(await listDatasets()))
      })
      server.middlewares.use('/api/local-label-data', async (req, res) => {
        try {
          const requestUrl = new URL(req.url ?? '/', 'http://localhost')
          const dataset = requestUrl.searchParams.get('dataset') || 'discord_messages.csv'
          if (req.method === 'GET') {
            res.setHeader('Content-Type', 'application/json')
            res.end(JSON.stringify({ messages: await loadMessagesFrom(dataset), labels: await loadLabels(dataset) }))
            return
          }
          if (req.method !== 'POST') {
            res.statusCode = 405
            res.end('Method not allowed')
            return
          }
          let body = ''
          for await (const chunk of req) body += chunk
          const value = JSON.parse(body) as LabelPayload
          if (!value.labeler || !/^[-a-z0-9]+$/.test(value.labeler) || !LABELS.has(value.label)) throw new Error('Invalid label payload')
          await fsp.mkdir(LABEL_DIR, { recursive: true })
          const stem = path.basename(dataset, '.csv').replace(/[^a-zA-Z0-9_-]+/g, '_')
          const file = path.join(LABEL_DIR, `${stem}_${value.labeler}_w${Math.trunc(value.window_size)}.jsonl`)
          const existing: LocalLabel[] = fs.existsSync(file) ? (await fsp.readFile(file, 'utf8')).split(/\r?\n/).filter(Boolean).map((line) => JSON.parse(line) as LocalLabel) : []
          const next = existing.filter((row: LocalLabel) => row.id !== value.message_id)
          next.push({ id: value.message_id, labeler: value.labeler, window_size: Math.trunc(value.window_size), label: value.label, queue_position: value.queue_position, algorithm_version: value.algorithm_version })
          await fsp.writeFile(file, next.sort((a, b) => a.queue_position - b.queue_position).map((row) => JSON.stringify(row)).join('\n') + '\n')
          res.setHeader('Content-Type', 'application/json')
          res.end(JSON.stringify({ ok: true }))
        } catch (error) {
          res.statusCode = 400
          res.end(error instanceof Error ? error.message : String(error))
        }
      })
    },
  }
}

// https://vite.dev/config/
export default defineConfig({
  plugins: [localLabelApi(), react(), tailwindcss()],
  resolve: { alias: { '@': path.resolve(__dirname, './src') } },
})
