// Deterministic dev seed. Run against `pixi run pb`: npm run seed
import PocketBase from 'pocketbase'
import { LABELS } from '../src/lib/emotions'

const PB_URL = process.env.PB_URL ?? 'http://127.0.0.1:8090'
const GUILD_ID = '100000000000000001'
const DAYS = 120
const pb = new PocketBase(PB_URL)
pb.autoCancellation(false)

// mulberry32 seeded PRNG
function prng(seed: number) {
  let a = seed >>> 0
  return () => {
    a = (a + 0x6d2b79f5) >>> 0
    let t = a
    t = Math.imul(t ^ (t >>> 15), t | 1)
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61)
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296
  }
}
const rand = prng(20261017)

const channels = [
  { channel_id: '300000000000000001', name: 'general' },
  { channel_id: '300000000000000002', name: 'memes' },
  { channel_id: '300000000000000003', name: 'support' },
  { channel_id: '300000000000000004', name: 'late-night' },
].map((c) => ({ ...c, guild_id: GUILD_ID, type: 'text' }))

const members = ['Ava', 'Ben', 'Cleo', 'Dev', 'Esme', 'Finn'].map((n, i) => ({
  guild_id: GUILD_ID,
  author_id: `20000000000000000${i + 1}`,
  display_name: n,
  avatar_url: '',
}))

// Per-emotion wave parameters: base weight, period (days), phase, amplitude.
const emo = LABELS.map((label, i) => {
  const dominant = ['joy', 'neutral', 'surprise'].includes(label)
  return {
    label,
    base: dominant ? 1.2 + rand() * 0.6 : 0.15 + rand() * 0.25,
    period: 18 + rand() * 30,
    phase: rand() * Math.PI * 2 + i,
    amp: dominant ? 0.6 + rand() * 0.5 : 0.15 + rand() * 0.2,
  }
})

function probs(day: number, ci: number, mi: number): number[] {
  const w = emo.map((e, k) => {
    // channel and member shift phase and tilt weights so filtering changes the lines
    const shift = ci * 1.3 + mi * 0.7 + (k % (ci + 2)) * 0.4
    const tilt = 0.6 + 0.9 * (0.5 + 0.5 * Math.sin(k * 1.7 + ci * 2.1 + mi * 1.1))
    const v = e.base * tilt * (1 + e.amp * Math.sin((2 * Math.PI * day) / e.period + e.phase + shift))
    return Math.max(0.01, v)
  })
  const s = w.reduce((a, b) => a + b, 0)
  return w.map((x) => x / s)
}

const dayStr = (daysAgo: number) => {
  const d = new Date()
  d.setUTCHours(0, 0, 0, 0)
  d.setUTCDate(d.getUTCDate() - daysAgo)
  return d.toISOString().slice(0, 10)
}

async function inChunks<T>(items: T[], fn: (t: T) => Promise<unknown>, size = 25) {
  for (let i = 0; i < items.length; i += size) await Promise.all(items.slice(i, i + size).map(fn))
}

async function wipe(col: string) {
  const rows = await pb.collection(col).getFullList({ filter: `guild_id = "${GUILD_ID}"`, fields: 'id' })
  await inChunks(rows, (r) => pb.collection(col).delete(r.id))
}

async function main() {
  for (const col of ['messages', 'daily_rollups', 'backfill_jobs', 'members', 'channels', 'guilds']) await wipe(col)

  const depth = new Date(Date.now() - DAYS * 86400000).toISOString()
  await pb.collection('guilds').create({ guild_id: GUILD_ID, name: 'Seed Server', icon_url: '', history_depth: depth })
  await inChunks(channels, (c) =>
    pb.collection('channels').create({ ...c, newest_fetched_id: '', oldest_fetched_id: '' }))
  await inChunks(members, (m) => pb.collection('members').create(m))

  const rollups: object[] = []
  channels.forEach((ch, ci) =>
    members.forEach((m, mi) => {
      for (let ago = DAYS - 1; ago >= 0; ago--) {
        const day = DAYS - 1 - ago
        // activity varies by member/channel; some days empty
        const activity = 2 + 5 * (0.5 + 0.5 * Math.sin(day / 9 + ci + mi * 0.8))
        if (rand() < 0.22) continue
        const count = Math.max(1, Math.round(activity * (0.4 + rand() * 1.2)))
        const p = probs(day, ci, mi)
        // sum of per-message noisy normalised distributions
        const sums = new Array(LABELS.length).fill(0)
        for (let n = 0; n < count; n++) {
          const noisy = p.map((x) => x * (0.7 + rand() * 0.6))
          const s = noisy.reduce((a, b) => a + b, 0)
          noisy.forEach((x, k) => (sums[k] += x / s))
        }
        const prob_sums: Record<string, number> = {}
        LABELS.forEach((l, k) => (prob_sums[l] = Number(sums[k].toFixed(6))))
        rollups.push({ guild_id: GUILD_ID, channel_id: ch.channel_id, author_id: m.author_id, day: dayStr(ago), count, prob_sums })
      }
    }))
  await inChunks(rollups, (r) => pb.collection('daily_rollups').create(r), 40)

  const iso = (ms: number) => new Date(ms).toISOString()
  await pb.collection('backfill_jobs').create({ guild_id: GUILD_ID, origin: 'extend', target: iso(Date.now() - DAYS * 86400000), status: 'done', scanned: 18432, error: '' })
  await pb.collection('backfill_jobs').create({ guild_id: GUILD_ID, origin: 'extend', target: iso(Date.now() - (DAYS + 60) * 86400000), status: 'running', scanned: 5210, error: '' })
  await pb.collection('backfill_jobs').create({ guild_id: GUILD_ID, origin: 'downtime', target: iso(Date.now() - 3 * 86400000), status: 'failed', scanned: 310, error: 'Missing Access: channel late-night' })

  // 25 pending messages with real-looking 18-digit snowflakes (> 2^53)
  const msgs = Array.from({ length: 25 }, (_, i) => {
    const id = (400000000000000001n + BigInt(i) * 1000003n).toString()
    const ch = channels[i % channels.length]
    const m = members[i % members.length]
    const ts = Date.now() - (25 - i) * 3600000
    return {
      message_id: id, guild_id: GUILD_ID, channel_id: ch.channel_id, author_id: m.author_id,
      created_at: iso(ts), reply_to_id: '', content: `Seed message ${i + 1}`,
      day: iso(ts).slice(0, 10), priority_key: id.padStart(20, '0'),
      score_status: 'pending', emotion: '', confidence: 0, model_id: '', rollup_applied: false,
    }
  })
  await inChunks(msgs, (m) => pb.collection('messages').create(m))

  console.log(`seeded: ${channels.length} channels, ${members.length} members, ${rollups.length} rollups, 3 jobs, ${msgs.length} messages`)
}

main().catch((e) => { console.error(e?.response ?? e); process.exit(1) })
