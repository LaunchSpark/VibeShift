import { pb } from './pb'
import type { Rollup } from './series'

export type Guild = { id: string; guild_id: string; name: string; icon_url: string; history_depth: string | null }
export type Channel = { id: string; channel_id: string; guild_id: string; name: string; type: string }
export type Member = { id: string; guild_id: string; author_id: string; display_name: string; avatar_url: string }
export type Job = {
  id: string
  guild_id: string
  origin: 'downtime' | 'extend'
  target: string | null
  status: 'pending' | 'running' | 'done' | 'failed'
  scanned: number
  error: string
  created: string
  updated: string
}

export type RollupParams = { guild: string; from: string; to: string; channels: string[]; people: string[] }

const q = (v: string) => `"${v.replace(/\\/g, '\\\\').replace(/"/g, '\\"')}"`
const orGroup = (field: string, vals: string[]) => `(${vals.map((v) => `${field} = ${q(v)}`).join(' || ')})`
const toDay = (s: unknown): string | null => (typeof s === 'string' && s ? s.slice(0, 10) : null)

export function rollupFilter(p: RollupParams): string {
  const parts = [`guild_id = ${q(p.guild)}`, `day >= ${q(p.from)}`, `day <= ${q(p.to)}`]
  if (p.channels.length) parts.push(orGroup('channel_id', p.channels))
  if (p.people.length) parts.push(orGroup('author_id', p.people))
  return parts.join(' && ')
}

export async function fetchGuilds(): Promise<Guild[]> {
  const rows = await pb.collection('guilds').getFullList({ sort: 'name' })
  return rows.map((r) => ({
    id: r.id,
    guild_id: r.guild_id,
    name: r.name,
    icon_url: r.icon_url,
    history_depth: toDay(r.history_depth),
  }))
}

export async function fetchChannels(guild: string): Promise<Channel[]> {
  const rows = await pb.collection('channels').getFullList({ filter: `guild_id = ${q(guild)}`, sort: 'name' })
  return rows.map((r) => ({ id: r.id, channel_id: r.channel_id, guild_id: r.guild_id, name: r.name, type: r.type }))
}

export async function fetchMembers(guild: string): Promise<Member[]> {
  const rows = await pb.collection('members').getFullList({ filter: `guild_id = ${q(guild)}`, sort: 'display_name' })
  return rows.map((r) => ({
    id: r.id,
    guild_id: r.guild_id,
    author_id: r.author_id,
    display_name: r.display_name,
    avatar_url: r.avatar_url,
  }))
}

export async function fetchRollups(p: RollupParams): Promise<Rollup[]> {
  const rows = await pb
    .collection('daily_rollups')
    .getFullList({ filter: rollupFilter(p), fields: 'day,count,prob_sums', batch: 1000 })
  return rows.map((r) => ({ day: r.day, count: r.count, prob_sums: r.prob_sums ?? {} }))
}

export async function fetchPendingCount(guild: string): Promise<number> {
  const res = await pb
    .collection('messages')
    .getList(1, 1, { filter: `guild_id = ${q(guild)} && score_status = "pending"`, fields: 'id' })
  return res.totalItems
}

export async function fetchJobs(guild: string): Promise<Job[]> {
  const rows = await pb.collection('backfill_jobs').getFullList({ filter: `guild_id = ${q(guild)}`, sort: '-created' })
  return rows.map((r) => ({
    id: r.id,
    guild_id: r.guild_id,
    origin: r.origin,
    target: toDay(r.target),
    status: r.status,
    scanned: r.scanned ?? 0,
    error: r.error ?? '',
    created: r.created,
    updated: r.updated,
  }))
}

export async function createExtendJob(guild: string, target: string): Promise<void> {
  await pb.collection('backfill_jobs').create({ guild_id: guild, origin: 'extend', target, status: 'pending', scanned: 0 })
}
