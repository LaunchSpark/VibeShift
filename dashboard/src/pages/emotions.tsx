import { useEffect, useMemo, useState } from 'react'
import { useShell } from '@/lib/shell'
import { EmotionChartCard } from '@/components/emotion-chart-card'
import { EmotionStatCards } from '@/components/emotion-stat-cards'
import { FiltersPanel } from '@/components/filters-panel'
import {
  fetchChannels,
  fetchMembers,
  fetchPendingCount,
  fetchRollups,
  type Channel,
  type Member,
} from '@/lib/data'
import type { Label } from '@/lib/emotions'
import { resolveRange, useFilters } from '@/lib/filters'
import { buildSeries, shiftDay, topEmotions, type Rollup } from '@/lib/series'

const DAY_MS = 86_400_000
const todayUtc = () => new Date().toISOString().slice(0, 10)

export default function EmotionsPage() {
  const { guild } = useShell()
  const [filters, update] = useFilters()
  const [channels, setChannels] = useState<Channel[]>([])
  const [members, setMembers] = useState<Member[]>([])
  const [rows, setRows] = useState<Rollup[]>([])
  const [pending, setPending] = useState(0)
  const [error, setError] = useState<string | null>(null)

  const gid = guild!.guild_id
  const { from, to } = useMemo(
    () => resolveRange(filters, todayUtc(), guild!.history_depth),
    [filters, guild],
  )
  const days = Math.round((Date.parse(to) - Date.parse(from)) / DAY_MS) + 1
  const peopleKey = filters.people.join(',')
  const channelsKey = filters.channels.join(',')

  useEffect(() => {
    let live = true
    Promise.all([fetchChannels(gid), fetchMembers(gid), fetchPendingCount(gid)]).then(
      ([c, m, p]) => live && (setChannels(c), setMembers(m), setPending(p)),
      (e) => live && setError(String(e?.message ?? e)),
    )
    return () => {
      live = false
    }
  }, [gid])

  useEffect(() => {
    let live = true
    fetchRollups({
      guild: gid,
      from: shiftDay(from, -(filters.ma - 1)),
      to,
      channels: channelsKey ? channelsKey.split(',') : [],
      people: peopleKey ? peopleKey.split(',') : [],
    }).then(
      (r) => live && (setRows(r), setError(null)),
      (e) => live && setError(String(e?.message ?? e)),
    )
    return () => {
      live = false
    }
  }, [gid, from, to, filters.ma, peopleKey, channelsKey])

  const series = useMemo(() => buildSeries(rows, from, to, filters.ma), [rows, from, to, filters.ma])
  const visible: Label[] = filters.emotions ?? topEmotions(rows, 4)

  const onToggle = (l: Label, on: boolean) => {
    const next = on ? [...visible, l] : visible.filter((v) => v !== l)
    if (next.length) update({ emotions: next })
  }

  return (
    <div className="flex flex-col gap-4">
      <FiltersPanel
        filters={filters}
        update={update}
        channels={channels}
        members={members}
        from={from}
        to={to}
      />
      {error && <p className="text-sm text-destructive">{error}</p>}
      {pending > 0 && (
        <p className="text-sm text-muted-foreground">{pending} messages waiting to be scored</p>
      )}
      <EmotionChartCard series={series} days={days} ma={filters.ma} visible={visible} onToggle={onToggle} />
      <EmotionStatCards series={series} visible={visible} />
    </div>
  )
}
