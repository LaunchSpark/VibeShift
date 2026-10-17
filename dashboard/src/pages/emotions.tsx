import { useMemo } from 'react'
import { useShell } from '@/lib/shell'
import { ErrorAlert, InfoAlert } from '@/components/app-alert'
import { EmotionChartCard, EmotionChartCardSkeleton } from '@/components/emotion-chart-card'
import { EmotionStatCards, EmotionStatCardsSkeleton } from '@/components/emotion-stat-cards'
import { FiltersPanel } from '@/components/filters-panel'
import { useFetched } from '@/hooks/use-fetched'
import { fetchChannels, fetchMembers, fetchPendingCount, fetchRollups } from '@/lib/data'
import { LABELS, type Label } from '@/lib/emotions'
import { resolveRange, useFilters } from '@/lib/filters'
import { buildSeries, shiftDay, topEmotions } from '@/lib/series'

const DAY_MS = 86_400_000
const todayUtc = () => new Date().toISOString().slice(0, 10)

/** Remounted per guild so no state (rows, channels, ...) leaks between guilds. */
export default function EmotionsPage() {
  const { guild } = useShell()
  return <EmotionsView key={guild!.guild_id} gid={guild!.guild_id} historyDepth={guild!.history_depth} />
}

function EmotionsView({ gid, historyDepth }: { gid: string; historyDepth: string | null }) {
  const [filters, update] = useFilters()
  const { ma, emotions } = filters
  const peopleKey = filters.people.join(',')
  const channelsKey = filters.channels.join(',')

  const today = todayUtc()
  const { from, to } = resolveRange(filters, today, historyDepth)
  const days = Math.round((Date.parse(to) - Date.parse(from)) / DAY_MS) + 1

  const channels = useFetched(gid, () => fetchChannels(gid))
  const members = useFetched(gid, () => fetchMembers(gid))
  const pending = useFetched(gid, () => fetchPendingCount(gid))
  const rollups = useFetched(`${gid}|${from}|${to}|${ma}|${peopleKey}|${channelsKey}`, () =>
    fetchRollups({
      guild: gid,
      from: shiftDay(from, -(ma - 1)),
      to,
      channels: channelsKey ? channelsKey.split(',') : [],
      people: peopleKey ? peopleKey.split(',') : [],
    }),
  )
  const rows = rollups.data

  const series = useMemo(() => (rows ? buildSeries(rows, from, to, ma) : []), [rows, from, to, ma])
  // Rank over the visible window only, not the moving-average lookback padding.
  const top = useMemo(
    () => (rows ? topEmotions(rows.filter((r) => r.day >= from && r.day <= to), 4) : []),
    [rows, from, to],
  )
  const emotionsKey = emotions?.join(',') ?? ''
  const visible: Label[] = useMemo(() => {
    const chosen = new Set<Label>(emotionsKey ? (emotionsKey.split(',') as Label[]) : top)
    return LABELS.filter((l) => chosen.has(l))
  }, [emotionsKey, top])

  return (
    <div className="flex flex-col gap-4">
      <FiltersPanel
        filters={filters}
        update={update}
        channels={channels.data ?? []}
        members={members.data ?? []}
        from={from}
        to={to}
        today={today}
      />
      {rollups.error && <ErrorAlert title="Could not load emotion data" message={rollups.error} />}
      {channels.error && <ErrorAlert title="Could not load channels" message={channels.error} />}
      {members.error && <ErrorAlert title="Could not load people" message={members.error} />}
      {pending.error && <ErrorAlert title="Could not load pending count" message={pending.error} />}
      {(pending.data ?? 0) > 0 && <InfoAlert title={`${pending.data} messages waiting to be scored`} />}
      {rows === null ? (
        rollups.error ? null : (
          <>
            <EmotionChartCardSkeleton days={days} ma={ma} />
            <EmotionStatCardsSkeleton />
          </>
        )
      ) : (
        <>
          <EmotionChartCard
            series={series}
            days={days}
            ma={ma}
            visible={visible}
            onChange={(next) => update({ emotions: next })}
          />
          <EmotionStatCards series={series} visible={visible} />
        </>
      )}
    </div>
  )
}
