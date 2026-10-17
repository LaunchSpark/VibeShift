import { CartesianGrid, Line, LineChart, XAxis, YAxis } from 'recharts'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { ChartContainer, ChartTooltip, type ChartConfig } from '@/components/ui/chart'
import { Empty, EmptyDescription, EmptyHeader, EmptyTitle } from '@/components/ui/empty'
import { Skeleton } from '@/components/ui/skeleton'
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'
import { EMOTION_COLORS, LABELS, labelName, type Label } from '@/lib/emotions'
import type { SeriesPoint } from '@/lib/series'

const MONO = 'var(--font-mono)'

function dotFor(color: string) {
  // Hollow circle in the line color; skip null points.
  return function Dot(props: { cx?: number; cy?: number; value?: unknown; index?: number }) {
    const { cx, cy, value, index } = props
    if (value === null || value === undefined || cx === undefined || cy === undefined) return <g key={index} />
    return <circle key={index} cx={cx} cy={cy} r={3} fill="var(--card)" stroke={color} strokeWidth={2} />
  }
}

/** Y-axis ceiling: (max visible value + 5) rounded up to a multiple of 10, capped at 100. */
function yMax(series: SeriesPoint[], visible: Label[]): number {
  let max = 0
  for (const p of series) for (const l of visible) max = Math.max(max, p.values[l] ?? 0)
  return Math.min(100, Math.ceil((max + 5) / 10) * 10)
}

function subtitle(days: number, ma: number): string {
  const span = `Last ${days} ${days === 1 ? 'day' : 'days'}`
  return ma === 1 ? `${span} · daily probability scores` : `${span} · ${ma}-day moving average of probability scores`
}

function Header({ days, ma }: { days: number; ma: number }) {
  return (
    <CardHeader>
      <CardTitle className="text-3xl font-light">Emotion Prevalence History</CardTitle>
      <CardDescription>{subtitle(days, ma)}</CardDescription>
    </CardHeader>
  )
}

export function EmotionChartCardSkeleton({ days, ma }: { days: number; ma: number }) {
  return (
    <Card>
      <Header days={days} ma={ma} />
      <CardContent className="flex flex-col gap-4">
        <div className="flex flex-wrap gap-2">
          {Array.from({ length: 4 }, (_, i) => (
            <Skeleton key={i} className="h-9 w-28 rounded-full" />
          ))}
        </div>
        <Skeleton className="h-72 w-full" />
      </CardContent>
    </Card>
  )
}

export function EmotionChartCard({
  series,
  days,
  ma,
  visible,
  onChange,
}: {
  series: SeriesPoint[]
  days: number
  ma: number
  /** Canonical LABELS order. */
  visible: Label[]
  onChange: (labels: Label[]) => void
}) {
  const config = Object.fromEntries(
    LABELS.map((l) => [l, { label: labelName(l), color: EMOTION_COLORS[l] }]),
  ) as ChartConfig
  const data = series.map((p) => ({ day: p.day, count: p.count, ...p.values }))
  const empty = !visible.some((l) => series.some((p) => p.values[l] !== null))
  const last = data.length - 1
  const ticks = data.filter((_, i) => (last - i) % 4 === 0).map((d) => d.day)
  const top = yMax(series, visible)

  return (
    <Card>
      <Header days={days} ma={ma} />
      <CardContent className="flex flex-col gap-4">
        <ToggleGroup
          type="multiple"
          variant="outline"
          spacing={2}
          className="flex-wrap"
          aria-label="Visible emotions"
          value={visible}
          onValueChange={(v) => {
            // Keep at least one emotion; always report in canonical order.
            if (v.length) onChange(LABELS.filter((l) => v.includes(l)))
          }}
        >
          {LABELS.map((l) => {
            const lastOne = visible.length === 1 && visible[0] === l
            return (
              <ToggleGroupItem
                key={l}
                value={l}
                disabled={lastOne}
                aria-label={lastOne ? `${labelName(l)} (at least one emotion must stay visible)` : labelName(l)}
                className="h-9 rounded-full px-4 data-[state=off]:opacity-50"
              >
                <span className="size-4 rounded-full" style={{ background: EMOTION_COLORS[l] }} />
                {labelName(l)}
              </ToggleGroupItem>
            )
          })}
        </ToggleGroup>
        <div className="rounded-xl bg-background/40 p-2 sm:p-4">
          {empty ? (
            <Empty className="h-72">
              <EmptyHeader>
                <EmptyTitle>No scored messages</EmptyTitle>
                <EmptyDescription>Nothing in this view has been scored yet. Try a wider range or fewer filters.</EmptyDescription>
              </EmptyHeader>
            </Empty>
          ) : (
            <ChartContainer config={config} className="h-72 w-full">
              <LineChart data={data} margin={{ left: 0, right: 28, top: 8 }}>
                <CartesianGrid vertical={false} stroke="var(--border)" />
                <XAxis
                  dataKey="day"
                  ticks={ticks}
                  tickLine={false}
                  axisLine={false}
                  tickMargin={8}
                  tickFormatter={(d: string) => d.slice(5)}
                  tick={{ fontFamily: MONO, fontSize: 12 }}
                />
                <YAxis
                  domain={[0, top]}
                  ticks={[0, top / 2, top]}
                  tickFormatter={(v: number) => `${v}%`}
                  tickLine={false}
                  axisLine={false}
                  width={44}
                  tick={{ fontFamily: MONO, fontSize: 12 }}
                />
                <ChartTooltip
                  content={({ active, payload, label }) => {
                    if (!active || !payload?.length) return null
                    const count = payload[0]?.payload?.count ?? 0
                    return (
                      <div className="grid min-w-36 gap-1.5 rounded-lg border bg-popover px-2.5 py-1.5 text-xs shadow-xl">
                        <div className="font-medium">{String(label)}</div>
                        <div className="text-muted-foreground">{count} messages</div>
                        {payload.map((p) => (
                          <div key={String(p.dataKey)} className="flex items-center justify-between gap-4">
                            <span className="flex items-center gap-1.5">
                              <span className="size-2 rounded-full" style={{ background: p.color }} />
                              {labelName(p.dataKey as Label)}
                            </span>
                            <span className="font-mono">{typeof p.value === 'number' ? `${Math.round(p.value)}%` : '—'}</span>
                          </div>
                        ))}
                      </div>
                    )
                  }}
                />
                {visible.map((l) => (
                  <Line
                    key={l}
                    dataKey={l}
                    type="monotone"
                    stroke={EMOTION_COLORS[l]}
                    strokeWidth={2.5}
                    connectNulls={false}
                    dot={dotFor(EMOTION_COLORS[l])}
                    activeDot={{ r: 4 }}
                    isAnimationActive={false}
                  />
                ))}
              </LineChart>
            </ChartContainer>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
