import { CartesianGrid, Line, LineChart, XAxis, YAxis } from 'recharts'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { ChartContainer, ChartTooltip, type ChartConfig } from '@/components/ui/chart'
import { Toggle } from '@/components/ui/toggle'
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

export function EmotionChartCard({
  series,
  days,
  ma,
  visible,
  onToggle,
}: {
  series: SeriesPoint[]
  days: number
  ma: number
  visible: Label[]
  onToggle: (l: Label, on: boolean) => void
}) {
  const config = Object.fromEntries(
    LABELS.map((l) => [l, { label: labelName(l), color: EMOTION_COLORS[l] }]),
  ) as ChartConfig
  const data = series.map((p) => ({ day: p.day, count: p.count, ...p.values }))
  const empty = !visible.some((l) => series.some((p) => p.values[l] !== null)) || visible.length === 0
  const last = data.length - 1
  const ticks = data.filter((_, i) => (last - i) % 4 === 0).map((d) => d.day)

  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-3xl font-light">Emotion Prevalence History</CardTitle>
        <CardDescription>
          {`${days}-day ${ma === 1 ? 'daily' : `${ma}-day average`} probability scores extracted from text history`}
        </CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-4">
        <div className="flex flex-wrap gap-2">
          {LABELS.map((l) => (
            <Toggle
              key={l}
              variant="outline"
              pressed={visible.includes(l)}
              onPressedChange={(on) => onToggle(l, on)}
              className="h-9 gap-2 rounded-full px-4 data-[state=off]:opacity-50"
            >
              <span className="size-4 rounded-full" style={{ background: EMOTION_COLORS[l] }} />
              {labelName(l)}
            </Toggle>
          ))}
        </div>
        <div className="rounded-xl bg-background/40 p-2 sm:p-4">
          {empty ? (
            <div className="flex h-72 items-center justify-center text-sm text-muted-foreground">
              No scored messages in this view yet.
            </div>
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
                  domain={[0, 100]}
                  ticks={[0, 50, 100]}
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
