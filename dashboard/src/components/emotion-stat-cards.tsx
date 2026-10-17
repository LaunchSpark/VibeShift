import { Card, CardContent, CardDescription, CardHeader } from '@/components/ui/card'
import { Skeleton } from '@/components/ui/skeleton'
import { EMOTION_COLORS, labelName, type Label } from '@/lib/emotions'
import { currentAndPeak, type SeriesPoint } from '@/lib/series'

const pct = (v: number | null) => (v === null ? '—' : `${Math.round(v)}%`)
const GRID = 'grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4'

export function EmotionStatCards({ series, visible }: { series: SeriesPoint[]; visible: Label[] }) {
  return (
    <div className={GRID}>
      {visible.map((l) => {
        const { current, peak } = currentAndPeak(series, l)
        return (
          <Card key={l} size="sm">
            <CardHeader>
              <CardDescription className="flex items-center gap-2">
                <span className="size-2 rounded-full" style={{ background: EMOTION_COLORS[l] }} />
                {labelName(l)} (Current)
              </CardDescription>
            </CardHeader>
            <CardContent className="flex flex-col gap-1 font-mono">
              <div className="text-3xl font-bold">{pct(current)}</div>
              <div className="text-sm text-muted-foreground italic">Peak: {pct(peak)}</div>
            </CardContent>
          </Card>
        )
      })}
    </div>
  )
}

export function EmotionStatCardsSkeleton() {
  return (
    <div className={GRID}>
      {Array.from({ length: 4 }, (_, i) => (
        <Skeleton key={i} className="h-28 w-full" />
      ))}
    </div>
  )
}
