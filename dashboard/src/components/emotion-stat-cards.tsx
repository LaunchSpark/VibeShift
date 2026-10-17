import { Card, CardContent } from '@/components/ui/card'
import { EMOTION_COLORS, labelName, type Label } from '@/lib/emotions'
import { currentAndPeak, type SeriesPoint } from '@/lib/series'

const pct = (v: number | null) => (v === null ? '—' : `${Math.round(v)}%`)

export function EmotionStatCards({ series, visible }: { series: SeriesPoint[]; visible: Label[] }) {
  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
      {visible.map((l) => {
        const { current, peak } = currentAndPeak(series, l)
        return (
          <Card key={l} className="gap-1 py-4">
            <CardContent className="flex flex-col gap-1 px-5">
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <span className="size-2 rounded-full" style={{ background: EMOTION_COLORS[l] }} />
                {labelName(l)} (Current)
              </div>
              <div className="font-mono text-3xl font-bold">{pct(current)}</div>
              <div className="font-mono text-sm italic text-muted-foreground">Peak: {pct(peak)}</div>
            </CardContent>
          </Card>
        )
      })}
    </div>
  )
}
