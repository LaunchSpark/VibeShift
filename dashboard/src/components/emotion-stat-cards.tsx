import { Card, CardContent, CardDescription, CardHeader } from '@/components/ui/card'
import { Skeleton } from '@/components/ui/skeleton'
import { EMOTION_COLORS, LABELS, labelName, type Label } from '@/lib/emotions'
import type { Rollup } from '@/lib/series'

const pct = (v: number | null) => (v === null ? '—' : `${Math.round(v)}%`)
const GRID = 'grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4'

export function EmotionStatCards({ rows }: { rows: Rollup[] }) {
  const totalMessages = rows.reduce((sum, row) => sum + row.count, 0)
  const probabilitySums = Object.fromEntries(LABELS.map((label) => [
    label,
    rows.reduce((sum, row) => sum + (row.prob_sums[label] ?? 0), 0),
  ])) as Record<Label, number>

  return (
    <div className={GRID}>
      {LABELS.map((l) => {
        const prevalence = totalMessages > 0 ? (100 * probabilitySums[l]) / totalMessages : null
        return (
          <Card key={l} size="sm">
            <CardHeader>
              <CardDescription className="flex items-center gap-2">
                <span className="size-2 rounded-full" style={{ background: EMOTION_COLORS[l] }} />
                {labelName(l)}
              </CardDescription>
            </CardHeader>
            <CardContent className="flex flex-col gap-1">
              <div className="font-mono text-3xl font-bold">{pct(prevalence)}</div>
              <div className="text-sm text-muted-foreground">Across {totalMessages.toLocaleString()} scored messages</div>
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
      {Array.from({ length: 7 }, (_, i) => (
        <Skeleton key={i} className="h-28 w-full" />
      ))}
    </div>
  )
}
