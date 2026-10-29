import { useCallback, useEffect, useState } from 'react'
import { CheckCheck, Tags } from 'lucide-react'
import { ErrorAlert } from '@/components/app-alert'
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card'
import { Empty, EmptyDescription, EmptyHeader, EmptyMedia, EmptyTitle } from '@/components/ui/empty'
import { Progress } from '@/components/ui/progress'
import { Skeleton } from '@/components/ui/skeleton'
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'
import { LABELS, labelName, type Label } from '@/lib/emotions'
import { fetchLabelQueue, shuffle, type LabelItem } from '@/lib/labeling'

export default function LabelPage() {
  const [queue, setQueue] = useState<LabelItem[] | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [index, setIndex] = useState(0)
  const [labels, setLabels] = useState<Record<string, Label>>({})

  useEffect(() => {
    // Shuffle once per session so messages from the same conversation never
    // appear back to back: each one is judged on its own text, not its neighbours.
    fetchLabelQueue().then(
      (items) => setQueue(shuffle(items)),
      (e) => setError(String(e?.message ?? e)),
    )
  }, [])

  const current = queue?.[index] ?? null

  const pick = useCallback(
    (label: Label) => {
      if (!current) return
      setLabels((prev) => ({ ...prev, [current.id]: label }))
      setIndex((i) => i + 1)
    },
    [current],
  )

  // Number keys 1-7 pick the matching label.
  useEffect(() => {
    if (!current) return
    const onKey = (e: KeyboardEvent) => {
      if (e.repeat || e.metaKey || e.ctrlKey || e.altKey) return
      const n = Number(e.key)
      if (n >= 1 && n <= LABELS.length) pick(LABELS[n - 1])
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [current, pick])

  if (error) return <ErrorAlert title="Could not load messages to label" message={error} />
  if (queue === null) return <Skeleton className="h-96 w-full" />
  if (queue.length === 0 || !current) {
    const done = Object.keys(labels).length
    return (
      <Empty className="border">
        <EmptyHeader>
          <EmptyMedia variant="icon">{done ? <CheckCheck /> : <Tags />}</EmptyMedia>
          <EmptyTitle>{done ? 'All caught up' : 'Nothing to label'}</EmptyTitle>
          <EmptyDescription>
            {done ? `You labeled ${done} message${done === 1 ? '' : 's'}.` : 'There are no messages waiting for a label.'}
          </EmptyDescription>
        </EmptyHeader>
      </Empty>
    )
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Label data</CardTitle>
        <CardDescription>
          Message {index + 1} of {queue.length} · shown in random order, without context
        </CardDescription>
      </CardHeader>
      <CardContent className="flex flex-col gap-6">
        <Progress value={(index / queue.length) * 100} aria-label="Labeling progress" />
        <blockquote className="flex min-h-48 items-center justify-center text-center text-2xl font-medium text-balance break-words md:text-3xl">
          {current.content}
        </blockquote>
      </CardContent>
      <CardFooter>
        <ToggleGroup
          type="single"
          variant="outline"
          size="lg"
          value=""
          onValueChange={(v) => v && pick(v as Label)}
          aria-label="Emotion"
          className="w-full flex-wrap justify-center"
        >
          {LABELS.map((label, i) => (
            <ToggleGroupItem key={label} value={label}>
              <span className="text-muted-foreground tabular-nums">{i + 1}</span>
              {labelName(label)}
            </ToggleGroupItem>
          ))}
        </ToggleGroup>
      </CardFooter>
    </Card>
  )
}
