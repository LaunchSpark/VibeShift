import { useCallback, useEffect, useState, type FormEvent } from 'react'
import { ArrowLeft, CheckCheck, DatabaseZap, UserRound } from 'lucide-react'
import { ErrorAlert } from '@/components/app-alert'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card'
import { Empty, EmptyDescription, EmptyHeader, EmptyMedia, EmptyTitle } from '@/components/ui/empty'
import { Input } from '@/components/ui/input'
import { Progress } from '@/components/ui/progress'
import { Skeleton } from '@/components/ui/skeleton'
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'
import { LABELS, labelName, type Label } from '@/lib/emotions'
import {
  DEFAULT_WINDOW_SIZE,
  describeError,
  MAX_WINDOW_SIZE,
  fetchLabel,
  fetchDatasets,
  fetchLabeledPositions,
  fetchQueueSize,
  fetchTarget,
  fetchWindow,
  firstGap,
  labelerStorageKey,
  normalizeLabeler,
  parseWindowSize,
  saveLabel,
  windowSizeStorageKey,
  type Dataset,
  type LabelTarget,
  type WindowMessage,
} from '@/lib/labeling'

function readStorage(key: string): string | null {
  try {
    return window.localStorage.getItem(key)
  } catch {
    return null
  }
}

function writeStorage(key: string, value: string | null) {
  try {
    if (value === null) window.localStorage.removeItem(key)
    else window.localStorage.setItem(key, value)
  } catch {
    // Storage blocked: the choice just won't survive a reload.
  }
}

function initialSize(): number {
  return parseWindowSize(readStorage(windowSizeStorageKey) ?? '') ?? DEFAULT_WINDOW_SIZE
}

const datasetStorageKey = 'vibeshift.labelDataset'

const errorText = (e: unknown) => describeError(e, 'the local data folder')

export default function LabelPage() {
  const [labeler, setLabeler] = useState<string | null>(() => readStorage(labelerStorageKey) || null)

  if (!labeler) {
    return (
      <LabelerGate
        onSubmit={(name) => {
          writeStorage(labelerStorageKey, name)
          setLabeler(name)
        }}
      />
    )
  }
  return (
    <Labeling
      labeler={labeler}
      onChangeLabeler={() => {
        writeStorage(labelerStorageKey, null)
        setLabeler(null)
      }}
    />
  )
}

function LabelerGate({ onSubmit }: { onSubmit: (name: string) => void }) {
  const [draft, setDraft] = useState('')
  const slug = normalizeLabeler(draft)
  const submit = (e: FormEvent) => {
    e.preventDefault()
    if (slug) onSubmit(slug)
  }
  return (
    <Card className="mx-auto max-w-md">
      <CardHeader>
        <CardTitle>Who is labeling?</CardTitle>
        <CardDescription>
          Your labels are saved under this name, and your export file is named after it. Use the same name every time.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <form onSubmit={submit} className="flex gap-2">
          <Input autoFocus value={draft} onChange={(e) => setDraft(e.target.value)} placeholder="Your name" aria-label="Your name" />
          <Button type="submit" disabled={!slug}>
            Start
          </Button>
        </form>
        {draft && slug !== draft && (
          <p className="text-muted-foreground mt-2 text-sm">{slug ? `Saved as “${slug}”` : 'Use letters or numbers'}</p>
        )}
      </CardContent>
    </Card>
  )
}

function Labeling({ labeler, onChangeLabeler }: { labeler: string; onChangeLabeler: () => void }) {
  const [datasets, setDatasets] = useState<Dataset[] | null>(null)
  const [dataset, setDataset] = useState<Dataset | null>(null)
  const [size, setSize] = useState(initialSize)
  const [sizeDraft, setSizeDraft] = useState(() => String(initialSize()))
  const [queueSize, setQueueSize] = useState<number | null>(null)
  const [labeled, setLabeled] = useState<Set<number> | null>(null)
  const [position, setPosition] = useState<number | null>(null)
  // The loaded window, tagged with the position and size it was loaded for, so a
  // stale window is never shown (or labeled) after moving on.
  const [loaded, setLoaded] = useState<{
    position: number
    size: number
    target: LabelTarget
    messages: WindowMessage[]
    current: Label | null
  } | null>(null)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetchDatasets().then(
      (names) => {
        setDatasets(names)
        const saved = readStorage(datasetStorageKey)
        setDataset(saved && names.includes(saved) ? saved : names[0] ?? null)
      },
      (e) => setError(errorText(e)),
    )
  }, [])

  // Queue size and this labeler's progress at this window size; resume at the first gap.
  useEffect(() => {
    if (!dataset) return
    let stale = false
    Promise.all([fetchQueueSize(dataset), fetchLabeledPositions(dataset, labeler, size)]).then(
      ([total, positions]) => {
        if (stale) return
        const set = new Set(positions)
        setQueueSize(total)
        setLabeled(set)
        setPosition(firstGap(positions))
      },
      (e) => !stale && setError(errorText(e)),
    )
    return () => {
      stale = true
    }
  }, [dataset, labeler, size])

  // The window ending on the target at `position`, plus any label already saved for it.
  useEffect(() => {
    if (!dataset || position === null || queueSize === null || position >= queueSize) return
    let stale = false
    fetchTarget(dataset, position)
      .then(async (t) => {
        if (!t) throw new Error(`No message at queue position ${position}; re-run pixi run label-import.`)
        const [win, saved] = await Promise.all([fetchWindow(dataset, t, size), fetchLabel(dataset, labeler, size, t.message_id)])
        if (!stale) setLoaded({ position, size, target: t, messages: win, current: saved })
      })
      .catch((e) => !stale && setError(errorText(e)))
    return () => {
      stale = true
    }
  }, [dataset, position, queueSize, size, labeler])

  const ready = loaded !== null && loaded.position === position && loaded.size === size
  const target = ready ? loaded.target : null
  const messages = ready ? loaded.messages : null
  const current = ready ? loaded.current : null

  const pick = useCallback(
    async (label: Label) => {
      if (saving || !target || !labeled) return
      setSaving(true)
      try {
        if (!dataset) return
        await saveLabel(dataset, labeler, size, target, label)
        const next = new Set(labeled).add(target.queue_position)
        setLabeled(next)
        setPosition(firstGap([...next]))
      } catch (e) {
        setError(errorText(e))
      } finally {
        setSaving(false)
      }
    },
    [saving, target, labeled, dataset, labeler, size],
  )

  const back = useCallback(() => {
    if (saving) return
    setPosition((p) => (p === null || p === 0 ? p : p - 1))
  }, [saving])

  // Keys 1-7 pick a label, Backspace goes back. Ignored while typing in a field.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.repeat || e.metaKey || e.ctrlKey || e.altKey) return
      const el = e.target as HTMLElement | null
      if (el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.isContentEditable)) return
      if (e.key === 'Backspace') {
        e.preventDefault()
        back()
        return
      }
      const n = Number(e.key)
      if (Number.isInteger(n) && n >= 1 && n <= LABELS.length) pick(LABELS[n - 1])
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [pick, back])

  // Applied on Enter or blur, never per keystroke: otherwise a label key pressed
  // while the box still has focus ("1" then "3") would silently switch to W=13.
  const applySize = () => {
    const n = parseWindowSize(sizeDraft)
    setSizeDraft(String(n ?? size))
    if (n === null || n === size) return
    writeStorage(windowSizeStorageKey, String(n))
    setError(null)
    setLabeled(null)
    setPosition(null)
    setSize(n)
  }

  const header = (
    <div className="flex flex-wrap items-center justify-between gap-3">
      <div className="flex items-center gap-2 text-sm">
        <UserRound className="text-muted-foreground size-4" />
        <span className="font-medium">{labeler}</span>
        <Button variant="link" size="sm" className="h-auto p-0" onClick={onChangeLabeler}>
          Change
        </Button>
      </div>
      <label className="flex items-center gap-2 text-sm">
        <span className="text-muted-foreground">Dataset</span>
        <select
          className="border-input bg-background h-9 max-w-64 rounded-md border px-2 text-sm"
          value={dataset ?? ''}
          onChange={(e) => {
            const next = e.target.value
            writeStorage(datasetStorageKey, next)
            setDataset(next || null)
            setLabeled(null)
            setPosition(null)
            setLoaded(null)
            setError(null)
          }}
          disabled={!datasets?.length}
          aria-label="Dataset"
        >
          {!datasets?.length && <option value="">No CSV datasets found</option>}
          {datasets?.map((name) => <option key={name} value={name}>{name}</option>)}
        </select>
      </label>
      <label className="flex items-center gap-2 text-sm">
        <span className="text-muted-foreground">Window size</span>
        <Input
          type="number"
          min={1}
          max={MAX_WINDOW_SIZE}
          value={sizeDraft}
          onChange={(e) => setSizeDraft(e.target.value)}
          onBlur={applySize}
          onKeyDown={(e) => {
            if (e.key === 'Enter') e.currentTarget.blur()
          }}
          className="w-20"
        />
      </label>
    </div>
  )

  if (error) {
    return (
      <div className="flex flex-col gap-4">
        {header}
        <ErrorAlert title="Labeling failed" message={error} />
      </div>
    )
  }

  if (datasets?.length === 0) {
    return (
      <div className="flex flex-col gap-4">
        {header}
        <Empty className="border">
          <EmptyHeader>
            <EmptyMedia variant="icon"><DatabaseZap /></EmptyMedia>
            <EmptyTitle>No datasets found</EmptyTitle>
            <EmptyDescription>Put one or more CSV files in <code>data/</code>, then refresh the page.</EmptyDescription>
          </EmptyHeader>
        </Empty>
      </div>
    )
  }

  if (queueSize === null || labeled === null || position === null) {
    return (
      <div className="flex flex-col gap-4">
        {header}
        <Skeleton className="h-96 w-full" />
      </div>
    )
  }

  if (queueSize === 0) {
    return (
      <div className="flex flex-col gap-4">
        {header}
        <Empty className="border">
          <EmptyHeader>
            <EmptyMedia variant="icon">
              <DatabaseZap />
            </EmptyMedia>
            <EmptyTitle>Nothing imported yet</EmptyTitle>
            <EmptyDescription>
              Put your CSV at <code>data/discord_messages.csv</code> and refresh the page.
            </EmptyDescription>
          </EmptyHeader>
        </Empty>
      </div>
    )
  }

  if (position >= queueSize) {
    return (
      <div className="flex flex-col gap-4">
        {header}
        <Empty className="border">
          <EmptyHeader>
            <EmptyMedia variant="icon">
              <CheckCheck />
            </EmptyMedia>
            <EmptyTitle>All caught up</EmptyTitle>
            <EmptyDescription>
              You labeled all {labeled.size} windows at window size {size}.
            </EmptyDescription>
          </EmptyHeader>
        </Empty>
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-4">
      {header}
      <Card>
        <CardHeader>
          <CardTitle>Label data</CardTitle>
          <CardDescription>
            Window {position + 1} of {queueSize} · {size}-message window · {labeled.size} labeled
          </CardDescription>
        </CardHeader>
        <CardContent className="flex flex-col gap-6">
          <Progress value={(labeled.size / queueSize) * 100} aria-label="Labeling progress" />
          {messages === null ? (
            <Skeleton className="h-64 w-full" />
          ) : (
            <ol className="flex flex-col gap-3">
              {messages.map((m) =>
                m.isTarget ? (
                  <li key={m.message_id} className="border-primary bg-muted/40 rounded-lg border-2 p-4">
                    <p className="text-muted-foreground mb-1 text-xs font-medium tracking-wide uppercase">
                      Label this message · {m.author}
                    </p>
                    <p className="text-xl font-medium break-words whitespace-pre-wrap md:text-2xl">{m.content}</p>
                  </li>
                ) : (
                  <li key={m.message_id} className="text-muted-foreground px-4 text-sm">
                    <span className="text-foreground font-medium">{m.author}</span>{' '}
                    <span className="break-words whitespace-pre-wrap">{m.content}</span>
                  </li>
                ),
              )}
            </ol>
          )}
        </CardContent>
        <CardFooter className="flex flex-col gap-3">
          <ToggleGroup
            type="single"
            variant="outline"
            size="lg"
            value={current ?? ''}
            onValueChange={(v) => v && pick(v as Label)}
            disabled={saving || messages === null}
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
          <Button variant="ghost" size="sm" onClick={back} disabled={saving || position === 0} className="self-start">
            <ArrowLeft />
            Back
          </Button>
        </CardFooter>
      </Card>
    </div>
  )
}
