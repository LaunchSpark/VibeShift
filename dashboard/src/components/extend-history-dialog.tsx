import { useState } from 'react'
import { parseISO } from 'date-fns'
import { Spinner } from '@/components/ui/spinner'
import { Button } from '@/components/ui/button'
import { Calendar } from '@/components/ui/calendar'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from '@/components/ui/dialog'
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'
import { ErrorAlert } from '@/components/app-alert'
import { canExtend } from '@/lib/backfill'
import { createExtendJob } from '@/lib/data'
import { shiftDay } from '@/lib/series'

const PRESETS = [30, 60, 90, 180, 365]

export function ExtendHistoryDialog({
  gid,
  depth,
  today,
  onCreated,
}: {
  gid: string
  depth: string | null
  today: string
  onCreated: () => void
}) {
  const [open, setOpen] = useState(false)
  const [target, setTarget] = useState<string | null>(null)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const preset = PRESETS.find((n) => shiftDay(today, -n) === target)
  const valid = target !== null && canExtend(depth, target, today)

  const reset = (next: boolean) => {
    setOpen(next)
    if (next) {
      setTarget(null)
      setError(null)
    }
  }

  async function submit() {
    if (!target || !valid) return
    setSaving(true)
    setError(null)
    try {
      await createExtendJob(gid, target)
      setOpen(false)
      onCreated()
    } catch (e) {
      setError(String((e as Error)?.message ?? e))
    } finally {
      setSaving(false)
    }
  }

  return (
    <Dialog open={open} onOpenChange={reset}>
      <DialogTrigger asChild>
        <Button>Extend history</Button>
      </DialogTrigger>
      <DialogContent className="max-h-[90dvh] overflow-y-auto sm:max-w-md">
        <DialogHeader>
          <DialogTitle>Extend history</DialogTitle>
          <DialogDescription>
            Fetch older messages. History only grows earlier
            {depth ? ` than ${depth}` : ''}.
          </DialogDescription>
        </DialogHeader>
        <div className="flex flex-col items-center gap-4">
          <ToggleGroup
            type="single"
            variant="outline"
            value={preset ? String(preset) : ''}
            onValueChange={(v) => v && setTarget(shiftDay(today, -Number(v)))}
          >
            {PRESETS.map((n) => (
              <ToggleGroupItem key={n} value={String(n)} disabled={!canExtend(depth, shiftDay(today, -n), today)}>
                {n}d
              </ToggleGroupItem>
            ))}
          </ToggleGroup>
          <Calendar
            mode="single"
            selected={target ? parseISO(target) : undefined}
            defaultMonth={parseISO(target ?? depth ?? today)}
            disabled={[{ after: parseISO(today) }, ...(depth ? [{ after: parseISO(shiftDay(depth, -1)) }] : [])]}
            onSelect={(_d, day) => setTarget(day.toLocaleDateString('en-CA'))}
          />
        </div>
        {error && <ErrorAlert title="Could not create job" message={error} />}
        <DialogFooter>
          <Button disabled={!valid || saving} onClick={submit}>
            {saving && <Spinner data-icon="inline-start" />}
            {target ? `Extend to ${target}` : 'Pick a date'}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
