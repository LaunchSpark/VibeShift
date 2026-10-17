import { Fragment, useState, type ReactNode } from 'react'
import { format, parseISO } from 'date-fns'
import { Check, ChevronDown, Pencil } from 'lucide-react'
import type { DateRange } from 'react-day-picker'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { Calendar } from '@/components/ui/calendar'
import { Card, CardAction, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible'
import { Command, CommandEmpty, CommandGroup, CommandInput, CommandItem, CommandList } from '@/components/ui/command'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { Separator } from '@/components/ui/separator'
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'
import type { Channel, Member } from '@/lib/data'
import type { Filters, MovingAverage, RangeKey } from '@/lib/filters'

const PRESETS: { key: RangeKey; label: string }[] = [
  { key: '7d', label: 'Last 7 days' },
  { key: '30d', label: 'Last 30 days' },
  { key: '60d', label: 'Last 60 days' },
  { key: '90d', label: 'Last 90 days' },
  { key: 'all', label: 'All time' },
]
const MAS: MovingAverage[] = [1, 7, 14, 21, 30]

const ymd = (d: Date) => format(d, 'yyyy-MM-dd')

function timeLabel(f: Filters, from: string, to: string): string {
  if (f.range === 'custom') return `${from} to ${to}`
  return PRESETS.find((p) => p.key === f.range)?.label ?? 'Last 30 days'
}

function Row({ label, value, children }: { label: string; value: string; children: ReactNode }) {
  return (
    <div className="flex items-center justify-between gap-2 py-1.5 text-sm">
      <span className="min-w-0 truncate">
        <span className="font-semibold">{label}</span>
        <span className="text-muted-foreground"> — {value}</span>
      </span>
      <Popover>
        <PopoverTrigger asChild>
          <Button variant="ghost" size="icon-sm" aria-label={`Edit ${label}`}>
            <Pencil />
          </Button>
        </PopoverTrigger>
        <PopoverContent align="end" className="w-auto max-w-[calc(100vw-2rem)]">
          {children}
        </PopoverContent>
      </Popover>
    </div>
  )
}

function MultiSelect({
  items,
  selected,
  onChange,
  placeholder,
}: {
  items: { id: string; name: string }[]
  selected: string[]
  onChange: (ids: string[]) => void
  placeholder: string
}) {
  const toggle = (id: string) =>
    onChange(selected.includes(id) ? selected.filter((s) => s !== id) : [...selected, id])
  return (
    <Command className="w-64">
      <CommandInput placeholder={placeholder} />
      <CommandList>
        <CommandEmpty>No matches.</CommandEmpty>
        <CommandGroup>
          {items.map((it) => (
            <CommandItem key={it.id} value={`${it.name} ${it.id}`} onSelect={() => toggle(it.id)}>
              <Check className={selected.includes(it.id) ? 'opacity-100' : 'opacity-0'} />
              {it.name}
            </CommandItem>
          ))}
        </CommandGroup>
      </CommandList>
    </Command>
  )
}

/**
 * Preset toggles plus a range calendar. react-day-picker reports a half-finished
 * range ({ from, to: undefined }) after the first click, so the in-progress pick
 * lives in a local draft and is committed only once both ends are chosen. The
 * draft resets when the popover closes (this component unmounts).
 */
function TimeEditor({ filters, update, from, to, today }: { filters: Filters; update: (p: Partial<Filters>) => void; from: string; to: string; today: string }) {
  const [draft, setDraft] = useState<DateRange | undefined>(undefined)
  const committed: DateRange | undefined =
    filters.range === 'custom' ? { from: parseISO(from), to: parseISO(to) } : undefined

  const pick = (day: Date) => {
    if (!draft?.from || draft.to) {
      setDraft({ from: day, to: undefined })
      return
    }
    const range = day < draft.from ? { from: day, to: draft.from } : { from: draft.from, to: day }
    setDraft(range)
    update({ range: 'custom', from: ymd(range.from), to: ymd(range.to) })
  }

  return (
    <div className="flex flex-col gap-2">
      <ToggleGroup
        type="single"
        variant="outline"
        size="sm"
        spacing={1}
        className="flex-wrap"
        value={filters.range === 'custom' ? '' : filters.range}
        onValueChange={(v) => {
          if (v) {
            setDraft(undefined)
            update({ range: v as RangeKey })
          }
        }}
      >
        {PRESETS.map((p) => (
          <ToggleGroupItem key={p.key} value={p.key}>
            {p.label}
          </ToggleGroupItem>
        ))}
      </ToggleGroup>
      <Calendar
        mode="range"
        selected={draft ?? committed}
        defaultMonth={parseISO(to)}
        disabled={{ after: parseISO(today) }}
        onSelect={(_range, day) => pick(day)}
      />
    </div>
  )
}

export function FiltersPanel({
  filters,
  update,
  channels,
  members,
  from,
  to,
  today,
}: {
  filters: Filters
  update: (patch: Partial<Filters>) => void
  channels: Channel[]
  members: Member[]
  from: string
  to: string
  today: string
}) {
  const [open, setOpen] = useState(true)
  const rows: { label: string; value: string; body: ReactNode }[] = [
    { label: 'Time', value: timeLabel(filters, from, to), body: <TimeEditor filters={filters} update={update} from={from} to={to} today={today} /> },
    {
      label: 'Average',
      value: filters.ma === 1 ? 'Raw' : `${filters.ma} days`,
      body: (
        <ToggleGroup
          type="single"
          variant="outline"
          size="sm"
          spacing={1}
          value={String(filters.ma)}
          onValueChange={(v) => v && update({ ma: Number(v) as MovingAverage })}
        >
          {MAS.map((m) => (
            <ToggleGroupItem key={m} value={String(m)} aria-label={m === 1 ? 'Raw' : `${m} days`}>
              {m === 1 ? 'Raw' : `${m}d`}
            </ToggleGroupItem>
          ))}
        </ToggleGroup>
      ),
    },
    {
      label: 'People',
      value: filters.people.length ? `${filters.people.length} selected` : 'Everyone',
      body: (
        <MultiSelect
          items={members.map((m) => ({ id: m.author_id, name: m.display_name || m.author_id }))}
          selected={filters.people}
          onChange={(people) => update({ people })}
          placeholder="Search people…"
        />
      ),
    },
    {
      label: 'Channels',
      value: filters.channels.length ? `${filters.channels.length} selected` : 'All',
      body: (
        <MultiSelect
          items={channels.map((c) => ({ id: c.channel_id, name: `#${c.name}` }))}
          selected={filters.channels}
          onChange={(ch) => update({ channels: ch })}
          placeholder="Search channels…"
        />
      ),
    },
  ]

  return (
    <Collapsible open={open} onOpenChange={setOpen} asChild>
      <Card size="sm">
        <CardHeader>
          <CardTitle>Filters</CardTitle>
          <CardAction>
            <CollapsibleTrigger asChild>
              <Button variant="ghost" size="icon-sm" aria-label={open ? 'Collapse filters' : 'Expand filters'}>
                <ChevronDown className={cn('transition-transform', open && 'rotate-180')} />
              </Button>
            </CollapsibleTrigger>
          </CardAction>
        </CardHeader>
        <CollapsibleContent>
          <CardContent>
            {rows.map((r, i) => (
              <Fragment key={r.label}>
                {i > 0 && <Separator />}
                <Row label={r.label} value={r.value}>
                  {r.body}
                </Row>
              </Fragment>
            ))}
          </CardContent>
        </CollapsibleContent>
      </Card>
    </Collapsible>
  )
}
