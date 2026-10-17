import { useState, type ReactNode } from 'react'
import { format, parseISO } from 'date-fns'
import { Check, ChevronDown, Pencil } from 'lucide-react'
import type { DateRange } from 'react-day-picker'
import { Button } from '@/components/ui/button'
import { Calendar } from '@/components/ui/calendar'
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible'
import { Command, CommandEmpty, CommandGroup, CommandInput, CommandItem, CommandList } from '@/components/ui/command'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
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

export function FiltersPanel({
  filters,
  update,
  channels,
  members,
  from,
  to,
}: {
  filters: Filters
  update: (patch: Partial<Filters>) => void
  channels: Channel[]
  members: Member[]
  from: string
  to: string
}) {
  const [open, setOpen] = useState(true)
  const selected: DateRange | undefined = filters.range === 'custom' ? { from: parseISO(from), to: parseISO(to) } : undefined
  return (
    <Collapsible open={open} onOpenChange={setOpen} className="rounded-xl border bg-card px-4 py-2">
      <CollapsibleTrigger asChild>
        <button type="button" className="flex w-full items-center justify-between py-1 text-sm font-medium">
          Filters
          <ChevronDown className={`size-4 transition-transform ${open ? 'rotate-180' : ''}`} />
        </button>
      </CollapsibleTrigger>
      <CollapsibleContent>
        <Row label="Time" value={timeLabel(filters, from, to)}>
          <div className="flex flex-col gap-2">
            <div className="flex flex-wrap gap-1">
              {PRESETS.map((p) => (
                <Button
                  key={p.key}
                  size="sm"
                  variant={filters.range === p.key ? 'secondary' : 'ghost'}
                  onClick={() => update({ range: p.key })}
                >
                  {p.label}
                </Button>
              ))}
            </div>
            <Calendar
              mode="range"
              selected={selected}
              defaultMonth={parseISO(to)}
              onSelect={(r) => {
                if (r?.from && r?.to) update({ range: 'custom', from: ymd(r.from), to: ymd(r.to) })
              }}
            />
          </div>
        </Row>
        <Row label="Average" value={filters.ma === 1 ? 'Raw' : `${filters.ma} days`}>
          <Select value={String(filters.ma)} onValueChange={(v) => update({ ma: Number(v) as MovingAverage })}>
            <SelectTrigger className="w-40">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {MAS.map((m) => (
                <SelectItem key={m} value={String(m)}>
                  {m === 1 ? 'Raw' : `${m} days`}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </Row>
        <Row label="People" value={filters.people.length ? `${filters.people.length} selected` : 'Everyone'}>
          <MultiSelect
            items={members.map((m) => ({ id: m.author_id, name: m.display_name || m.author_id }))}
            selected={filters.people}
            onChange={(people) => update({ people })}
            placeholder="Search people…"
          />
        </Row>
        <Row label="Channels" value={filters.channels.length ? `${filters.channels.length} selected` : 'All'}>
          <MultiSelect
            items={channels.map((c) => ({ id: c.channel_id, name: `#${c.name}` }))}
            selected={filters.channels}
            onChange={(ch) => update({ channels: ch })}
            placeholder="Search channels…"
          />
        </Row>
      </CollapsibleContent>
    </Collapsible>
  )
}
