import { useCallback, useEffect, useState } from 'react'
import { History } from 'lucide-react'
import { useShell } from '@/lib/shell'
import { ErrorAlert, InfoAlert } from '@/components/app-alert'
import { ExtendHistoryDialog } from '@/components/extend-history-dialog'
import { Badge } from '@/components/ui/badge'
import { Card, CardAction, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Empty, EmptyDescription, EmptyHeader, EmptyMedia, EmptyTitle } from '@/components/ui/empty'
import { Skeleton } from '@/components/ui/skeleton'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip'
import { selectFetched, type Fetched } from '@/hooks/use-fetched'
import { fetchGuild, fetchJobs, fetchPendingCount, type Job } from '@/lib/data'

const POLL_MS = 5000
const todayUtc = () => new Date().toISOString().slice(0, 10)

type Raw<T> = { key: string; data: T | null; error: string | null } | null

/** Fetches on mount and every `ms` while mounted. Results are key-scoped; `reload` refetches now. */
function usePolled<T>(key: string, fn: () => Promise<T>, ms: number): [Fetched<T>, () => void] {
  const [raw, setRaw] = useState<Raw<T>>(null)
  const [nonce, setNonce] = useState(0)

  useEffect(() => {
    let live = true
    const run = () =>
      fn().then(
        (data) => live && setRaw({ key, data, error: null }),
        (e) =>
          live &&
          // Keep the last good data of this key; show the error alongside it.
          setRaw((prev) => ({ key, data: prev?.key === key ? prev.data : null, error: String(e?.message ?? e) })),
      )
    run()
    const id = setInterval(run, ms)
    return () => {
      live = false
      clearInterval(id)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps -- `key` encodes fn's inputs
  }, [key, ms, nonce])

  return [selectFetched(raw, key), useCallback(() => setNonce((n) => n + 1), [])]
}

/** Remounted per guild so no jobs/counts leak between guilds. */
export default function BackfillPage() {
  const { guild } = useShell()
  return <BackfillView key={guild!.guild_id} gid={guild!.guild_id} initialDepth={guild!.history_depth} />
}

function BackfillView({ gid, initialDepth }: { gid: string; initialDepth: string | null }) {
  const [jobs, reloadJobs] = usePolled(gid, () => fetchJobs(gid), POLL_MS)
  const [pending, reloadPending] = usePolled(gid, () => fetchPendingCount(gid), POLL_MS)
  const [guild, reloadGuild] = usePolled(gid, () => fetchGuild(gid), POLL_MS)
  const depth = guild.data ? guild.data.history_depth : initialDepth
  const today = todayUtc()

  return (
    <div className="flex flex-col gap-4">
      <Card>
        <CardHeader>
          <CardTitle>
            {depth ? (
              <>
                History goes back to <strong>{depth}</strong>
              </>
            ) : (
              'Live only (no history fetched)'
            )}
          </CardTitle>
          <CardDescription>
            {pending.data === null
              ? 'Counting messages waiting to be scored...'
              : `${pending.data} messages waiting to be scored`}
          </CardDescription>
          <CardAction>
            <ExtendHistoryDialog
              gid={gid}
              depth={depth}
              today={today}
              onCreated={() => {
                reloadJobs()
                reloadPending()
                reloadGuild()
              }}
            />
          </CardAction>
        </CardHeader>
      </Card>
      {pending.error && <ErrorAlert title="Could not load pending count" message={pending.error} />}
      {guild.error && <ErrorAlert title="Could not refresh history depth" message={guild.error} />}
      {jobs.error && <ErrorAlert title="Could not load jobs" message={jobs.error} />}
      {jobs.data === null ? (
        jobs.error ? null : <Skeleton className="h-48 w-full" />
      ) : jobs.data.length === 0 ? (
        <Empty className="border">
          <EmptyHeader>
            <EmptyMedia variant="icon">
              <History />
            </EmptyMedia>
            <EmptyTitle>No backfill jobs yet</EmptyTitle>
            <EmptyDescription>Extend history to fetch older messages.</EmptyDescription>
          </EmptyHeader>
        </Empty>
      ) : (
        <JobsTable jobs={jobs.data} />
      )}
      {jobs.data && jobs.data.some((j) => j.status === 'pending' || j.status === 'running') && (
        <InfoAlert title="Jobs in progress">Refreshing every 5 seconds.</InfoAlert>
      )}
    </div>
  )
}

const STATUS_VARIANT = {
  pending: 'outline',
  running: 'secondary',
  done: 'default',
  failed: 'destructive',
} as const

function JobsTable({ jobs }: { jobs: Job[] }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle>Backfill jobs</CardTitle>
        <CardDescription>Downtime jobs catch up automatically; extend jobs are started by you.</CardDescription>
      </CardHeader>
      <CardContent>
        <TooltipProvider>
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Origin</TableHead>
                <TableHead>Target</TableHead>
                <TableHead>Status</TableHead>
                <TableHead className="text-right">Scanned</TableHead>
                <TableHead>Created (UTC)</TableHead>
                <TableHead>Error</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {jobs.map((j) => (
                <TableRow key={j.id}>
                  <TableCell>
                    <Badge variant={j.origin === 'downtime' ? 'secondary' : 'outline'}>
                      {j.origin === 'downtime' ? 'Downtime' : 'Extend'}
                    </Badge>
                  </TableCell>
                  <TableCell>{j.target ?? 'catch-up'}</TableCell>
                  <TableCell>
                    <Badge variant={STATUS_VARIANT[j.status]}>{j.status}</Badge>
                  </TableCell>
                  <TableCell className="text-right tabular-nums">{j.scanned.toLocaleString()}</TableCell>
                  <TableCell className="whitespace-nowrap">{j.created.slice(0, 16)}</TableCell>
                  <TableCell className="max-w-48">
                    {j.error ? (
                      <Tooltip>
                        <TooltipTrigger asChild>
                          <span className="block truncate text-destructive">{j.error}</span>
                        </TooltipTrigger>
                        <TooltipContent className="max-w-80 break-words">{j.error}</TooltipContent>
                      </Tooltip>
                    ) : (
                      <span className="text-muted-foreground">-</span>
                    )}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </TooltipProvider>
      </CardContent>
    </Card>
  )
}
