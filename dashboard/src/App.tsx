import { useEffect, useState } from 'react'
import { BrowserRouter, Outlet, Route, Routes } from 'react-router'
import { AppSidebar } from '@/components/app-sidebar'
import { ErrorAlert } from '@/components/app-alert'
import { Empty, EmptyDescription, EmptyHeader, EmptyTitle } from '@/components/ui/empty'
import { Skeleton } from '@/components/ui/skeleton'
import { SidebarInset, SidebarProvider, SidebarTrigger } from '@/components/ui/sidebar'
import { useFilters } from '@/lib/filters'
import { fetchGuilds, type Guild } from '@/lib/data'
import type { ShellContext } from '@/lib/shell'
import EmotionsPage from '@/pages/emotions'
import BackfillPage from '@/pages/backfill'
import LabelPage from '@/pages/label'

function Layout() {
  const [filters] = useFilters()
  const [guilds, setGuilds] = useState<Guild[] | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    fetchGuilds().then(setGuilds, (e) => setError(String(e?.message ?? e)))
  }, [])

  const guild = guilds?.find((g) => g.guild_id === filters.guild) ?? guilds?.[0] ?? null

  return (
    <SidebarProvider>
      <AppSidebar guilds={guilds ?? []} guild={guild} />
      <SidebarInset className="min-w-0">
        <header className="flex h-12 items-center gap-2 px-4 md:hidden">
          <SidebarTrigger />
          <span className="text-sm font-medium">{guild?.name ?? 'VibeShift'}</span>
        </header>
        <div className="mx-auto w-full max-w-6xl min-w-0 p-4 md:p-8">
          {error ? (
            <ErrorAlert title="Could not load guilds" message={error} />
          ) : guilds === null ? (
            <div className="flex flex-col gap-4">
              <Skeleton className="h-24 w-full" />
              <Skeleton className="h-96 w-full" />
            </div>
          ) : guild ? (
            <Outlet context={{ guilds, guild } satisfies ShellContext} />
          ) : (
            <Empty className="border">
              <EmptyHeader>
                <EmptyTitle>No guilds found</EmptyTitle>
                <EmptyDescription>Add the bot to a Discord server to see its stats here.</EmptyDescription>
              </EmptyHeader>
            </Empty>
          )}
        </div>
      </SidebarInset>
    </SidebarProvider>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<EmotionsPage />} />
          <Route path="backfill" element={<BackfillPage />} />
          <Route path="label" element={<LabelPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}
