import { useEffect, useState } from 'react'
import { BrowserRouter, Outlet, Route, Routes } from 'react-router'
import { AppSidebar } from '@/components/app-sidebar'
import { SidebarInset, SidebarProvider, SidebarTrigger } from '@/components/ui/sidebar'
import { useFilters } from '@/lib/filters'
import { fetchGuilds, type Guild } from '@/lib/data'
import type { ShellContext } from '@/lib/shell'
import EmotionsPage from '@/pages/emotions'
import BackfillPage from '@/pages/backfill'


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
            <p className="text-sm text-destructive">Could not load guilds: {error}</p>
          ) : guilds === null ? (
            <p className="text-sm text-muted-foreground">Loading…</p>
          ) : guild ? (
            <Outlet context={{ guilds, guild } satisfies ShellContext} />
          ) : (
            <p className="text-sm text-muted-foreground">No guilds found.</p>
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
        </Route>
      </Routes>
    </BrowserRouter>
  )
}
