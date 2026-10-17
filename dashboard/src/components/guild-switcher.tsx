import { ChevronsUpDown } from 'lucide-react'
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from '@/components/ui/dropdown-menu'
import { SidebarMenu, SidebarMenuButton, SidebarMenuItem } from '@/components/ui/sidebar'
import { useFilters } from '@/lib/filters'
import type { Guild } from '@/lib/data'

function GuildIcon({ guild }: { guild: Guild | null }) {
  return guild?.icon_url ? (
    <img src={guild.icon_url} alt="" className="size-8 rounded-lg" />
  ) : (
    <div className="flex size-8 items-center justify-center rounded-lg bg-muted text-sm font-semibold">
      {(guild?.name ?? '?').charAt(0).toUpperCase()}
    </div>
  )
}

export function GuildSwitcher({ guilds, guild }: { guilds: Guild[]; guild: Guild | null }) {
  const [, update] = useFilters()
  return (
    <SidebarMenu>
      <SidebarMenuItem>
        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <SidebarMenuButton size="lg">
              <GuildIcon guild={guild} />
              <span className="truncate font-medium">{guild?.name ?? 'VibeShift'}</span>
              <ChevronsUpDown className="ml-auto size-4" />
            </SidebarMenuButton>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="start" className="min-w-56">
            {guilds.map((g) => (
              <DropdownMenuItem
                key={g.id}
                onSelect={() => update({ guild: g.guild_id, people: [], channels: [] })}
              >
                <GuildIcon guild={g} />
                {g.name}
              </DropdownMenuItem>
            ))}
          </DropdownMenuContent>
        </DropdownMenu>
      </SidebarMenuItem>
    </SidebarMenu>
  )
}
