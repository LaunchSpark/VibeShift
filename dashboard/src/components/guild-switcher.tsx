import { ChevronsUpDown } from 'lucide-react'
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuGroup,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { SidebarMenu, SidebarMenuButton, SidebarMenuItem } from '@/components/ui/sidebar'
import { useFilters } from '@/lib/filters'
import type { Guild } from '@/lib/data'

function GuildIcon({ guild, size = 'default' }: { guild: Guild | null; size?: 'default' | 'sm' }) {
  return (
    <Avatar size={size}>
      {guild?.icon_url && <AvatarImage src={guild.icon_url} alt="" />}
      <AvatarFallback>{(guild?.name ?? '?').charAt(0).toUpperCase()}</AvatarFallback>
    </Avatar>
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
              <ChevronsUpDown className="ml-auto" />
            </SidebarMenuButton>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="start" className="min-w-56">
            <DropdownMenuGroup>
              {guilds.map((g) => (
                <DropdownMenuItem
                  key={g.id}
                  onSelect={() => update({ guild: g.guild_id, people: [], channels: [] })}
                >
                  <GuildIcon guild={g} size="sm" />
                  {g.name}
                </DropdownMenuItem>
              ))}
            </DropdownMenuGroup>
          </DropdownMenuContent>
        </DropdownMenu>
      </SidebarMenuItem>
    </SidebarMenu>
  )
}
