import { NavLink, useLocation } from 'react-router'
import { Activity, History, Sparkles, Users } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import {
  Sidebar,
  SidebarContent,
  SidebarGroup,
  SidebarGroupContent,
  SidebarGroupLabel,
  SidebarHeader,
  SidebarMenu,
  SidebarMenuButton,
  SidebarMenuItem,
} from '@/components/ui/sidebar'
import { GuildSwitcher } from '@/components/guild-switcher'
import type { Guild } from '@/lib/data'

const SOON = [
  { title: 'Impact', icon: Sparkles },
  { title: 'Auto-roles', icon: Users },
]

export function AppSidebar({ guilds, guild }: { guilds: Guild[]; guild: Guild | null }) {
  const { pathname, search } = useLocation()
  return (
    <Sidebar>
      <SidebarHeader>
        <GuildSwitcher guilds={guilds} guild={guild} />
      </SidebarHeader>
      <SidebarContent>
        <SidebarGroup>
          <SidebarGroupLabel>Stats</SidebarGroupLabel>
          <SidebarGroupContent>
            <SidebarMenu>
              <SidebarMenuItem>
                <SidebarMenuButton asChild isActive={pathname === '/'}>
                  <NavLink to={{ pathname: '/', search }}>
                    <Activity />
                    <span>Emotions</span>
                  </NavLink>
                </SidebarMenuButton>
              </SidebarMenuItem>
              <SidebarMenuItem>
                <SidebarMenuButton asChild isActive={pathname === '/backfill'}>
                  <NavLink to={{ pathname: '/backfill', search }}>
                    <History />
                    <span>Backfill</span>
                  </NavLink>
                </SidebarMenuButton>
              </SidebarMenuItem>
              {SOON.map(({ title, icon: Icon }) => (
                <SidebarMenuItem key={title}>
                  <SidebarMenuButton disabled aria-disabled>
                    <Icon />
                    <span>{title}</span>
                    <Badge variant="secondary" className="ml-auto">Soon</Badge>
                  </SidebarMenuButton>
                </SidebarMenuItem>
              ))}
            </SidebarMenu>
          </SidebarGroupContent>
        </SidebarGroup>
      </SidebarContent>
    </Sidebar>
  )
}
