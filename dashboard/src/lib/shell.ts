import { useOutletContext } from 'react-router'
import type { Guild } from './data'

export type ShellContext = { guilds: Guild[]; guild: Guild | null }
export const useShell = () => useOutletContext<ShellContext>()
