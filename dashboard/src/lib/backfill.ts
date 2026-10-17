/** History only grows earlier: a target must precede the current depth (or today when there is none). */
export function canExtend(currentDepth: string | null, target: string, today: string): boolean {
  return target < (currentDepth ?? today)
}
