import { useEffect, useState } from 'react'

type Raw<T> = { key: string; data: T | null; error: string | null } | null
export type Fetched<T> = { data: T | null; error: string | null; loading: boolean }

/** Pure: only a result tagged with the current key counts; anything else is "absent". */
export function selectFetched<T>(raw: Raw<T>, key: string): Fetched<T> {
  if (!raw || raw.key !== key) return { data: null, error: null, loading: true }
  return { data: raw.data, error: raw.error, loading: false }
}

/**
 * Runs `fn` whenever `key` changes. Results are tagged with their key, so data
 * (or an error) from a previous key is never returned for the current one.
 * `key` must encode every input `fn` closes over (primitives only).
 */
export function useFetched<T>(key: string, fn: () => Promise<T>): Fetched<T> {
  const [raw, setRaw] = useState<Raw<T>>(null)

  useEffect(() => {
    let live = true
    fn().then(
      (data) => live && setRaw({ key, data, error: null }),
      (e) => live && setRaw({ key, data: null, error: String(e?.message ?? e) }),
    )
    return () => {
      live = false
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps -- `key` encodes fn's inputs
  }, [key])

  return selectFetched(raw, key)
}
