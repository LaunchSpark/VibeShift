import { useEffect, useState } from 'react'

export type Fetched<T> = { data: T | null; error: string | null; loading: boolean }

/**
 * Runs `fn` whenever `key` changes. Keeps the previous data while refetching,
 * clears the error on success, and ignores results from superseded runs.
 * `key` must encode every input `fn` closes over (primitives only).
 */
export function useFetched<T>(key: string, fn: () => Promise<T>): Fetched<T> {
  const [state, setState] = useState<Fetched<T>>({ data: null, error: null, loading: true })

  useEffect(() => {
    let live = true
    setState((s) => (s.loading ? s : { ...s, loading: true }))
    fn().then(
      (data) => live && setState({ data, error: null, loading: false }),
      (e) => live && setState((s) => ({ data: s.data, error: String(e?.message ?? e), loading: false })),
    )
    return () => {
      live = false
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps -- `key` encodes fn's inputs
  }, [key])

  return state
}
