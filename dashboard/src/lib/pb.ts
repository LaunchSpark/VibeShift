import PocketBase from 'pocketbase'

const url =
  (import.meta as { env?: Record<string, string | undefined> }).env?.VITE_POCKETBASE_URL ??
  'http://127.0.0.1:8090'

export const pb = new PocketBase(url)
// Overlapping React effects reuse request keys; auto-cancel would abort them.
pb.autoCancellation(false)
