/** One message to label. Only the text is shown, so nothing hints at its conversation. */
export type LabelItem = { id: string; content: string }

/** Fisher-Yates shuffle into a new array. */
export function shuffle<T>(items: readonly T[], random: () => number = Math.random): T[] {
  const out = [...items]
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(random() * (i + 1))
    ;[out[i], out[j]] = [out[j], out[i]]
  }
  return out
}

/** Placeholder queue (the evaluation sample messages) until the real data source is wired up. */
export async function fetchLabelQueue(): Promise<LabelItem[]> {
  return [
    { id: 'msg_001', content: 'oh great, another server crash' },
    { id: 'msg_002', content: 'I can’t believe we actually won!!' },
    { id: 'msg_003', content: 'i’m so sorry i forgot about the meeting' },
    { id: 'msg_004', content: 'wait what the hell is going on rn' },
    { id: 'msg_005', content: 'ugh this update broke everything again' },
    { id: 'msg_006', content: 'lol yeah sure that’ll totally work' },
    { id: 'msg_007', content: 'thanks everyone for the help today :)' },
    { id: 'msg_008', content: 'honestly kind of scared about the deadline' },
    { id: 'msg_009', content: 'k' },
    { id: 'msg_010', content: 'that’s so gross why would they do that' },
  ]
}
