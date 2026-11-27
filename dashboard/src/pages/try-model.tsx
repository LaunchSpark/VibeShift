import { useCallback, useEffect, useMemo, useState } from 'react'
import type { CSSProperties } from 'react'
import { LoaderCircle, Send } from 'lucide-react'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'
import { Textarea } from '@/components/ui/textarea'
import { EMOTION_COLORS, LABELS, labelName, type Label } from '@/lib/emotions'

type Prediction = {
  emotion: Label
  confidence: number
  probabilities: Record<Label, number>
  model_id: string
}

export default function TryModelPage() {
  const [message, setMessage] = useState('')
  const [prediction, setPrediction] = useState<Prediction | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [apiReady, setApiReady] = useState<boolean | null>(null)

  useEffect(() => {
    fetch('/api/model-health').then((response) => setApiReady(response.ok)).catch(() => setApiReady(false))
  }, [])

  const scores = useMemo(() => prediction
    ? LABELS.map((label) => ({ label, value: prediction.probabilities[label] ?? 0 }))
      .sort((a, b) => b.value - a.value)
    : [], [prediction])

  const submit = useCallback(async () => {
    if (!message.trim() || loading) return
    setLoading(true)
    setError(null)
    setPrediction(null)
    try {
      const response = await fetch('/api/model-score', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: message.trim() }),
      })
      const data = await response.json()
      if (!response.ok) throw new Error(data.error ?? 'Scoring request failed.')
      setPrediction(data as Prediction)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Could not reach the local model API.')
    } finally {
      setLoading(false)
    }
  }, [loading, message])

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold">Try the model</h1>
        <p className="mt-1 text-sm text-muted-foreground">Enter one message to see how the bot classifies its emotion.</p>
      </div>

      {apiReady === false && (
        <Alert>
          <AlertTitle>Bot scoring is offline</AlertTitle>
          <AlertDescription>Start the bot with <code>pixi run bot</code>. The dashboard sends requests to the bot’s local scoring API.</AlertDescription>
        </Alert>
      )}

      <Card>
        <CardHeader>
          <CardTitle>Single message</CardTitle>
          <CardDescription>Uses the same Laya checkpoint and scoring function as the bot. The message is not saved.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-3">
          <Textarea
            value={message}
            onChange={(event) => setMessage(event.target.value)}
            onKeyDown={(event) => { if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') void submit() }}
            maxLength={2000}
            placeholder="I’m excited to see everyone tonight!"
            aria-label="Message to classify"
            className="min-h-32"
          />
          <div className="flex items-center justify-between gap-3">
            <span className="text-xs text-muted-foreground">{message.length}/2000 · Ctrl+Enter to score</span>
            <Button onClick={() => void submit()} disabled={!message.trim() || loading}>
              {loading ? <LoaderCircle className="animate-spin" /> : <Send />}
              {loading ? 'Scoring…' : 'Score message'}
            </Button>
          </div>
          {error && <p role="alert" className="text-sm text-destructive">{error}</p>}
        </CardContent>
      </Card>

      {prediction && (
        <Card>
          <CardHeader className="flex-row items-center justify-between gap-4">
            <div>
              <CardTitle>Prediction</CardTitle>
              <CardDescription>Seven emotion scores returned by the live scoring path.</CardDescription>
            </div>
            <Badge variant="secondary" className="capitalize">{labelName(prediction.emotion)} · {(prediction.confidence * 100).toFixed(1)}%</Badge>
          </CardHeader>
          <CardContent className="space-y-4">
            {scores.map(({ label, value }) => (
              <div key={label} className="space-y-1.5">
                <div className="flex justify-between text-sm">
                  <span>{labelName(label)}</span><span className="text-muted-foreground">{(value * 100).toFixed(1)}%</span>
                </div>
                <Progress value={value * 100} className="h-2 [&_[data-slot=progress-indicator]]:bg-[var(--emotion-color)]" style={{ '--emotion-color': EMOTION_COLORS[label] } as CSSProperties} />
              </div>
            ))}
            <p className="pt-1 text-xs text-muted-foreground">{prediction.model_id}</p>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
