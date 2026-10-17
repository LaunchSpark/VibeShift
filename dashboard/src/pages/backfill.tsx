import { History } from 'lucide-react'
import { Empty, EmptyDescription, EmptyHeader, EmptyMedia, EmptyTitle } from '@/components/ui/empty'

export default function BackfillPage() {
  return (
    <Empty className="border">
      <EmptyHeader>
        <EmptyMedia variant="icon">
          <History />
        </EmptyMedia>
        <EmptyTitle>Backfill is coming soon</EmptyTitle>
        <EmptyDescription>Extending history will be available here.</EmptyDescription>
      </EmptyHeader>
    </Empty>
  )
}
