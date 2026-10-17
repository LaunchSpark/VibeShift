"""Scorer worker: drains the pending deque newest-first and maintains daily rollups."""
import asyncio
import logging
from collections.abc import Callable
from zoneinfo import ZoneInfo

from vibeshift import rollups
from vibeshift.emotions import MODEL_ID, Score
from vibeshift.store import Store, StoredMessage

log = logging.getLogger(__name__)


class Scorer:
    def __init__(self, store: Store, score_fn: Callable[[list[str]], list[Score]],
                 chunk: int, tz: ZoneInfo):
        self.store = store
        self.score_fn = score_fn
        self.chunk = chunk
        self.tz = tz
        self._lock = asyncio.Lock()  # guards every rollup read-modify-write
        self._wake = asyncio.Event()

    def wake(self) -> None:
        self._wake.set()

    async def _score(self, msgs: list[StoredMessage]) -> list[Score | None]:
        """Score serially in a worker thread; on chunk failure retry each message singly."""
        try:
            return list(await asyncio.to_thread(self.score_fn, [m.content for m in msgs]))
        except Exception:
            log.warning("chunk scoring failed; retrying singly", exc_info=True)
        out: list[Score | None] = []
        for m in msgs:
            try:
                out.append((await asyncio.to_thread(self.score_fn, [m.content]))[0])
            except Exception:
                log.exception("scoring failed for message %s", m.message_id)
                out.append(None)
        return out

    async def _adjust(self, m: StoredMessage, probs: dict, sign: int) -> None:
        # The day is the stored bucket of created_at, never the edit time.
        cur = await self.store.get_rollup(m.guild_id, m.channel_id, m.author_id, m.day)
        count, sums = cur if cur else (0, {})
        sums, count = rollups.apply(sums, count, probs, sign)
        if count <= 0:
            count, sums = 0, {k: 0.0 for k in sums}  # no float residue
        await self.store.put_rollup(m.guild_id, m.channel_id, m.author_id, m.day, count, sums)

    async def _commit(self, popped: StoredMessage, score: Score) -> bool:
        async with self._lock:
            fresh = await self.store.get_message(popped.message_id)
            if (fresh is None or fresh.deleted_at is not None
                    or fresh.priority_key != popped.priority_key
                    or fresh.edited_at != popped.edited_at
                    or fresh.content != popped.content):
                return False  # edited/deleted mid-scoring: stale score, stays pending
            if fresh.rollup_applied and fresh.probabilities:
                await self._adjust(fresh, fresh.probabilities, -1)
            await self._adjust(fresh, score.probabilities, +1)
            await self.store.save_score(
                fresh.message_id, score.emotion, score.probabilities, score.confidence,
                MODEL_ID, rollup_applied=True)
            return True

    async def run_once(self) -> int:
        msgs = await self.store.pop_chunk(self.chunk)
        if not msgs:
            return 0
        scores = await self._score(msgs)
        for m, s in zip(msgs, scores):
            if s is None:
                await self.store.mark_failed(m.message_id)
            else:
                await self._commit(m, s)
        return len(msgs)

    async def run_forever(self, idle_sleep: float = 1.0) -> None:
        while True:
            try:
                n = await self.run_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("scorer iteration failed")
                n = 0
            if n == 0:
                self._wake.clear()
                try:
                    await asyncio.wait_for(self._wake.wait(), idle_sleep)
                except asyncio.TimeoutError:
                    pass

    async def on_deleted(self, msg: StoredMessage) -> None:
        async with self._lock:
            fresh = await self.store.get_message(msg.message_id)
            if fresh and fresh.rollup_applied and fresh.probabilities:
                await self._adjust(fresh, fresh.probabilities, -1)
                await self.store.save_score(
                    fresh.message_id, fresh.emotion, fresh.probabilities,
                    fresh.confidence or 0.0, fresh.model_id, rollup_applied=False)
