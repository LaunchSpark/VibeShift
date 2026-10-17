"""Scorer worker: drains the pending deque newest-first and maintains daily rollups."""
import asyncio
import logging
from datetime import datetime
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

    def _call(self, texts: list[str]) -> list[Score]:
        out = list(self.score_fn(texts))
        if len(out) != len(texts):
            raise ValueError(f"score_fn returned {len(out)} results for {len(texts)} texts")
        return out

    async def _score(self, msgs: list[StoredMessage]) -> list[Score | None]:
        """Score serially in a worker thread; on chunk failure retry each message singly."""
        try:
            return await asyncio.to_thread(self._call, [m.content for m in msgs])
        except Exception:
            log.warning("chunk scoring failed; retrying singly", exc_info=True)
        out: list[Score | None] = []
        for m in msgs:
            try:
                out.append((await asyncio.to_thread(self._call, [m.content]))[0])
            except Exception:
                log.exception("scoring failed for message %s", m.message_id)
                out.append(None)
        return out

    async def _fresh(self, popped: StoredMessage) -> StoredMessage | None:
        """The current row, or None if it was edited/deleted since it was popped."""
        fresh = await self.store.get_message(popped.message_id)
        if (fresh is None or fresh.deleted_at is not None
                or fresh.priority_key != popped.priority_key
                or fresh.edited_at != popped.edited_at
                or fresh.content != popped.content):
            return None
        return fresh

    async def _commit(self, popped: StoredMessage, score: Score) -> bool:
        async with self._lock:
            fresh = await self._fresh(popped)
            if fresh is None:
                return False  # stale score; the row stays pending
            # The day is the stored bucket of created_at, never the edit time.
            args = (fresh.guild_id, fresh.channel_id, fresh.author_id, fresh.day)
            before = await self.store.get_rollup(*args)
            count, sums = before if before else (0, {})
            if fresh.rollup_applied and fresh.probabilities:
                sums, count = rollups.apply(sums, count, fresh.probabilities, -1)
            sums, count = rollups.apply(sums, count, score.probabilities, +1)
            if count <= 0:
                count, sums = 0, {k: 0.0 for k in sums}  # no float residue
            await self.store.put_rollup(*args, count, sums)
            try:
                await self.store.save_score(
                    fresh.message_id, score.emotion, score.probabilities, score.confidence,
                    MODEL_ID, rollup_applied=True)
            except Exception:
                # keep rollup and row consistent: undo the rollup write
                try:
                    b_count, b_sums = before if before else (0, {k: 0.0 for k in sums})
                    await self.store.put_rollup(*args, b_count, b_sums)
                except Exception:
                    log.exception("rollup rollback failed for %s", fresh.message_id)
                raise
            return True

    async def _fail(self, popped: StoredMessage) -> None:
        async with self._lock:
            if await self._fresh(popped) is not None:
                await self.store.mark_failed(popped.message_id)

    async def on_edited(self, message_id: str, content: str, edited_at: datetime) -> None:
        async with self._lock:
            await self.store.mark_edited(message_id, content, edited_at)
        self.wake()

    async def run_once(self) -> int:
        msgs = await self.store.pop_chunk(self.chunk)
        if not msgs:
            return 0
        scores = await self._score(msgs)
        for m, s in zip(msgs, scores):
            try:
                if s is None:
                    await self._fail(m)
                else:
                    await self._commit(m, s)
            except Exception:
                log.exception("commit failed for message %s", m.message_id)
                try:
                    await self._fail(m)
                except Exception:
                    log.exception("mark_failed failed for message %s", m.message_id)
        return len(msgs)

    async def run_forever(self, idle_sleep: float = 1.0) -> None:
        while True:
            self._wake.clear()  # before working, so a wake during run_once isn't lost
            try:
                n = await self.run_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("scorer iteration failed")
                n = 0
            if n == 0:
                try:
                    await asyncio.wait_for(self._wake.wait(), idle_sleep)
                except asyncio.TimeoutError:
                    pass

    async def on_deleted(self, msg: StoredMessage) -> None:
        async with self._lock:
            fresh = await self.store.get_message(msg.message_id)
            if fresh and fresh.rollup_applied and fresh.probabilities:
                args = (fresh.guild_id, fresh.channel_id, fresh.author_id, fresh.day)
                cur = await self.store.get_rollup(*args)
                count, sums = cur if cur else (0, {})
                sums, count = rollups.apply(sums, count, fresh.probabilities, -1)
                if count <= 0:
                    count, sums = 0, {k: 0.0 for k in sums}
                await self.store.put_rollup(*args, count, sums)
                await self.store.save_score(
                    fresh.message_id, fresh.emotion, fresh.probabilities,
                    fresh.confidence or 0.0, fresh.model_id, rollup_applied=False)
