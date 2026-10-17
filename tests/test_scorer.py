import asyncio
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from vibeshift.emotions import Score
from vibeshift.scorer import Scorer
from vibeshift.store import NewMessage, Store

T0 = datetime(2026, 10, 17, 12, 0, tzinfo=timezone.utc)
TZ = ZoneInfo("UTC")
DAY = "2026-10-17"


def msg(i: int, **kw) -> NewMessage:
    base = dict(message_id=str(i), guild_id="g", channel_id="c", author_id="a",
                created_at=T0, content=f"m{i}", day=DAY)
    base.update(kw)
    return NewMessage(**base)


def fake(probs=None):
    probs = probs or {"joy": 0.7, "anger": 0.3}

    def fn(texts):
        return [Score("joy", dict(probs), 0.7) for _ in texts]
    return fn


@pytest.fixture
async def store(pb_url):
    async with Store(pb_url) as s:
        yield s


async def rollup(store):
    return await store.get_rollup("g", "c", "a", DAY)


async def test_scores_chunk_and_writes_rollup(store):
    for i in (1, 2, 3):
        await store.insert_pending(msg(i))
    n = await Scorer(store, fake(), 10, TZ).run_once()
    assert n == 3
    count, sums = await rollup(store)
    assert count == 3
    assert sums["joy"] == pytest.approx(2.1) and sums["anger"] == pytest.approx(0.9)
    assert await store.pop_chunk(10) == []


async def test_rescore_after_edit_does_not_double_count(store):
    await store.insert_pending(msg(1))
    await Scorer(store, fake(), 10, TZ).run_once()
    await store.mark_edited("1", "new", datetime(2026, 10, 18, tzinfo=timezone.utc))
    await Scorer(store, fake({"joy": 0.1, "anger": 0.9}), 10, TZ).run_once()
    count, sums = await rollup(store)
    assert count == 1
    assert sums["joy"] == pytest.approx(0.1) and sums["anger"] == pytest.approx(0.9)


async def test_delete_after_score_subtracts(store):
    await store.insert_pending(msg(1))
    sc = Scorer(store, fake(), 10, TZ)
    await sc.run_once()
    deleted = await store.mark_deleted("1", T0)
    await sc.on_deleted(deleted)
    count, sums = await rollup(store)
    assert count == 0
    assert all(v == 0 for v in sums.values())


async def test_delete_before_score_is_noop(store):
    await store.insert_pending(msg(1))
    deleted = await store.mark_deleted("1", T0)
    await Scorer(store, fake(), 10, TZ).on_deleted(deleted)
    assert await rollup(store) is None


async def test_failed_chunk_marks_failed_and_continues(store):
    for i in (1, 2, 3):
        await store.insert_pending(msg(i))

    def fn(texts):
        if "m2" in texts:
            raise RuntimeError("boom")
        return [Score("joy", {"joy": 1.0}, 1.0) for _ in texts]

    await Scorer(store, fn, 10, TZ).run_once()
    count, _ = await rollup(store)
    assert count == 2
    m2 = await store.get_message("2")
    assert m2.score_status == "failed"
    assert (await store.get_message("1")).score_status == "scored"
    assert (await store.get_message("3")).score_status == "scored"


async def test_edit_during_scoring_not_lost(store):
    await store.insert_pending(msg(1))
    loop = asyncio.get_running_loop()

    def fn(texts):
        asyncio.run_coroutine_threadsafe(
            store.mark_edited("1", "edited", datetime(2026, 10, 18, tzinfo=timezone.utc)),
            loop).result()
        return [Score("joy", {"joy": 1.0}, 1.0) for _ in texts]

    await Scorer(store, fn, 10, TZ).run_once()
    m = await store.get_message("1")
    assert m.score_status == "pending" and m.content == "edited"
    assert await rollup(store) is None
    # picked up again with the new content
    await Scorer(store, fake({"joy": 1.0}), 10, TZ).run_once()
    assert (await store.get_message("1")).score_status == "scored"
    assert (await rollup(store))[0] == 1


async def test_rollup_uses_stored_day_not_edit_time(store):
    await store.insert_pending(msg(1))
    sc = Scorer(store, fake(), 10, TZ)
    await sc.run_once()
    await store.mark_edited("1", "x", datetime(2026, 10, 25, tzinfo=timezone.utc))
    await sc.run_once()
    assert await store.get_rollup("g", "c", "a", "2026-10-25") is None
    assert (await rollup(store))[0] == 1


async def test_wake_ends_idle_wait(store):
    sc = Scorer(store, fake(), 10, TZ)
    task = asyncio.create_task(sc.run_forever(idle_sleep=60))
    await asyncio.sleep(0.5)
    await store.insert_pending(msg(1))
    sc.wake()
    for _ in range(50):
        await asyncio.sleep(0.1)
        if await rollup(store):
            break
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert (await rollup(store))[0] == 1


async def test_on_edited_mid_commit_leaves_pending(store):
    await store.insert_pending(msg(1))
    sc = Scorer(store, fake({"joy": 1.0}), 10, TZ)
    real_put = store.put_rollup
    tasks = []

    async def put(*a, **k):
        await real_put(*a, **k)
        if not tasks:  # edit arrives between rollup write and save
            tasks.append(asyncio.create_task(
                sc.on_edited("1", "edited", datetime(2026, 10, 18, tzinfo=timezone.utc))))
            await asyncio.sleep(0.2)

    store.put_rollup = put
    await sc.run_once()
    await tasks[0]
    store.put_rollup = real_put
    m = await store.get_message("1")
    assert m.score_status == "pending" and m.content == "edited"
    await sc.run_once()  # re-score subtracts old contribution first
    assert (await rollup(store))[0] == 1
    assert (await store.get_message("1")).score_status == "scored"


async def test_save_failure_isolated_and_not_double_counted(store):
    for i in (1, 2, 3):
        await store.insert_pending(msg(i))
    real = store.save_score

    async def save(message_id, *a, **k):
        if message_id == "2":
            raise RuntimeError("db down")
        return await real(message_id, *a, **k)

    store.save_score = save
    sc = Scorer(store, fake(), 10, TZ)
    assert await sc.run_once() == 3
    assert (await rollup(store))[0] == 2
    assert (await store.get_message("2")).score_status == "failed"
    assert (await store.get_message("1")).score_status == "scored"
    assert (await store.get_message("3")).score_status == "scored"
    assert await sc.run_once() == 0
    assert (await rollup(store))[0] == 2


async def test_cancel_during_save_rolls_back_rollup(store):
    await store.insert_pending(msg(1))
    sc = Scorer(store, fake(), 10, TZ)
    popped = (await store.pop_chunk(1))[0]

    async def save(*a, **k):
        raise asyncio.CancelledError  # bot shutdown between put_rollup and save_score

    store.save_score = save
    with pytest.raises(asyncio.CancelledError):
        await sc._commit(popped, fake()(["x"])[0])
    r = await rollup(store)
    assert r is None or r[0] == 0  # not left counted while the row is still pending
    assert (await store.get_message("1")).score_status == "pending"
