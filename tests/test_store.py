from datetime import datetime, timezone

import pytest

from vibeshift.store import NewMessage, Store

T0 = datetime(2026, 10, 17, 12, 0, tzinfo=timezone.utc)


def msg(i: int, **kw) -> NewMessage:
    base = dict(message_id=str(i), guild_id="g", channel_id="c", author_id="a",
                created_at=T0, content=f"m{i}", day="2026-10-17")
    base.update(kw)
    return NewMessage(**base)


@pytest.fixture
async def store(pb_url):
    async with Store(pb_url) as s:
        yield s


async def ids(store, n):
    return [m.message_id for m in await store.pop_chunk(n)]


async def test_pop_newest_first(store):
    for i in (10, 30, 20):
        await store.insert_pending(msg(i))
    assert await ids(store, 2) == ["30", "20"]


async def test_extend_goes_to_bottom(store):
    await store.insert_pending(msg(100))
    await store.insert_pending(msg(200))
    await store.insert_pending(msg(50))
    assert await ids(store, 3) == ["200", "100", "50"]


async def test_live_beats_catchup_inserted_later(store):
    await store.insert_pending(msg(500))
    await store.insert_pending(msg(300))
    await store.insert_pending(msg(400))
    assert await ids(store, 1) == ["500"]


async def test_edit_goes_to_top(store):
    await store.insert_pending(msg(10))
    await store.insert_pending(msg(20))
    for i in (10, 20):
        await store.save_score(str(i), "joy", {"joy": 1.0}, 0.9, "m")
    await store.mark_edited("10", "new", datetime.now(timezone.utc))
    top = await store.pop_chunk(1)
    assert [m.message_id for m in top] == ["10"]
    assert top[0].content == "new"
    assert top[0].probabilities == {"joy": 1.0}
    assert top[0].edited_at is not None and top[0].deleted_at is None


async def test_mark_edited_unknown_is_noop(store):
    await store.mark_edited("999", "x", T0)
    assert await ids(store, 5) == []


async def test_pop_skips_deleted(store):
    await store.insert_pending(msg(10))
    gone = await store.mark_deleted("10", T0)
    assert gone is not None and gone.deleted_at == T0
    assert await ids(store, 5) == []
    assert await store.mark_deleted("nope", T0) is None


async def test_pop_does_not_mutate(store):
    await store.insert_pending(msg(10))
    assert await ids(store, 1) == ["10"]
    assert await ids(store, 1) == ["10"]


async def test_insert_idempotent(store):
    assert await store.insert_pending(msg(10)) is True
    assert await store.insert_pending(msg(10)) is False


async def test_failed_leaves_queue(store):
    await store.insert_pending(msg(10))
    await store.mark_failed("10")
    assert await ids(store, 5) == []


async def test_cursors_monotonic(store):
    await store.upsert_channel("c", "g", "general", "text")
    await store.set_cursors("c", newest="5")
    await store.set_cursors("c", newest="3")
    assert (await store.get_channel("c"))["newest_fetched_id"] == "5"
    await store.set_cursors("c", oldest="5")
    await store.set_cursors("c", oldest="7")
    assert (await store.get_channel("c"))["oldest_fetched_id"] == "5"
    await store.set_cursors("c", newest="10")  # longer string is bigger
    assert (await store.get_channel("c"))["newest_fetched_id"] == "10"


async def test_upserts_and_rollup_and_jobs(store):
    await store.upsert_guild("g", "G", "u")
    await store.upsert_guild("g", "G2", "u")
    await store.upsert_member("g", "a", "A", "u")
    await store.upsert_member("g", "a", "A2", "u")
    assert await store.get_rollup("g", "c", "a", "2026-10-17") is None
    await store.put_rollup("g", "c", "a", "2026-10-17", 2, {"joy": 1.5})
    await store.put_rollup("g", "c", "a", "2026-10-17", 3, {"joy": 2.5})
    assert await store.get_rollup("g", "c", "a", "2026-10-17") == (3, {"joy": 2.5})
    assert await store.next_pending_job() is None
    jid = await store.create_job("g", "extend", target=T0)
    job = await store.next_pending_job()
    assert job["id"] == jid and job["origin"] == "extend"
    await store.update_job(jid, status="done", scanned=4)
    assert await store.next_pending_job() is None
