import types
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import discord
import pytest

from vibeshift.history import create_downtime_jobs, job_loop, run_job, to_new_message
from vibeshift.store import Store

TZ = ZoneInfo("UTC")
T0 = datetime(2026, 10, 17, 12, 0, tzinfo=timezone.utc)


def ts(i: int) -> datetime:
    return T0 + timedelta(minutes=i)


class FakeAuthor:
    def __init__(self, id=1, bot=False):
        self.id, self.bot = id, bot
        self.display_name = f"user{id}"
        self.display_avatar = types.SimpleNamespace(url=f"http://a/{id}.png")


def fm(i, content=None, author=None, reply=None, channel_id=7, created=None):
    return types.SimpleNamespace(
        id=i, content=f"m{i}" if content is None else content,
        author=author or FakeAuthor(), created_at=created or ts(i),
        guild=types.SimpleNamespace(id=5), channel=types.SimpleNamespace(id=channel_id),
        reference=types.SimpleNamespace(message_id=reply) if reply else None)


class FakeChannel:
    type = discord.ChannelType.text

    def __init__(self, id, name, msgs, forbidden=False):
        self.id, self.name, self.msgs, self.forbidden = id, name, msgs, forbidden
        self.calls = []
        self.yielded = 0
        self.ignore_after = False

    async def history(self, limit=100, before=None, after=None, oldest_first=None):
        self.calls.append(dict(before=before, after=after, oldest_first=oldest_first))
        if self.forbidden:
            raise discord.Forbidden(types.SimpleNamespace(status=403, reason="Forbidden"), "missing access")

        def ok(m, bound, gt):
            if bound is None:
                return True
            v, mv = (bound.id, m.id) if hasattr(bound, "id") else (bound, m.created_at)
            return mv > v if gt else mv < v
        out = [m for m in self.msgs if (self.ignore_after or ok(m, after, True)) and ok(m, before, False)]
        # discord.py defaults to oldest_first when `after` is given
        asc = oldest_first if oldest_first is not None else after is not None
        for m in sorted(out, key=lambda m: m.id, reverse=not asc):
            self.yielded += 1
            yield m


def guild(*channels):
    return types.SimpleNamespace(id=5, name="G", icon=None, text_channels=list(channels))


@pytest.fixture
async def store(pb_url):
    async with Store(pb_url) as s:
        yield s


async def job(store, origin, target=None):
    await store.create_job("5", origin, target)
    j = await store.next_pending_job()
    return j


async def final(store, j):
    return await store._first("backfill_jobs", f'id="{j["id"]}"')


async def all_ids(store):
    r = await store._list("messages", sort="priority_key", per_page=100)
    return [x["message_id"] for x in r]


async def test_downtime_fetches_after_cursor_only(store):
    await store.upsert_channel("7", "5", "gen", "text")
    await store.set_cursors("7", newest="100", oldest="100")
    ch = FakeChannel(7, "gen", [fm(90), fm(110), fm(120)])
    j = await job(store, "downtime")
    await run_job(j, guild(ch), store, TZ)
    assert await all_ids(store) == ["110", "120"]
    rec = await store.get_channel("7")
    assert rec["newest_fetched_id"] == "120" and rec["oldest_fetched_id"] == "100"
    assert (await final(store, j))["status"] == "done"
    assert ch.calls[0]["oldest_first"] is True


async def test_downtime_skips_channel_without_cursor(store):
    ch = FakeChannel(7, "gen", [fm(110)])
    j = await job(store, "downtime")
    await run_job(j, guild(ch), store, TZ)
    assert await all_ids(store) == []


async def test_extend_stops_at_target_and_sets_depth(store):
    await store.upsert_channel("7", "5", "gen", "text")
    await store.set_cursors("7", newest="50", oldest="50")
    ch = FakeChannel(7, "gen", [fm(10, created=ts(-100)), fm(20, created=ts(-5)),
                                fm(30, created=ts(-4)), fm(60, created=ts(1))])
    target = ts(-10)
    j = await job(store, "extend", target)
    await run_job(j, guild(ch), store, TZ)
    assert await all_ids(store) == ["20", "30"]
    rec = await store.get_channel("7")
    assert rec["oldest_fetched_id"] == "20" and rec["newest_fetched_id"] == "50"
    assert ch.calls[0]["oldest_first"] is False
    assert ch.calls[0]["before"].id == 50
    g = await store._first("guilds", 'guild_id="5"')
    assert g["history_depth"].startswith(target.strftime("%Y-%m-%d %H:%M"))
    assert (await final(store, j))["status"] == "done"


async def test_extend_without_cursors_sets_both(store):
    ch = FakeChannel(7, "gen", [fm(20, created=ts(-5)), fm(30, created=ts(-4))])
    j = await job(store, "extend", ts(-10))
    await run_job(j, guild(ch), store, TZ)
    rec = await store.get_channel("7")
    assert rec["newest_fetched_id"] == "30" and rec["oldest_fetched_id"] == "20"
    assert ch.calls[0]["before"] is None


async def test_forbidden_channel_skipped(store):
    bad = FakeChannel(8, "locked", [], forbidden=True)
    good = FakeChannel(7, "gen", [fm(20, created=ts(-5))])
    j = await job(store, "extend", ts(-10))
    await run_job(j, guild(bad, good), store, TZ)
    f = await final(store, j)
    assert f["status"] == "done" and "8" in f["error"]
    assert await all_ids(store) == ["20"]


async def test_failure_marks_failed(store):
    class Boom(FakeChannel):
        async def history(self, **kw):
            raise RuntimeError("kaput")
            yield
    j = await job(store, "extend", ts(-10))
    await run_job(j, guild(Boom(7, "gen", [])), store, TZ)
    f = await final(store, j)
    assert f["status"] == "failed" and "kaput" in f["error"]


async def test_bot_and_empty_messages_ignored(store):
    ch = FakeChannel(7, "gen", [fm(1, author=FakeAuthor(2, bot=True)), fm(2, content="  \n"),
                                fm(3, content=""), fm(4)])
    j = await job(store, "extend", ts(-10) - timedelta(days=1))
    await run_job(j, guild(ch), store, TZ)
    assert await all_ids(store) == ["4"]


async def test_members_upserted_and_scanned(store):
    msgs = [fm(i, author=FakeAuthor(1 + i % 2), created=ts(-50) + timedelta(seconds=i))
            for i in range(1, 251)]
    ch = FakeChannel(7, "gen", msgs)
    j = await job(store, "extend", ts(-100))
    await run_job(j, guild(ch), store, TZ)
    ms = await store._list("members", per_page=10)
    assert {m["author_id"] for m in ms} == {"1", "2"}
    assert ms[0]["display_name"].startswith("user") and ms[0]["avatar_url"].startswith("http")
    assert (await final(store, j))["scanned"] == 250


def test_to_new_message_fields():
    m = fm(42, reply=41)
    n = to_new_message(m, ZoneInfo("America/New_York"))
    assert (n.message_id, n.channel_id, n.guild_id, n.author_id) == ("42", "7", "5", "1")
    assert n.reply_to_id == "41" and n.created_at.tzinfo is not None
    assert n.day == ts(42).astimezone(ZoneInfo("America/New_York")).date().isoformat()
    assert to_new_message(fm(1), TZ).reply_to_id == ""
    naive = fm(1, created=datetime(2026, 1, 1))
    assert to_new_message(naive, TZ).created_at.tzinfo is not None


async def test_create_downtime_jobs(store):
    await store.upsert_channel("7", "5", "gen", "text")
    await store.set_cursors("7", newest="100")
    await store.upsert_channel("9", "6", "x", "text")  # guild 6 has no cursor
    g1 = guild(FakeChannel(7, "gen", []))
    g2 = types.SimpleNamespace(id=6, name="H", icon=None, text_channels=[FakeChannel(9, "x", [])])
    await create_downtime_jobs(types.SimpleNamespace(guilds=[g1, g2]), store)
    jobs = await store._list("backfill_jobs", per_page=10)
    assert [(j["guild_id"], j["origin"]) for j in jobs] == [("5", "downtime")]


async def test_job_loop_runs_pending_job(store):
    import asyncio
    await store.upsert_channel("7", "5", "gen", "text")
    await store.set_cursors("7", newest="100")
    ch = FakeChannel(7, "gen", [fm(110)])
    await store.create_job("5", "downtime")
    bot = types.SimpleNamespace(get_guild=lambda gid: guild(ch) if gid == 5 else None)
    t = asyncio.create_task(job_loop(bot, store, TZ, 0.05))
    for _ in range(100):
        await asyncio.sleep(0.1)
        if await store.next_pending_job() is None and "110" in await all_ids(store):
            break
    t.cancel()
    with pytest.raises(asyncio.CancelledError):
        await t
    assert await all_ids(store) == ["110"]


class Flaky:
    """Store proxy whose insert_pending raises on one message id."""
    def __init__(self, store, bad):
        self._s, self._bad = store, bad

    def __getattr__(self, k):
        return getattr(self._s, k)

    async def insert_pending(self, nm):
        if nm.message_id == self._bad:
            raise RuntimeError("db down")
        return await self._s.insert_pending(nm)


async def test_downtime_cursor_stops_before_failed_message(store):
    await store.upsert_channel("7", "5", "gen", "text")
    await store.set_cursors("7", newest="100", oldest="100")
    ch = FakeChannel(7, "gen", [fm(110), fm(120), fm(130)])
    j = await job(store, "downtime")
    await run_job(j, guild(ch), Flaky(store, "120"), TZ)
    assert (await store.get_channel("7"))["newest_fetched_id"] == "110"
    assert (await final(store, j))["status"] == "failed"


async def test_extend_cursor_stops_before_failed_message(store):
    await store.upsert_channel("7", "5", "gen", "text")
    await store.set_cursors("7", newest="50", oldest="50")
    ch = FakeChannel(7, "gen", [fm(20, created=ts(-5)), fm(30, created=ts(-4)), fm(40, created=ts(-3))])
    j = await job(store, "extend", ts(-10))
    await run_job(j, guild(ch), Flaky(store, "30"), TZ)
    assert (await store.get_channel("7"))["oldest_fetched_id"] == "40"


async def test_later_extend_does_not_shrink_depth(store):
    ch = FakeChannel(7, "gen", [])
    for t in (ts(-100), ts(-10)):
        j = await job(store, "extend", t)
        await run_job(j, guild(ch), store, TZ)
    g = await store._first("guilds", 'guild_id="5"')
    assert g["history_depth"].startswith(ts(-100).strftime("%Y-%m-%d %H:%M"))


async def test_extend_breaks_past_target(store):
    msgs = [fm(i, created=ts(i - 30)) for i in range(1, 30)]
    ch = FakeChannel(7, "gen", msgs)
    ch.ignore_after = True
    j = await job(store, "extend", ts(-5))
    await run_job(j, guild(ch), store, TZ)
    assert ch.yielded <= 5
    assert sorted(await all_ids(store)) == ["26", "27", "28", "29"]


async def test_reset_running_jobs(store):
    await store.create_job("5", "downtime")
    j = await store.next_pending_job()
    await store.update_job(j["id"], status="running")
    assert await store.next_pending_job() is None
    await store.reset_running_jobs()
    assert (await store.next_pending_job())["id"] == j["id"]


async def test_create_downtime_jobs_skips_active_and_tracks(pb_url):
    async with Store(pb_url) as store:
        await store.upsert_channel("7", "5", "gen", "text")
        await store.set_cursors("7", newest="10", oldest="10")
        guild = types.SimpleNamespace(id=5, text_channels=[types.SimpleNamespace(id=7)])
        bot = types.SimpleNamespace(guilds=[guild])
        out: set = set()
        await create_downtime_jobs(bot, store, out)
        await create_downtime_jobs(bot, store, out)
        assert out == {"5"}
        assert await store.has_active_job("5", "downtime")
        first = await store.next_pending_job()
        await store.update_job(first["id"], status="done")
        assert not await store.has_active_job("5", "downtime")
        assert await store.next_pending_job() is None  # only one job was ever created


async def test_job_loop_retries_when_reset_fails():
    import asyncio

    class FlakyStore:
        calls = 0

        async def reset_running_jobs(self):
            FlakyStore.calls += 1
            if FlakyStore.calls < 3:
                raise RuntimeError("pocketbase down")

        async def next_pending_job(self):
            return None

    task = asyncio.create_task(job_loop(types.SimpleNamespace(), FlakyStore(), TZ, 0.01))
    for _ in range(100):
        if FlakyStore.calls >= 3:
            break
        await asyncio.sleep(0.02)
    task.cancel()
    assert FlakyStore.calls >= 3


async def test_failed_downtime_job_keeps_flag_and_requeues(pb_url):
    import asyncio
    async with Store(pb_url) as store:
        await store.upsert_channel("7", "5", "gen", "text")
        await store.set_cursors("7", newest="10", oldest="10")
        ch = FakeChannel(7, "gen", [fm(11)])

        async def boom(*a, **k):
            raise RuntimeError("discord down")
            yield
        ch.history = boom
        guild = types.SimpleNamespace(id=5, name="g", icon=None, text_channels=[ch])
        bot = types.SimpleNamespace(guilds=[guild], get_guild=lambda i: guild)
        out: set = set()
        await create_downtime_jobs(bot, store, out)
        task = asyncio.create_task(job_loop(bot, store, TZ, 0.01, out))
        await asyncio.sleep(0.5)
        task.cancel()
        assert out == {"5"}
        assert await store.has_active_job("5", "downtime")
        assert (await store.get_channel("7"))["newest_fetched_id"] == "10"
        from vibeshift.bot import Handlers
        from vibeshift.scorer import Scorer
        h = Handlers(store, Scorer(store, lambda t: [], 8, TZ), TZ)
        h.startup_pending = False
        h.outstanding_downtime = out
        live = fm(50, channel_id=7)
        live.guild, live.channel = types.SimpleNamespace(id=5), types.SimpleNamespace(id=7, name="gen")
        await h.on_message(live)
        assert (await store.get_channel("7"))["newest_fetched_id"] == "10"  # flag held


async def test_failing_guild_does_not_stall_other_jobs(pb_url):
    import asyncio
    async with Store(pb_url) as store:
        await store.upsert_channel("7", "5", "gen", "text")
        await store.set_cursors("7", newest="10", oldest="10")

        async def boom(*a, **k):
            raise RuntimeError("down")
            yield
        bad = FakeChannel(7, "gen", [])
        bad.history = boom
        ga = types.SimpleNamespace(id=5, name="a", icon=None, text_channels=[bad])
        gb = types.SimpleNamespace(id=6, name="b", icon=None, text_channels=[])
        bot = types.SimpleNamespace(guilds=[ga, gb],
                                    get_guild=lambda i: {5: ga, 6: gb}.get(i))
        out: set = set()
        await create_downtime_jobs(bot, store, out)  # guild 5 downtime
        ext = await store.create_job("6", "extend", ts(-5))
        task = asyncio.create_task(job_loop(bot, store, TZ, 1.0, out))  # backoff >= 2s
        for _ in range(40):
            await asyncio.sleep(0.1)
            if (await store._list("backfill_jobs", f'id="{ext}" && status="done"')):
                break
        task.cancel()
        assert await store._list("backfill_jobs", f'id="{ext}" && status="done"')
        assert out == {"5"}


async def test_downtime_job_for_unknown_guild_fails_and_drops_flag(pb_url):
    import asyncio
    async with Store(pb_url) as store:
        jid = await store.create_job("99", "downtime")
        bot = types.SimpleNamespace(guilds=[], get_guild=lambda i: None, is_ready=lambda: True)
        out = {"99"}
        task = asyncio.create_task(job_loop(bot, store, TZ, 0.01, out))
        await asyncio.sleep(0.5)
        task.cancel()
        rec = (await store._list("backfill_jobs", f'id="{jid}"'))[0]
        assert rec["status"] == "failed" and rec["error"] == "bot is not in this guild"
        assert out == set()


async def test_transient_missing_guild_keeps_job_and_flag(pb_url):
    import asyncio
    async with Store(pb_url) as store:
        jid = await store.create_job("99", "downtime")
        for bot in (
            types.SimpleNamespace(guilds=[], get_guild=lambda i: None, is_ready=lambda: False),
            types.SimpleNamespace(guilds=[types.SimpleNamespace(id=99)],
                                  get_guild=lambda i: types.SimpleNamespace(id=99, unavailable=True),
                                  is_ready=lambda: True),
            types.SimpleNamespace(guilds=[types.SimpleNamespace(id=99)],
                                  get_guild=lambda i: None, is_ready=lambda: True),
        ):
            out = {"99"}
            task = asyncio.create_task(job_loop(bot, store, TZ, 0.01, out))
            await asyncio.sleep(0.6)
            task.cancel()
            rec = (await store._list("backfill_jobs", f'id="{jid}"'))[0]
            assert rec["status"] == "pending" and out == {"99"}
