import types
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from vibeshift.bot import Handlers
from vibeshift.emotions import Score
from vibeshift.scorer import Scorer
from vibeshift.store import Store
from tests.fixtures.chat import CHANNEL, GUILD, parse_chat

TZ = ZoneInfo("UTC")
DAY = "2026-10-17"
G, C = str(GUILD.id), str(CHANNEL.id)
COUNTS = {"PixelKnight": 3, "SaltySpittoon": 3, "cozy_cat": 4, "TechWizard": 2, "GrieferDan": 1}


def fake_score(texts):
    return [Score("joy", {"joy": 0.7, "anger": 0.3}, 0.7) for _ in texts]


@pytest.fixture
async def env(pb_url):
    async with Store(pb_url) as store:
        scorer = Scorer(store, fake_score, 8, TZ)
        h = Handlers(store, scorer, TZ)
        h.startup_pending = False  # tests simulate a started bot
        yield store, scorer, h


async def drain(scorer):
    while await scorer.run_once():
        pass


async def count(store, author_id):
    r = await store.get_rollup(G, C, author_id, DAY)
    return r[0] if r else 0


def by_author(chat):
    return {m.author.name: str(m.author.id) for m in chat}


async def test_full_flow(env):
    store, scorer, h = env
    chat = parse_chat()
    assert len(chat) == 14
    for m in chat:
        await h.on_message(m)
    assert len(await store.pop_chunk(100)) == 13  # bot_alert ignored
    assert await store.get_message(str(chat[-1].id)) is None

    await drain(scorer)
    assert await store.pop_chunk(100) == []
    ids = by_author(chat)
    for name, n in COUNTS.items():
        assert await count(store, ids[name]) == n, name

    # edit one of PixelKnight's messages: re-scored, count unchanged
    target = chat[0]
    before = (await store.get_rollup(G, C, ids["PixelKnight"], DAY))[1]
    await h.on_raw_message_edit(types.SimpleNamespace(
        guild_id=GUILD.id, message_id=target.id, data={"content": "ok the queue is fine now",
                                    "edited_timestamp": "2026-10-17T15:00:00.000000+00:00"}))
    assert (await store.get_message(str(target.id))).score_status == "pending"
    # payloads without content (embed-only updates) are ignored
    await h.on_raw_message_edit(types.SimpleNamespace(guild_id=GUILD.id, message_id=chat[1].id, data={"embeds": []}))
    assert (await store.get_message(str(chat[1].id))).score_status == "scored"
    await drain(scorer)
    assert (await store.get_message(str(target.id))).content == "ok the queue is fine now"
    assert await count(store, ids["PixelKnight"]) == 3
    assert (await store.get_rollup(G, C, ids["PixelKnight"], DAY))[1] == pytest.approx(before)

    # delete one: count drops
    await h.on_raw_message_delete(types.SimpleNamespace(guild_id=GUILD.id, message_id=chat[-2].id))
    assert await count(store, ids["cozy_cat"]) == 3
    await h.on_raw_message_delete(types.SimpleNamespace(guild_id=GUILD.id, message_id=999))  # unknown id is a no-op


async def test_cursors_seed_and_advance(env):
    store, scorer, h = env
    chat = parse_chat()
    await h.on_message(chat[0])
    rec = await store.get_channel(C)
    assert rec["newest_fetched_id"] == rec["oldest_fetched_id"] == str(chat[0].id)
    await h.on_message(chat[1])
    rec = await store.get_channel(C)
    assert rec["newest_fetched_id"] == str(chat[1].id)
    assert rec["oldest_fetched_id"] == str(chat[0].id)


async def test_downtime_job_blocks_cursor_advance(env):
    store, scorer, h = env
    chat = parse_chat()
    await h.on_message(chat[0])
    h.outstanding_downtime.add(G)
    await h.on_message(chat[1])
    rec = await store.get_channel(C)
    assert rec["newest_fetched_id"] == str(chat[0].id)  # not advanced
    assert len(await store.pop_chunk(100)) == 2  # still captured
    h.outstanding_downtime.discard(G)
    await h.on_message(chat[2])
    assert (await store.get_channel(C))["newest_fetched_id"] == str(chat[2].id)


async def test_startup_pending_blocks_cursor_advance(env):
    store, scorer, h = env
    chat = parse_chat()
    await h.on_message(chat[0])  # seeds (no cursors yet)
    h.startup_pending = True
    await h.on_message(chat[1])
    assert (await store.get_channel(C))["newest_fetched_id"] == str(chat[0].id)
    h.startup_pending = False
    await h.on_message(chat[2])
    assert (await store.get_channel(C))["newest_fetched_id"] == str(chat[2].id)


async def test_empty_edit_bulk_delete_and_dm(env):
    store, scorer, h = env
    chat = parse_chat()
    for m in chat:
        await h.on_message(m)
    await drain(scorer)
    ids = by_author(chat)
    await h.on_raw_message_edit(types.SimpleNamespace(
        guild_id=GUILD.id, message_id=chat[0].id, data={"content": "  "}))
    assert await count(store, ids["PixelKnight"]) == 2
    await h.on_raw_bulk_message_delete(types.SimpleNamespace(
        guild_id=GUILD.id, message_ids={chat[2].id, chat[5].id}))
    assert await count(store, ids["cozy_cat"]) == 2
    # DMs are ignored
    await h.on_raw_message_delete(types.SimpleNamespace(guild_id=None, message_id=chat[1].id))
    assert await count(store, ids["SaltySpittoon"]) == 3
    dm = types.SimpleNamespace(**{**vars(chat[1]), "guild": None, "id": 1})
    await h.on_message(dm)
    assert await store.get_message("1") is None
