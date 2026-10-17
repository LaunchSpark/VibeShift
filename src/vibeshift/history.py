"""History jobs: downtime catch-up (after newest) and depth extension (before oldest)."""
import asyncio
import logging
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import discord

from vibeshift.keys import day_bucket
from vibeshift.store import NewMessage, Store, _from_pb, _snow_gt

log = logging.getLogger(__name__)

SCAN_EVERY = 100


def to_new_message(m: discord.Message, tz: ZoneInfo) -> NewMessage | None:
    """Shared with live capture. None for bot authors and empty/whitespace content."""
    if m.author.bot or not (m.content or "").strip():
        return None
    created = m.created_at
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)
    created = created.astimezone(timezone.utc)
    ref = getattr(m, "reference", None)
    reply = getattr(ref, "message_id", None) if ref else None
    return NewMessage(
        message_id=str(m.id), guild_id=str(m.guild.id), channel_id=str(m.channel.id),
        author_id=str(m.author.id), created_at=created, content=m.content,
        day=day_bucket(created, tz), reply_to_id=str(reply) if reply else "")


async def _ingest_channel(job, channel, store: Store, tz, members: set, progress) -> None:
    origin = job["origin"]
    target = _from_pb(job.get("target"))
    cid = str(channel.id)
    rec = await store.get_channel(cid)
    if origin == "downtime":
        newest = (rec or {}).get("newest_fetched_id") or ""
        if not newest:
            return
        it = channel.history(limit=None, after=discord.Object(id=int(newest)), oldest_first=True)
    else:
        oldest = (rec or {}).get("oldest_fetched_id") or ""
        before = discord.Object(id=int(oldest)) if oldest else None
        # oldest_first=False is explicit: discord.py flips to oldest-first when `after` is set.
        it = channel.history(limit=None, before=before, after=target, oldest_first=False)

    hi = lo = None  # extremes seen, bot/empty messages included, to keep the range contiguous
    try:
        async for m in it:
            if origin == "extend" and target and m.created_at <= target:
                break  # discord.py only filters by `after`; stop paging past the target
            nm = to_new_message(m, tz)
            if nm:
                await store.insert_pending(nm)
                if nm.author_id not in members:
                    members.add(nm.author_id)
                    await store.upsert_member(
                        nm.guild_id, nm.author_id, m.author.display_name,
                        str(m.author.display_avatar.url))
            # Cursors move only after the message was handled, so a failure leaves no hole.
            mid = str(m.id)
            hi = mid if hi is None or _snow_gt(mid, hi) else hi
            lo = mid if lo is None or _snow_gt(lo, mid) else lo
            await progress()
    finally:
        if hi is not None:
            if origin == "downtime":
                await store.set_cursors(cid, newest=hi)
            else:
                no_newest = not (rec or {}).get("newest_fetched_id")
                await store.set_cursors(cid, oldest=lo, newest=hi if no_newest else None)


async def run_job(job: dict, guild: discord.Guild, store: Store, tz: ZoneInfo) -> str:
    jid = job["id"]
    await store.update_job(jid, status="running")
    scanned = 0
    notes: list[str] = []
    members: set[str] = set()

    async def progress():
        nonlocal scanned
        scanned += 1
        if scanned % SCAN_EVERY == 0:
            await store.update_job(jid, scanned=scanned)

    try:
        await store.upsert_guild(str(guild.id), guild.name,
                                 str(guild.icon.url) if getattr(guild, "icon", None) else "")
        for ch in guild.text_channels:
            await store.upsert_channel(str(ch.id), str(guild.id), ch.name, "text")
            try:
                await _ingest_channel(job, ch, store, tz, members, progress)
            except discord.Forbidden:
                notes.append(f"channel {ch.id}: forbidden, skipped")
        if job["origin"] == "extend" and job.get("target"):
            await store.set_history_depth(str(guild.id), _from_pb(job["target"]))
        await store.update_job(jid, status="done", scanned=scanned, error="; ".join(notes))
        return "done"
    except Exception as e:  # noqa: BLE001 - job failure is recorded, not raised
        log.exception("history job %s failed", jid)
        await store.update_job(jid, status="failed", scanned=scanned,
                               error="; ".join(notes + [f"{type(e).__name__}: {e}"]))
        return "failed"


async def job_loop(bot, store: Store, tz: ZoneInfo, poll_seconds: float,
                   outstanding_downtime: set[str] | None = None) -> None:
    """Run pending jobs forever. `outstanding_downtime` holds guild ids whose downtime job
    is pending/running; a guild is dropped once its job ends (done, failed, or unrunnable)."""
    delay = poll_seconds
    while True:  # resume jobs interrupted by a crash; retry while PocketBase is down
        try:
            await store.reset_running_jobs()
            break
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            log.exception("resetting running jobs failed; retrying in %.1fs", delay)
            await asyncio.sleep(delay)
            delay = min(delay * 2, 60)
    fails: dict[str, int] = {}  # consecutive downtime failures per guild, for backoff
    while True:
        try:
            job = await store.next_pending_job()
            if job:
                gid, downtime = job["guild_id"], job["origin"] == "downtime"
                guild = bot.get_guild(int(gid))
                if guild is None:
                    status = "failed"
                    if not downtime:
                        await store.update_job(job["id"], status="failed", error="guild not available")
                    else:
                        await store.update_job(job["id"], status="pending", error="guild not available")
                else:
                    status = await run_job(job, guild, store, tz)
                    if status != "done" and downtime:
                        await store.update_job(job["id"], status="pending")  # retry, keep the flag
                if downtime:
                    if status == "done":
                        fails.pop(gid, None)
                        if outstanding_downtime is not None:
                            outstanding_downtime.discard(gid)  # only a finished job covers the gap
                    else:
                        fails[gid] = fails.get(gid, 0) + 1
                        await asyncio.sleep(min(poll_seconds * 2 ** fails[gid], 300))
                continue
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            log.exception("job loop error")
        await asyncio.sleep(poll_seconds)


async def create_downtime_jobs(bot, store: Store,
                               outstanding: set[str] | None = None) -> None:
    """One downtime job per guild with captured history, unless one is already queued/running.
    Guilds with such a job are added to `outstanding` (live capture then leaves cursors alone)."""
    for g in bot.guilds:
        gid = str(g.id)
        if await store.has_active_job(gid, "downtime"):
            if outstanding is not None:
                outstanding.add(gid)
            continue
        for ch in g.text_channels:
            rec = await store.get_channel(str(ch.id))
            if rec and rec.get("newest_fetched_id"):
                await store.create_job(gid, "downtime")
                if outstanding is not None:
                    outstanding.add(gid)
                break
