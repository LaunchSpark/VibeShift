"""VibeShift Discord bot: captures messages into PocketBase, scores them, runs history jobs."""

import asyncio
import logging
import os
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import discord
from discord.ext import commands
from dotenv import load_dotenv

from vibeshift.config import Settings
from vibeshift.emotions import load_model, predict_emotion, score_batch
from vibeshift.history import (create_downtime_jobs, ensure_downtime_job, job_loop,
                               to_new_message)
from vibeshift.scorer import Scorer
from vibeshift.store import Store

log = logging.getLogger("vibeshift")


class Handlers:
    """Event logic, free of any Discord connection so it can be driven by tests."""

    def __init__(self, store: Store, scorer: Scorer, tz: ZoneInfo, retry_delay: float = 1.0):
        self.store, self.scorer, self.tz = store, scorer, tz
        self.retry_delay = retry_delay
        # Guilds with a pending/running downtime job; that job fetches up to "now", so live
        # capture must not move their newest cursor.
        self.outstanding_downtime: set[str] = set()
        # True until the startup downtime-job check completed: live messages arriving before
        # then must not advance cursors either (the set isn't filled yet).
        self.startup_pending = True
        # Later downtime checks in flight (re-IDENTIFY, guild join/available, failed capture).
        # Each is counted *synchronously* when its trigger fires, so no live message can
        # advance a cursor between the possible gap and the job that covers it.
        self._all_checks = 0
        self._guild_checks: dict[str, int] = {}
        self._failure_checks: set[str] = set()  # guilds with a failure job not yet queued
        self._ready_seen = False
        self.checks: set[asyncio.Task] = set()

    async def retry(self, what: str, fn):
        delay = self.retry_delay
        while True:
            try:
                return await fn()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001 - PocketBase may be down
                log.exception("%s failed; retrying in %.1fs", what, delay)
                await asyncio.sleep(delay)
                delay = min(delay * 2, 60)

    def may_advance(self, guild_id: str) -> bool:
        return (not self.startup_pending and self._all_checks == 0
                and not self._guild_checks.get(guild_id)
                and guild_id not in self.outstanding_downtime)

    def _check(self, gid: str | None, make) -> asyncio.Task:
        """Block cursor advancement (now, synchronously) until `make()` has queued the job."""
        if gid is None:
            self._all_checks += 1
        else:
            self._guild_checks[gid] = self._guild_checks.get(gid, 0) + 1

        async def run():
            try:
                await self.retry("downtime job check", make)
            finally:
                if gid is None:
                    self._all_checks -= 1
                else:
                    self._guild_checks[gid] -= 1
                    if not self._guild_checks[gid]:
                        del self._guild_checks[gid]

        task = asyncio.create_task(run())
        self.checks.add(task)
        task.add_done_callback(self.checks.discard)
        return task

    def _check_guilds(self, guilds, gid: str | None) -> asyncio.Task:
        guilds = list(guilds)
        return self._check(gid, lambda: create_downtime_jobs(
            None, self.store, self.outstanding_downtime, guilds=guilds))

    def on_ready(self, guilds) -> asyncio.Task | None:
        """Call synchronously from every on_ready. The first one is covered by startup; a
        later one means a fresh IDENTIFY after the session was lost, i.e. a possible gap."""
        if not self._ready_seen:
            self._ready_seen = True
            return None
        return self._check_guilds(guilds, None)

    def on_guild_join(self, guild) -> asyncio.Task:
        """A re-added guild's old cursors must not jump past the time it was gone."""
        return self._check_guilds([guild], str(guild.id))

    def on_guild_available(self, guild) -> asyncio.Task:
        return self._check_guilds([guild], str(guild.id))

    def _on_capture_failed(self, gid: str) -> None:
        if gid in self._failure_checks:
            return  # a job queued after this failure is already on its way and covers it
        self._failure_checks.add(gid)

        async def make():
            await ensure_downtime_job(self.store, gid, self.outstanding_downtime)
            self._failure_checks.discard(gid)
        self._check(gid, make)

    async def sync_guild(self, guild) -> None:
        gid = str(guild.id)
        await self.store.upsert_guild(gid, guild.name,
                                      str(guild.icon.url) if getattr(guild, "icon", None) else "")
        for ch in guild.text_channels:
            await self.store.upsert_channel(str(ch.id), gid, ch.name, "text")
        for m in getattr(guild, "members", []):
            if not m.bot:
                await self.store.upsert_member(gid, str(m.id), m.display_name,
                                               str(m.display_avatar.url))

    async def on_channel_create(self, channel) -> None:
        if isinstance(channel, discord.TextChannel):
            await self.store.upsert_channel(
                str(channel.id), str(channel.guild.id), channel.name, "text")

    async def on_message(self, message) -> None:
        if getattr(message, "guild", None) is None:
            return  # DM
        # Only guild text channels: history jobs walk guild.text_channels, so a thread or
        # voice-chat cursor would never be caught up and would leave gaps.
        if isinstance(message.channel, (discord.Thread, discord.VoiceChannel,
                                        discord.StageChannel)):
            return
        nm = to_new_message(message, self.tz)
        if nm is None:
            return
        try:
            await self._capture(nm, message)
        except Exception:
            # This message may be lost; a downtime job from the current cursor refetches it
            # (and the next live message must not advance past it meanwhile).
            self._on_capture_failed(nm.guild_id)
            raise

    async def _capture(self, nm, message) -> None:
        await self.store.insert_pending(nm)
        rec = await self.store.get_channel(nm.channel_id)
        if rec is None:
            await self.store.upsert_channel(nm.channel_id, nm.guild_id, message.channel.name, "text")
            rec = {}
        mid = nm.message_id
        if not rec.get("newest_fetched_id"):
            # First sight of this channel: seed both cursors.
            await self.store.set_cursors(nm.channel_id, newest=mid, oldest=mid)
        elif self.may_advance(nm.guild_id):
            await self.store.set_cursors(nm.channel_id, newest=mid)
        await self.store.upsert_member(nm.guild_id, nm.author_id, message.author.display_name,
                                       str(message.author.display_avatar.url))
        self.scorer.wake()

    async def on_raw_message_edit(self, payload) -> None:
        if payload.guild_id is None:
            return
        data = payload.data or {}
        if "content" not in data:
            return
        if not data["content"].strip():
            await self._delete(payload.message_id)  # emptied: no longer scoreable
            return
        ts = data.get("edited_timestamp")
        edited_at = (datetime.fromisoformat(ts.replace("Z", "+00:00"))
                     if ts else datetime.now(timezone.utc))
        await self.scorer.on_edited(str(payload.message_id), data["content"], edited_at)

    async def _delete(self, message_id) -> None:
        msg = await self.store.mark_deleted(str(message_id), datetime.now(timezone.utc))
        if msg is not None:
            await self.scorer.on_deleted(msg)

    async def on_raw_message_delete(self, payload) -> None:
        if payload.guild_id is None:
            return
        await self._delete(payload.message_id)

    async def on_raw_bulk_message_delete(self, payload) -> None:
        if payload.guild_id is None:
            return
        for mid in payload.message_ids:
            await self._delete(mid)


def make_bot(settings: Settings | None = None) -> commands.Bot:
    settings = settings or Settings()
    tz = ZoneInfo(settings.tz)
    intents = discord.Intents.default()
    intents.message_content = True  # "Message Content Intent" in the dev portal
    intents.members = True  # "Server Members Intent" in the dev portal
    bot = commands.Bot(command_prefix="!", intents=intents)

    store = Store(settings.pocketbase_url)
    scorer = Scorer(store, score_batch, settings.score_chunk, tz)
    h = Handlers(store, scorer, tz, retry_delay=settings.job_poll_seconds)
    bot.vibe_handlers = h
    tasks: list[asyncio.Task] = []
    started = False  # on_ready fires on every re-IDENTIFY; start everything once

    async def startup():
        # Downtime check first (cheap, gates cursor advancement), then the slow member sync.
        await h.retry("downtime job check",
                      lambda: create_downtime_jobs(bot, store, h.outstanding_downtime))
        h.startup_pending = False
        # Start the job loop only now, so a job can't finish before its guild is flagged.
        tasks.append(asyncio.create_task(
            job_loop(bot, store, tz, settings.job_poll_seconds, h.outstanding_downtime)))
        for g in list(bot.guilds):
            await h.retry(f"sync guild {g.id}", lambda g=g: h.sync_guild(g))

    @bot.event
    async def setup_hook():
        await asyncio.to_thread(load_model)  # warm the model off the event loop

    @bot.event
    async def on_ready():
        nonlocal started
        h.on_ready(bot.guilds)  # synchronous: a later ready flags all guilds before any await
        if started:
            return
        started = True  # synchronous guard, before any await
        log.info("Logged in as %s (id=%s)", bot.user, bot.user.id)
        tasks.append(asyncio.create_task(scorer.run_forever()))
        tasks.append(asyncio.create_task(startup()))

    @bot.event
    async def on_message(message: discord.Message):
        if message.author.bot:
            return
        try:
            await h.on_message(message)
        except Exception:  # noqa: BLE001 - capture failure must not break commands
            log.exception("capture failed for message %s", message.id)
        await bot.process_commands(message)

    @bot.event
    async def on_raw_message_edit(payload):
        try:
            await h.on_raw_message_edit(payload)
        except Exception:  # noqa: BLE001
            log.exception("edit handling failed for %s", payload.message_id)

    @bot.event
    async def on_raw_message_delete(payload):
        try:
            await h.on_raw_message_delete(payload)
        except Exception:  # noqa: BLE001
            log.exception("delete handling failed for %s", payload.message_id)

    @bot.event
    async def on_raw_bulk_message_delete(payload):
        try:
            await h.on_raw_bulk_message_delete(payload)
        except Exception:  # noqa: BLE001
            log.exception("bulk delete handling failed")

    @bot.event
    async def on_guild_join(guild):
        h.on_guild_join(guild)  # flag before the first await
        await h.retry(f"sync guild {guild.id}", lambda: h.sync_guild(guild))

    @bot.event
    async def on_guild_available(guild):
        h.on_guild_available(guild)

    @bot.event
    async def on_guild_channel_create(channel):
        await h.on_channel_create(channel)

    @bot.command()
    async def vibe(ctx: commands.Context, *, text: str):
        """!vibe <text> — reply with the predicted emotion."""
        emotion = await asyncio.to_thread(predict_emotion, text)
        await ctx.reply(f"Vibe: **{emotion}**")

    return bot


def main():
    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    # httpx logs every PocketBase poll at INFO; keep only its warnings and errors.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    token = os.environ.get("DISCORD_TOKEN")
    if not token:
        raise SystemExit("DISCORD_TOKEN is not set; copy .env.example to .env and fill it in.")
    make_bot().run(token, log_handler=None)


if __name__ == "__main__":
    main()
