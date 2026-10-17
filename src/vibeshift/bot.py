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
from vibeshift.history import create_downtime_jobs, job_loop, to_new_message
from vibeshift.scorer import Scorer
from vibeshift.store import Store

log = logging.getLogger("vibeshift")


class Handlers:
    """Event logic, free of any Discord connection so it can be driven by tests."""

    def __init__(self, store: Store, scorer: Scorer, tz: ZoneInfo):
        self.store, self.scorer, self.tz = store, scorer, tz
        # Guilds with a pending/running downtime job; that job fetches up to "now", so live
        # capture must not move their newest cursor.
        self.outstanding_downtime: set[str] = set()

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
        nm = to_new_message(message, self.tz)
        if nm is None:
            return
        await self.store.insert_pending(nm)
        rec = await self.store.get_channel(nm.channel_id)
        if rec is None:
            await self.store.upsert_channel(nm.channel_id, nm.guild_id, message.channel.name, "text")
            rec = {}
        mid = nm.message_id
        if not rec.get("newest_fetched_id"):
            # First sight of this channel: seed both cursors.
            await self.store.set_cursors(nm.channel_id, newest=mid, oldest=mid)
        elif nm.guild_id not in self.outstanding_downtime:
            await self.store.set_cursors(nm.channel_id, newest=mid)
        await self.store.upsert_member(nm.guild_id, nm.author_id, message.author.display_name,
                                       str(message.author.display_avatar.url))
        self.scorer.wake()

    async def on_raw_message_edit(self, payload) -> None:
        data = payload.data or {}
        if "content" not in data:
            return
        ts = data.get("edited_timestamp")
        edited_at = (datetime.fromisoformat(ts.replace("Z", "+00:00"))
                     if ts else datetime.now(timezone.utc))
        await self.scorer.on_edited(str(payload.message_id), data["content"], edited_at)

    async def on_raw_message_delete(self, payload) -> None:
        msg = await self.store.mark_deleted(str(payload.message_id), datetime.now(timezone.utc))
        if msg is not None:
            await self.scorer.on_deleted(msg)


def make_bot(settings: Settings | None = None) -> commands.Bot:
    settings = settings or Settings()
    tz = ZoneInfo(settings.tz)
    intents = discord.Intents.default()
    intents.message_content = True  # "Message Content Intent" in the dev portal
    intents.members = True  # "Server Members Intent" in the dev portal
    bot = commands.Bot(command_prefix="!", intents=intents)

    store = Store(settings.pocketbase_url)
    scorer = Scorer(store, score_batch, settings.score_chunk, tz)
    h = Handlers(store, scorer, tz)
    bot.vibe_handlers = h
    tasks: list[asyncio.Task] = []  # started once; on_ready can fire on every reconnect

    @bot.event
    async def setup_hook():
        await asyncio.to_thread(load_model)  # warm the model off the event loop

    @bot.event
    async def on_ready():
        log.info("Logged in as %s (id=%s)", bot.user, bot.user.id)
        for g in bot.guilds:
            await h.sync_guild(g)
        if tasks:
            return
        await create_downtime_jobs(bot, store, h.outstanding_downtime)
        tasks.append(asyncio.create_task(scorer.run_forever()))
        tasks.append(asyncio.create_task(
            job_loop(bot, store, tz, settings.job_poll_seconds, h.outstanding_downtime)))

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
    async def on_guild_join(guild):
        await h.sync_guild(guild)

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
    token = os.environ.get("DISCORD_TOKEN")
    if not token:
        raise SystemExit("DISCORD_TOKEN is not set; copy .env.example to .env and fill it in.")
    make_bot().run(token, log_handler=None)


if __name__ == "__main__":
    main()
