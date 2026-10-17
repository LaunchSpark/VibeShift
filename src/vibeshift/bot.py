"""VibeShift Discord bot: tags each message with its predicted emotion."""

import asyncio
import logging
import os

import discord
from discord.ext import commands
from dotenv import load_dotenv

from vibeshift.emotions import load_model, predict_emotion

log = logging.getLogger("vibeshift")

intents = discord.Intents.default()
intents.message_content = True  # also enable "Message Content Intent" in the dev portal
bot = commands.Bot(command_prefix="!", intents=intents)


@bot.event
async def on_ready():
    log.info("Logged in as %s (id=%s)", bot.user, bot.user.id)


@bot.event
async def on_message(message: discord.Message):
    if message.author.bot:
        return
    if message.content and not message.content.startswith(bot.command_prefix):
        # Model inference is blocking; keep it off the event loop.
        emotion = await asyncio.to_thread(predict_emotion, message.content)
        log.info("[#%s] %s: %s -> %s", message.channel, message.author, message.content, emotion)
    await bot.process_commands(message)


@bot.command()
async def vibe(ctx: commands.Context, *, text: str):
    """!vibe <text> — reply with the predicted emotion."""
    emotion = await asyncio.to_thread(predict_emotion, text)
    await ctx.reply(f"Vibe: **{emotion}**")


def main():
    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    token = os.environ.get("DISCORD_TOKEN")
    if not token:
        raise SystemExit("DISCORD_TOKEN is not set; copy .env.example to .env and fill it in.")
    load_model()  # warm the model before connecting
    bot.run(token, log_handler=None)


if __name__ == "__main__":
    main()
