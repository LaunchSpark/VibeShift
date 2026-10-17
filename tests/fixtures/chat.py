"""Parse sample_chat.txt into fake discord messages."""
import re
import types
from datetime import datetime, timezone
from pathlib import Path

import discord

SAMPLE = Path(__file__).parent / "sample_chat.txt"
LINE = re.compile(r"^\[(\d\d):(\d\d)\] <([^>]+)> (.*)$")
GUILD = types.SimpleNamespace(id=5000, name="Test Guild", icon=None)
CHANNEL = types.SimpleNamespace(id=7000, name="general", guild=GUILD)
BOT_NAMES = {"bot_alert"}


def author_ids(lines: list[str]) -> dict[str, str]:
    names = list(dict.fromkeys(LINE.match(l).group(3) for l in lines))
    return {n: str(1000 + i) for i, n in enumerate(names)}


def parse_chat(path: Path = SAMPLE) -> list[types.SimpleNamespace]:
    lines = [l for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    ids = author_ids(lines)
    out = []
    for i, line in enumerate(lines):
        hh, mm, name, text = LINE.match(line).groups()
        created = datetime(2026, 10, 17, int(hh), int(mm), tzinfo=timezone.utc)
        author = types.SimpleNamespace(
            id=int(ids[name]), name=name, bot=name in BOT_NAMES, display_name=name,
            display_avatar=types.SimpleNamespace(url=f"http://a/{name}.png"))
        out.append(types.SimpleNamespace(
            id=discord.utils.time_snowflake(created) + i, content=text, author=author,
            created_at=created, guild=GUILD, channel=CHANNEL, reference=None))
    return out
