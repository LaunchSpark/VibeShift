from datetime import datetime
from zoneinfo import ZoneInfo

import discord


def priority_key(snowflake: int) -> str:
    """Zero-padded to 20 digits so lexical order equals numeric order."""
    return f"{snowflake:020d}"


def edit_priority_key(edited_at: datetime) -> str:
    return priority_key(discord.utils.time_snowflake(edited_at, high=True))


def day_bucket(created_at: datetime, tz: ZoneInfo) -> str:
    return created_at.astimezone(tz).date().isoformat()
