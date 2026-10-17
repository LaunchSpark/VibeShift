from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import discord

from vibeshift.keys import day_bucket, edit_priority_key, priority_key

UTC = timezone.utc


def test_priority_key_pads_to_20_digits():
    assert priority_key(5) == "00000000000000000005"


def test_priority_key_orders_lexically():
    assert priority_key(2**63 - 1) > priority_key(2**62)
    assert priority_key(9) < priority_key(10)


def test_edit_key_after_earlier_messages():
    t = datetime(2026, 10, 17, 12, 0, tzinfo=UTC)
    created = discord.utils.time_snowflake(t - timedelta(seconds=1), high=True)
    assert edit_priority_key(t) > priority_key(created)
    assert len(edit_priority_key(t)) == 20


def test_day_bucket_uses_timezone():
    t = datetime(2026, 10, 17, 3, 0, tzinfo=UTC)
    assert day_bucket(t, ZoneInfo("America/Los_Angeles")) == "2026-10-16"
    assert day_bucket(t, ZoneInfo("UTC")) == "2026-10-17"
