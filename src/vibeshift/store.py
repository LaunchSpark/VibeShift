"""PocketBase client. The `messages` collection is the pending-work deque."""
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import httpx

from vibeshift.keys import edit_priority_key, priority_key


def _q(value: str) -> str:
    """Quote a value for a PocketBase filter expression."""
    return '"' + str(value).replace("\\", "\\\\").replace('"', '\\"') + '"'


def _to_pb(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] + "Z"


def _from_pb(s: str | None) -> datetime | None:
    if not s:
        return None
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)


def _snow_gt(a: str, b: str) -> bool:
    return (len(a), a) > (len(b), b)


@dataclass
class NewMessage:
    message_id: str
    guild_id: str
    channel_id: str
    author_id: str
    created_at: datetime
    content: str
    day: str
    reply_to_id: str = ""
    priority_key: str | None = None  # defaults to the snowflake of message_id


@dataclass
class StoredMessage:
    id: str  # PocketBase record id
    message_id: str
    guild_id: str
    channel_id: str
    author_id: str
    created_at: datetime | None
    edited_at: datetime | None
    deleted_at: datetime | None
    reply_to_id: str
    content: str
    day: str
    priority_key: str
    score_status: str
    emotion: str
    probabilities: dict | None
    confidence: float | None
    model_id: str
    rollup_applied: bool

    @classmethod
    def from_record(cls, r: dict[str, Any]) -> "StoredMessage":
        probs = r.get("probabilities")
        return cls(
            id=r["id"], message_id=r["message_id"], guild_id=r["guild_id"],
            channel_id=r["channel_id"], author_id=r["author_id"],
            created_at=_from_pb(r.get("created_at")), edited_at=_from_pb(r.get("edited_at")),
            deleted_at=_from_pb(r.get("deleted_at")),
            reply_to_id=r.get("reply_to_id") or "", content=r.get("content") or "",
            day=r.get("day") or "", priority_key=r.get("priority_key") or "",
            score_status=r.get("score_status") or "", emotion=r.get("emotion") or "",
            probabilities=probs if probs else None,
            confidence=r.get("confidence") if r.get("confidence") is not None else None,
            model_id=r.get("model_id") or "", rollup_applied=bool(r.get("rollup_applied")),
        )


class Store:
    def __init__(self, base_url: str):
        self._http = httpx.AsyncClient(base_url=base_url.rstrip("/"), timeout=30)

    async def aclose(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> "Store":
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()

    # -- low-level helpers -------------------------------------------------
    async def _list(self, coll: str, filter: str = "", sort: str = "", per_page: int = 1) -> list[dict]:
        params = {"perPage": per_page, "page": 1, "skipTotal": 1}
        if filter:
            params["filter"] = filter
        if sort:
            params["sort"] = sort
        r = await self._http.get(f"/api/collections/{coll}/records", params=params)
        r.raise_for_status()
        return r.json()["items"]

    async def _first(self, coll: str, filter: str) -> dict | None:
        items = await self._list(coll, filter)
        return items[0] if items else None

    async def _create(self, coll: str, body: dict) -> dict:
        r = await self._http.post(f"/api/collections/{coll}/records", json=body)
        r.raise_for_status()
        return r.json()

    async def _patch(self, coll: str, rec_id: str, body: dict) -> dict:
        r = await self._http.patch(f"/api/collections/{coll}/records/{rec_id}", json=body)
        r.raise_for_status()
        return r.json()

    async def _upsert(self, coll: str, filter: str, body: dict) -> dict:
        existing = await self._first(coll, filter)
        if existing:
            return await self._patch(coll, existing["id"], body)
        return await self._create(coll, body)

    # -- reference data ----------------------------------------------------
    async def upsert_guild(self, guild_id: str, name: str, icon_url: str) -> None:
        await self._upsert("guilds", f"guild_id={_q(guild_id)}",
                           {"guild_id": guild_id, "name": name, "icon_url": icon_url or ""})

    async def set_history_depth(self, guild_id: str, depth: datetime) -> None:
        rec = await self._first("guilds", f"guild_id={_q(guild_id)}")
        if rec:
            await self._patch("guilds", rec["id"], {"history_depth": _to_pb(depth)})

    async def upsert_channel(self, channel_id: str, guild_id: str, name: str, type: str) -> None:
        await self._upsert("channels", f"channel_id={_q(channel_id)}",
                           {"channel_id": channel_id, "guild_id": guild_id, "name": name, "type": type})

    async def upsert_member(self, guild_id: str, author_id: str, display_name: str,
                            avatar_url: str) -> None:
        await self._upsert(
            "members", f"guild_id={_q(guild_id)} && author_id={_q(author_id)}",
            {"guild_id": guild_id, "author_id": author_id, "display_name": display_name,
             "avatar_url": avatar_url or ""})

    # -- message deque -----------------------------------------------------
    async def _message(self, message_id: str) -> dict | None:
        return await self._first("messages", f"message_id={_q(message_id)}")

    async def insert_pending(self, msg: NewMessage) -> bool:
        if await self._message(msg.message_id):
            return False
        key = msg.priority_key or priority_key(int(msg.message_id))
        r = await self._http.post("/api/collections/messages/records", json={
            "message_id": msg.message_id, "guild_id": msg.guild_id,
            "channel_id": msg.channel_id, "author_id": msg.author_id,
            "created_at": _to_pb(msg.created_at), "reply_to_id": msg.reply_to_id or "",
            "content": msg.content, "day": msg.day, "priority_key": key,
            "score_status": "pending", "rollup_applied": False,
        })
        if r.status_code == 400 and await self._message(msg.message_id):
            return False  # lost a race against the unique index
        r.raise_for_status()
        return True

    async def mark_edited(self, message_id: str, content: str, edited_at: datetime) -> None:
        rec = await self._message(message_id)
        if not rec:
            return
        # probabilities / rollup_applied are deliberately untouched: the scorer
        # subtracts the old contribution before applying the new one.
        await self._patch("messages", rec["id"], {
            "content": content, "edited_at": _to_pb(edited_at),
            "priority_key": edit_priority_key(edited_at), "score_status": "pending",
        })

    async def mark_deleted(self, message_id: str, deleted_at: datetime) -> StoredMessage | None:
        rec = await self._message(message_id)
        if not rec:
            return None
        return StoredMessage.from_record(
            await self._patch("messages", rec["id"], {"deleted_at": _to_pb(deleted_at)}))

    async def pop_chunk(self, n: int) -> list[StoredMessage]:
        items = await self._list(
            "messages", 'score_status="pending" && deleted_at=""', "-priority_key", per_page=n)
        return [StoredMessage.from_record(r) for r in items]

    async def get_message(self, message_id: str) -> StoredMessage | None:
        rec = await self._message(message_id)
        return StoredMessage.from_record(rec) if rec else None

    async def save_score(self, message_id: str, emotion: str, probabilities: dict,
                         confidence: float, model_id: str,
                         rollup_applied: bool | None = None) -> None:
        rec = await self._message(message_id)
        if rec:
            body = {"emotion": emotion, "probabilities": probabilities, "confidence": confidence,
                    "model_id": model_id, "score_status": "scored"}
            if rollup_applied is not None:
                body["rollup_applied"] = rollup_applied
            await self._patch("messages", rec["id"], body)

    async def mark_failed(self, message_id: str) -> None:
        rec = await self._message(message_id)
        if rec:
            await self._patch("messages", rec["id"], {"score_status": "failed"})

    # -- rollups -----------------------------------------------------------
    @staticmethod
    def _rollup_filter(guild_id, channel_id, author_id, day) -> str:
        return (f"guild_id={_q(guild_id)} && channel_id={_q(channel_id)} && "
                f"author_id={_q(author_id)} && day={_q(day)}")

    async def get_rollup(self, guild_id: str, channel_id: str, author_id: str,
                         day: str) -> tuple[int, dict] | None:
        rec = await self._first("daily_rollups", self._rollup_filter(guild_id, channel_id, author_id, day))
        if not rec:
            return None
        return int(rec["count"] or 0), (rec["prob_sums"] or {})

    async def put_rollup(self, guild_id: str, channel_id: str, author_id: str, day: str,
                         count: int, prob_sums: dict) -> None:
        await self._upsert(
            "daily_rollups", self._rollup_filter(guild_id, channel_id, author_id, day),
            {"guild_id": guild_id, "channel_id": channel_id, "author_id": author_id,
             "day": day, "count": count, "prob_sums": prob_sums})

    # -- channel cursors ---------------------------------------------------
    async def get_channel(self, channel_id: str) -> dict | None:
        return await self._first("channels", f"channel_id={_q(channel_id)}")

    async def set_cursors(self, channel_id: str, newest: str | None = None,
                          oldest: str | None = None) -> None:
        rec = await self.get_channel(channel_id)
        if not rec:
            return
        body = {}
        cur = rec.get("newest_fetched_id") or ""
        if newest is not None and (not cur or _snow_gt(newest, cur)):
            body["newest_fetched_id"] = newest
        cur = rec.get("oldest_fetched_id") or ""
        if oldest is not None and (not cur or _snow_gt(cur, oldest)):
            body["oldest_fetched_id"] = oldest
        if body:
            await self._patch("channels", rec["id"], body)

    # -- backfill jobs -----------------------------------------------------
    async def create_job(self, guild_id: str, origin: str, target: datetime | None = None) -> str:
        body = {"guild_id": guild_id, "origin": origin, "status": "pending", "scanned": 0}
        if target is not None:
            body["target"] = _to_pb(target)
        return (await self._create("backfill_jobs", body))["id"]

    async def next_pending_job(self) -> dict | None:
        items = await self._list("backfill_jobs", 'status="pending"', "created")
        return items[0] if items else None

    async def update_job(self, id: str, **fields: Any) -> None:
        for k, v in fields.items():
            if isinstance(v, datetime):
                fields[k] = _to_pb(v)
        await self._patch("backfill_jobs", id, fields)
