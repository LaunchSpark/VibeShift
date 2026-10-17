import httpx
import pytest

COLLECTIONS = ["guilds", "channels", "members", "messages", "daily_rollups", "backfill_jobs"]


@pytest.mark.parametrize("name", COLLECTIONS)
def test_collections_exist(pb_url, name):
    r = httpx.get(f"{pb_url}/api/collections/{name}/records", params={"perPage": 1})
    assert r.status_code == 200


def test_snowflake_roundtrip(pb_url):
    sid = "1234567890123456789"
    r = httpx.post(f"{pb_url}/api/collections/messages/records", json={"message_id": sid})
    assert r.status_code == 200, r.text
    got = httpx.get(f"{pb_url}/api/collections/messages/records/{r.json()['id']}").json()
    assert got["message_id"] == sid
