import os
from dataclasses import dataclass, field


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass(frozen=True)
class Settings:
    pocketbase_url: str = field(default_factory=lambda: _env("POCKETBASE_URL", "http://127.0.0.1:8090"))
    tz: str = field(default_factory=lambda: _env("VIBESHIFT_TZ", "UTC"))
    score_chunk: int = field(default_factory=lambda: int(_env("SCORE_CHUNK", "8")))
    job_poll_seconds: float = field(default_factory=lambda: float(_env("JOB_POLL_SECONDS", "5")))
