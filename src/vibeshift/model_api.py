"""Loopback-only HTTP access to the bot's production emotion scoring path."""

import asyncio
import logging

from aiohttp import web

from vibeshift.emotions import MODEL_ID, score_batch

log = logging.getLogger("vibeshift.model_api")
MAX_MESSAGE_LENGTH = 2000


class ModelScoringApi:
    def __init__(self, port: int = 8765):
        self.port = port
        self.runner: web.AppRunner | None = None

    async def start(self) -> None:
        app = web.Application()
        app.router.add_get("/api/model-health", self.health)
        app.router.add_post("/api/model-score", self.score)
        self.runner = web.AppRunner(app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, "127.0.0.1", self.port)
        try:
            await site.start()
        except Exception:
            await self.runner.cleanup()
            self.runner = None
            raise
        log.info("Local model scoring API listening on http://127.0.0.1:%d", self.port)

    async def close(self) -> None:
        if self.runner is not None:
            await self.runner.cleanup()
            self.runner = None

    async def health(self, _request: web.Request) -> web.Response:
        return web.json_response({"ready": True, "model_id": MODEL_ID})

    async def score(self, request: web.Request) -> web.Response:
        try:
            payload = await request.json()
        except (ValueError, web.HTTPException):
            return web.json_response({"error": "Request body must be valid JSON."}, status=400)
        message = payload.get("message") if isinstance(payload, dict) else None
        if not isinstance(message, str) or not message.strip():
            return web.json_response({"error": "Enter a message to score."}, status=400)
        if len(message) > MAX_MESSAGE_LENGTH:
            return web.json_response({"error": f"Messages must be {MAX_MESSAGE_LENGTH} characters or fewer."}, status=400)
        try:
            # Use the same target marker, W=5 prompt, checkpoint and score_batch as live scoring.
            scored = await asyncio.to_thread(score_batch, [f"TARGET MESSAGE: {message.strip()}"])
            result = scored[0]
            return web.json_response({
                "emotion": result.emotion,
                "confidence": result.confidence,
                "probabilities": result.probabilities,
                "model_id": MODEL_ID,
            })
        except Exception as exc:  # noqa: BLE001 - return an actionable API error
            log.exception("Model scoring request failed")
            return web.json_response({"error": f"Model scoring failed: {exc}"}, status=503)
