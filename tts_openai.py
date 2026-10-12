"""
Speech from anything that serves /v1/audio/speech — the OpenAI voice protocol.

WHO SPEAKS IT
    OpenAI itself, the subscription's own voice servers (backend/gpu/
    speech_server.py, reached through the relay by providers/service.py), and
    the local servers people run at home — Kokoro-FastAPI, openedai-speech,
    LocalAI. Same request, same answer: `response_format=pcm` is raw 24 kHz
    16-bit mono, streamed, which is exactly what the speaker is open at. Nothing
    is decoded, nothing is resampled, and the first bytes that arrive are the
    first sound.

NO LANGUAGE PARAMETER
    The protocol has none and the voices it reaches detect the language from the
    text. What language the assistant speaks is the model's and the system
    prompt's decision, never this file's.
"""
from __future__ import annotations

from importlib.util import find_spec
from typing import AsyncIterator

from .base import DEFAULT_CAPS, Field

PRESETS: dict[str, str] = {
    "OpenAI":          "https://api.openai.com/v1",
    "Kokoro-FastAPI":  "http://127.0.0.1:8880/v1",
    "LocalAI":         "http://127.0.0.1:8080/v1",
}

# The slice the speaker queue receives: 50 ms at 24 kHz, 16-bit. Small enough
# that an interruption stops the voice almost at once, and always an even
# number of bytes, so a sample is never split across two slices.
_SLICE = 2400


class OpenAISpeechTTS:
    def __init__(self, cfg: dict):
        from .llm_openai import _normalise

        self.base = _normalise(cfg.get("base_url") or "")
        self.key = (cfg.get("api_key") or "").strip()
        self.model = (cfg.get("model") or "").strip()
        self.voice = (cfg.get("voice") or "").strip()
        self.cache_phrases = str(cfg.get("cache_phrases", "1")).lower() not in ("0", "false", "no")
        # Set only by core/service.py for the subscription's relay: its pinned
        # certificate and the device headers. Never typed into the panel.
        self._verify = cfg.get("verify", True)
        self._extra = dict(cfg.get("headers") or {})
        self.caps = dict(DEFAULT_CAPS)
        self.caps["audio_out"] = True

    def _headers(self) -> dict:
        h = {**self._extra, "Content-Type": "application/json"}
        if self.key:
            h["Authorization"] = f"Bearer {self.key}"
        return h

    def _payload(self, text: str, voice: str) -> dict:
        body = {"input": text, "response_format": "pcm"}
        if self.model:
            body["model"] = self.model
        if voice:
            body["voice"] = voice
        return body

    async def speak(self, text: str, voice: str = "") -> AsyncIterator[bytes]:
        from . import _voice_cache
        from ._http import async_client
        from .llm_openai import _is_loopback, _terse

        text = (text or "").strip()
        if not text:
            return
        if not self.base:
            raise RuntimeError("no voice server address is set")
        voice = voice or self.voice

        # A line this voice has said before comes off the disk in a millisecond.
        kept = _voice_cache.path_for(text, "openai_speech", self.base, self.model, voice,
                                     enabled=self.cache_phrases)
        already = _voice_cache.read(kept)
        if already is not None:
            for i in range(0, len(already), _SLICE):
                yield already[i:i + _SLICE]
            return
        collected = bytearray() if kept is not None else None

        cl = async_client(self.base, verify=self._verify, trust_env=not _is_loopback(self.base))
        buf = b""
        async with cl.stream("POST", f"{self.base}/audio/speech", headers=self._headers(),
                             json=self._payload(text, voice)) as r:
            if r.status_code >= 400:
                detail = _terse((await r.aread()).decode("utf-8", "replace"))
                raise RuntimeError(f"{r.status_code} {detail}")
            async for piece in r.aiter_bytes():
                buf += piece
                while len(buf) >= _SLICE:
                    out, buf = buf[:_SLICE], buf[_SLICE:]
                    if collected is not None:
                        collected.extend(out)
                    yield out
        buf = buf[: len(buf) // 2 * 2]
        if buf:
            if collected is not None:
                collected.extend(buf)
            yield buf
        if collected:
            _voice_cache.write(kept, bytes(collected))

    def probe(self) -> tuple[bool, str]:
        """Say one word and report how long the first sound took."""
        import asyncio
        import time

        if not self.base:
            return False, "No address set."

        async def once() -> tuple[float, int]:
            t0, first, total = time.monotonic(), None, 0
            async for pcm in self.speak("OK.", self.voice):
                first = first if first is not None else time.monotonic() - t0
                total += len(pcm)
            return (first or 0.0), total

        try:
            saved, self.cache_phrases = self.cache_phrases, False
            try:
                first, total = asyncio.run(once())
            finally:
                self.cache_phrases = saved
        except Exception as e:
            msg = str(e)
            if "401" in msg or "403" in msg:
                return False, "The server refused the key."
            return False, msg[:120]
        if not total:
            return False, "The server answered with no audio."
        return True, f"First sound in {first:.2f}s"


def _make(cfg: dict) -> OpenAISpeechTTS:
    return OpenAISpeechTTS(cfg)


PROVIDER = {
    "id": "openai_speech",
    "kind": "tts",
    "label": "OpenAI-compatible voice",
    "hint": "OpenAI · Kokoro-FastAPI · LocalAI · any /v1/audio/speech server",
    "requires": ("httpx",),
    "catalog": (
        {"id": "openai_speech", "label": "OpenAI voice", "note": "cloud · needs a key",
         "local": False, "values": {"base_url": PRESETS["OpenAI"], "model": "gpt-4o-mini-tts",
                                    "voice": "alloy"}},
        {"id": "kokoro_fastapi", "label": "Kokoro-FastAPI", "note": "on this computer · free",
         "local": True, "values": {"base_url": PRESETS["Kokoro-FastAPI"], "model": "kokoro",
                                   "voice": "af_heart"}},
        {"id": "speech_custom", "label": "Another voice server", "note": "type the address yourself",
         "local": None, "values": {}},
    ),
    "needs": (
        Field("base_url", "Address", kind="choice", choices=tuple(PRESETS.values()),
              default=PRESETS["OpenAI"], hint="Any address serving /v1/audio/speech"),
        Field("model", "Model", default="", required=False),
        Field("voice", "Voice", default="", required=False),
        Field("api_key", "API key", kind="secret", required=False,
              hint="Local servers usually need none"),
        Field("cache_phrases", "Reuse short phrases", kind="bool", default=True,
              required=False, advanced=True,
              hint="Keeps short confirmations on disk instead of making them twice"),
    ),
    "caps": {"audio_out": True},
    "factory": _make,
}


def available() -> bool:
    return find_spec("httpx") is not None
