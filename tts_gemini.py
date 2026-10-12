"""Gemini's own voices, for the lane that is not Live.

WHY THIS IS THE ONE THAT MATTERS FOR COST
    In the cheap lane the bill breaks down, measured, roughly like this for one
    short exchange: recognition $0.0002, the model with a cached prefix
    $0.0009, the spoken answer $0.0015. The VOICE is the expensive half. Two
    things follow, and both are implemented here.

    First, the cheaper of the two hosted voices is the default: 2.5 Flash TTS
    charges $10 per million audio tokens against $20 for the 3.1 preview, and
    on a one-sentence confirmation nobody can hear the difference.

    Second, and this is the bigger one: short confirmations repeat. "Tamam",
    "Açtım", "Gönderildi" — the same handful of sentences, all day, every day,
    for every user. Synthesising them again each time is paying for the same
    audio over and over, so a small on-disk cache keeps them. It is keyed by
    voice and text, it holds only clips short enough to be confirmations, and it
    is capped so it cannot grow into a problem. The first "tamam" of a user's
    life costs a fraction of a cent; every one after it is free.

WHY 24 kHz WITHOUT RESAMPLING
    `_play_audio` in main.py opens the speaker at 24 kHz, and Gemini's TTS
    returns 24 kHz signed 16-bit mono PCM. The two numbers already agree, which
    is the only reason this file has no audio maths in it.
"""
from __future__ import annotations

import asyncio
import time
from typing import AsyncIterator

from .base import DEFAULT_CAPS, Field

_DEFAULT_MODEL = "gemini-2.5-flash-preview-tts"
_MODELS = ("gemini-2.5-flash-preview-tts", "gemini-3.1-flash-tts-preview",
           "gemini-2.5-pro-preview-tts")

# Gemini's prebuilt voices. Names rather than a fetched list: the set is small,
# stable, and a picker that has to make a network call to draw itself is the
# mistake this project already fixed once in the settings panel.
_VOICES = ("Zephyr", "Puck", "Charon", "Kore", "Fenrir", "Leda",
           "Orus", "Aoede", "Callirrhoe", "Autonoe", "Enceladus", "Iapetus")
_DEFAULT_VOICE = "Charon"

_CHUNK = 4096

# An empty answer happens; see _synthesise. Three tries covers it without
# turning a real outage into a long wait.
_RETRIES = 3


class GeminiTTS:
    def __init__(self, cfg: dict):
        self.model = (cfg.get("model") or _DEFAULT_MODEL).strip()
        self.voice = (cfg.get("voice") or _DEFAULT_VOICE).strip()
        self.caps = dict(DEFAULT_CAPS)
        self.caps["audio_out"] = True
        self._key = (cfg.get("api_key") or "").strip()
        self._reuse = str(cfg.get("cache_phrases", "1")).lower() not in ("0", "false", "no")

    # -- the phrase cache ----------------------------------------------------
    # Shared with the other hosted voice: see providers/_voice_cache.py. Both
    # wanted it, neither owns it, and one copy each would be two caches with a
    # bug fixed in one of them.
    def _cache_path(self, text: str):
        from . import _voice_cache
        return _voice_cache.path_for(text, "gemini", self.model, self.voice,
                                     enabled=self._reuse)

    # -- synthesis -----------------------------------------------------------
    def _once(self, text: str, voice: str) -> bytes:
        from google.genai import types as gtypes
        from core import gemini
        client = gemini.client(key=self._key or gemini.api_key())
        response = client.models.generate_content(
            model=self.model,
            contents=text,
            config=gtypes.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=gtypes.SpeechConfig(
                    voice_config=gtypes.VoiceConfig(
                        prebuilt_voice_config=gtypes.PrebuiltVoiceConfig(
                            voice_name=voice))),
            ),
        )
        for candidate in (getattr(response, "candidates", None) or ()):
            for part in (getattr(getattr(candidate, "content", None), "parts", None) or ()):
                blob = getattr(part, "inline_data", None)
                if blob is not None and getattr(blob, "data", None):
                    return bytes(blob.data)
        return b""

    def _synthesise(self, text: str, voice: str) -> bytes:
        """The clip, or an exception. Retries an EMPTY answer.

        Measured while building this: the same request, unchanged, sometimes
        comes back with `finish_reason=OTHER` and no audio at all — not an
        error, not a timeout, just nothing. Roughly one attempt in six here.
        The assistant going silently mute is the worst failure this lane can
        have, and it costs nothing to ask again: an empty answer has no audio
        tokens to be billed for.
        """
        last = ""
        for attempt in range(_RETRIES):
            try:
                pcm = self._once(text, voice)
                if pcm:
                    return pcm
                last = "the voice returned no audio"
            except Exception as exc:
                last = f"{type(exc).__name__}: {exc}"
            time.sleep(0.25 * (attempt + 1))
        raise RuntimeError(last or "the voice returned no audio")

    async def speak(self, text: str, voice: str = "") -> AsyncIterator[bytes]:
        said = (text or "").strip()
        if not said:
            return
        use_voice = (voice or self.voice or _DEFAULT_VOICE).strip()

        from . import _voice_cache
        path = self._cache_path(said)
        kept = await asyncio.to_thread(_voice_cache.read, path)
        if kept is not None:
            for i in range(0, len(kept), _CHUNK):
                yield kept[i:i + _CHUNK]
            return

        try:
            pcm = await asyncio.to_thread(self._synthesise, said, use_voice)
        except Exception as exc:
            raise RuntimeError(f"Gemini voice failed: {exc}") from exc

        await asyncio.to_thread(_voice_cache.write, path, pcm)

        for i in range(0, len(pcm), _CHUNK):
            yield pcm[i:i + _CHUNK]


def _make(cfg: dict) -> GeminiTTS:
    return GeminiTTS(cfg)


def available() -> bool:
    try:
        from core import gemini
        return bool(gemini.api_key())
    except Exception:
        return False


def voices() -> tuple:
    return _VOICES


PROVIDER = {
    "id": "gemini_tts",
    "kind": "tts",
    "label": "Gemini voice",
    "hint": "The same voices as the live engine — nothing to download",
    "requires": (),
    "catalog": (
        {"id": _DEFAULT_MODEL, "label": "Gemini 2.5 Flash TTS",
         "note": "$10 / 1M audio tokens · repeated phrases are cached",
         "local": False,
         "values": {"model": _DEFAULT_MODEL, "voice": _DEFAULT_VOICE}},
    ),
    "needs": (
        Field("voice", "Voice", kind="choice", choices=_VOICES,
              default=_DEFAULT_VOICE, required=False),
        Field("model", "Model", kind="choice", choices=_MODELS,
              default=_DEFAULT_MODEL, required=False, advanced=True),
        Field("cache_phrases", "Reuse short phrases", kind="bool", default=True,
              required=False,
              hint="Keeps spoken confirmations on disk instead of paying twice",
              advanced=True),
        Field("api_key", "API key", kind="secret", default="", required=False,
              hint="Leave empty to use the assistant's own Gemini key",
              advanced=True),
    ),
    "caps": {"audio_out": True},
    "factory": _make,
}
