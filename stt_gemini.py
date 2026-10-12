"""Speech recognition through Gemini, for the lane that is not Live.

WHY A HOSTED RECOGNISER WHEN A LOCAL ONE ALREADY WORKS
    `stt_whisper` runs on the user's own machine and costs nothing per minute.
    It is the right answer for someone who will not send a microphone anywhere,
    and the wrong default for a paid product: it needs a model download, it
    needs CPU the user may not have, and on a slow machine it turns a one-second
    answer into a four-second one. The people who feel that are exactly the
    people least able to fix it.

    So this exists as the recogniser for the cheap lane — the path a short
    command takes instead of waking the Live socket. Measured: speech costs
    about 25 tokens a second, so a four-second command is a hundredth of a
    cent, against $0.005 a minute for Live's own ears — and it leaves the
    user's computer alone.

WHY WAV AND NOT RAW PCM
    The Live API takes `audio/pcm;rate=16000` on the wire. generateContent does
    not: its audio types are the file formats (wav, mp3, flac, ogg, aac, aiff).
    The microphone hands this project raw 16-bit mono PCM, so a 44-byte RIFF
    header goes in front of it here. No dependency, no copy of the samples, no
    re-encoding — the header IS the conversion.
"""
from __future__ import annotations

import asyncio
import struct
from typing import Optional

from .base import DEFAULT_CAPS, Field

# WHY NOT `gemini-3.5-transcribe`, WHICH IS THE ONE NAMED FOR THIS JOB
#     Measured on this key: through generateContent it answers with an empty
#     string, every time, for audio a general model transcribes perfectly —
#     audio-only, with a prompt, and with response_modalities=["TEXT"] alike.
#     It is a model for the streaming socket, and this lane is not that socket.
#     It stays in the list below for the day that changes, and the default is
#     the model that actually returns words.
#
#     Flash-lite is also cheaper here: it bills audio at $0.30 per million
#     tokens like everything else, and speech costs about 25 tokens a second —
#     a four-second command is 100 tokens, a hundredth of a cent.
_DEFAULT_MODEL = "gemini-3.5-flash-lite"

_MODELS = ("gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-3.5-flash",
           "gemini-3.5-transcribe")

# A clip shorter than this is a cough, a door, or the tail of the assistant's
# own voice. Sending it costs a request and returns nothing useful.
_MIN_SAMPLES = 16000 // 4          # a quarter second at 16 kHz


def _wav(pcm: bytes, sample_rate: int) -> bytes:
    """A RIFF header in front of 16-bit mono samples."""
    return (b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVEfmt "
            + struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16)
            + b"data" + struct.pack("<I", len(pcm)) + pcm)


class GeminiSTT:
    def __init__(self, cfg: dict):
        self.model = (cfg.get("model") or _DEFAULT_MODEL).strip()
        self.caps = dict(DEFAULT_CAPS)
        self._key = (cfg.get("api_key") or "").strip()

    def _client(self):
        from core import gemini
        return gemini.client(key=self._key or gemini.api_key())

    async def transcribe(self, pcm: bytes, sample_rate: int = 16000,
                         language: str = "") -> str:
        if not pcm or len(pcm) < _MIN_SAMPLES * 2:
            return ""

        def _run() -> str:
            from google.genai import types as gtypes
            client = self._client()
            # No language is named even when the caller knows one. The
            # assistant changes language mid-conversation and the recogniser
            # that was TOLD Turkish mishears the English sentence after it.
            response = client.models.generate_content(
                model=self.model,
                contents=[
                    gtypes.Part.from_bytes(data=_wav(pcm, sample_rate),
                                           mime_type="audio/wav"),
                    "Write out exactly what is said, in the language it is "
                    "spoken in. No translation, no commentary, no quotation "
                    "marks. Empty reply if there is no speech.",
                ],
                config=gtypes.GenerateContentConfig(temperature=0.0),
            )
            return (getattr(response, "text", "") or "").strip()

        try:
            text = await asyncio.to_thread(_run)
        except Exception as exc:
            raise RuntimeError(f"Gemini transcription failed: {exc}") from exc
        # A recogniser handed silence sometimes narrates the silence.
        return "" if text.lower() in ("", "(no speech)", "no speech", "...") else text


def _make(cfg: dict) -> GeminiSTT:
    return GeminiSTT(cfg)


def available() -> bool:
    """True when a Gemini key is configured — the same key the assistant is
    already using, so choosing this provider asks the user for nothing."""
    try:
        from core import gemini
        return bool(gemini.api_key())
    except Exception:
        return False


PROVIDER = {
    "id": "gemini_stt",
    "kind": "stt",
    "label": "Gemini transcription",
    "hint": "Hosted recognition — nothing to download, no load on this computer",
    "requires": (),
    "catalog": (
        {"id": _DEFAULT_MODEL, "label": "Gemini 3.5 Flash Lite",
         "note": "~$0.0001 per command · nothing to download", "local": False,
         "values": {"model": _DEFAULT_MODEL}},
    ),
    "needs": (
        Field("model", "Model", kind="choice", choices=_MODELS,
              default=_DEFAULT_MODEL, required=False, advanced=True),
        Field("api_key", "API key", kind="secret", default="", required=False,
              hint="Leave empty to use the assistant's own Gemini key",
              advanced=True),
    ),
    "caps": {"audio_in": True},
    "factory": _make,
}
