"""
The subscription's hearing, thinking and voice — the same three providers an
open-source user configures by hand, filled in from the licence instead.

Nothing here talks to a server itself: each factory builds the ordinary
OpenAI-protocol provider (llm_openai, stt_whisper's API backend, tts_openai)
with the address, credential and trusted certificate core/service.py derives
from the sign-in. So everything those providers already do — streaming, tool
calls, the reasoning filter, kept-open connections, the phrase cache — the
subscription gets without a second copy of it.

Hidden from the settings panel's pickers: a subscriber does not choose these,
and an open-source user cannot use them without a licence.
"""
from __future__ import annotations

from importlib.util import find_spec


def _linked() -> bool:
    try:
        from core import service
        return service.linked()
    except Exception:
        return False


def _cfg(kind: str, overrides: dict) -> dict:
    from core import service
    cfg = service.provider_config(kind)
    if not cfg:
        raise RuntimeError("this computer is not signed in to a subscription")
    # Only the panel's harmless extras (`log`) pass through; the address and
    # credential always come from the licence.
    return {**cfg, **{k: v for k, v in (overrides or {}).items() if k == "log"}}


def _llm(overrides: dict):
    from core import service
    from .llm_openai import OpenAICompatLLM
    return OpenAICompatLLM(_cfg(service.LLM, overrides))


def _stt(overrides: dict):
    from core import service
    from .stt_whisper import WhisperSTT
    return WhisperSTT(_cfg(service.STT, overrides))


def _tts(overrides: dict):
    from core import service
    from .tts_openai import OpenAISpeechTTS
    return OpenAISpeechTTS(_cfg(service.TTS, overrides))


def available() -> bool:
    return find_spec("httpx") is not None and _linked()


_COMMON = {"requires": ("httpx",), "needs": (), "hidden": True, "managed": _linked}

PROVIDERS = (
    {**_COMMON, "id": "service_llm", "kind": "llm", "label": "Subscription model",
     "hint": "The subscription's language model",
     "caps": {"tools": True, "tool_format": "openai", "vision": True}, "factory": _llm},
    {**_COMMON, "id": "service_stt", "kind": "stt", "label": "Subscription speech recognition",
     "hint": "The subscription's speech recognition", "caps": {}, "factory": _stt},
    {**_COMMON, "id": "service_tts", "kind": "tts", "label": "Subscription voice",
     "hint": "The subscription's voice", "caps": {"audio_out": True}, "factory": _tts},
)
