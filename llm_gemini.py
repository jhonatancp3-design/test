"""The thinking half of the cheap lane: Gemini's text models, with tools.

WHY NOT llm_openai POINTED AT GEMINI
    Gemini's OpenAI-compatible endpoint exists, and using it would cost nothing
    to write. It would also throw away the one thing this lane is built for:
    caching. Google discounts a repeated prompt prefix by 90%, automatically, on
    paid projects — measured on this key, 16,355 of 18,513 prompt tokens came
    back marked cached on the second identical call, and an explicit cache
    covered 18,501 of them. The native client reports `cached_content_token_count`
    so the saving can be SEEN; the compatibility layer hides it.

    That 90% is the whole economics of the cheap lane. A short command costs
    about $0.0029 here against $0.0139 on Live, and most of the difference is
    that the assistant's instructions and tool list are paid for once instead of
    once per sentence.

WHAT THIS FILE DOES ABOUT CACHE HITS
    A cache hit needs the prompt to START the same way, byte for byte, every
    time. So the system instruction and the tool declarations go first and are
    never rewritten mid-conversation, and everything that changes — the clock,
    the user's name, what they just said — goes after them. The same discipline
    is what makes the hit rate carry ACROSS users of one project: every install
    running the same version sends the same first ten thousand tokens.

WHY THE HISTORY ARRIVES IN OPENAI'S SHAPE
    `engine_pipeline` keeps its history that way because `llm_openai` was the
    first provider written. Converting here, in the one file that needs it, is
    smaller than changing the engine and every provider that already works.
"""
from __future__ import annotations

import asyncio
import json
import queue
import threading
from typing import Any, AsyncIterator, Optional

from .base import DEFAULT_CAPS, Field, FunctionCall

_DEFAULT_MODEL = "gemini-3.5-flash-lite"
_MODELS = ("gemini-3.5-flash-lite", "gemini-3.1-flash-lite", "gemini-3.5-flash",
           "gemini-3.6-flash", "gemini-3.8-flash")

_SENTINEL = object()


def _sig_key(call_id: Any, name: str, args: dict) -> str:
    """How a thought signature is found again.

    By the call's own id when there is one, and by name-and-arguments when
    there is not — `engine_pipeline` invents an id for a call that arrives
    without one, and the invented id is stable for the same call, so both
    routes lead back to the same signature.
    """
    if call_id:
        return f"id:{call_id}"
    try:
        return f"call:{name}:{json.dumps(args, sort_keys=True, ensure_ascii=False)}"
    except Exception:
        return f"call:{name}"


def _text_of(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):          # some callers pass parts
        return " ".join(str(p) for p in content if isinstance(p, str))
    return "" if content is None else str(content)


def _to_contents(messages: list[dict], signatures: dict | None = None) -> tuple[str, list]:
    """(system instruction, Gemini contents) out of OpenAI-shaped messages.

    `signatures` carries the thought signatures the model attached to its own
    function calls. Gemini 3 REQUIRES them back when that call is replayed as
    history — without it the next turn is refused outright:

        400 INVALID_ARGUMENT: Function call is missing a thought_signature in
        functionCall parts.

    which is what happened the first time this engine was used for real: the
    greeting worked, the assistant called `system_status`, and the turn after
    it died. The history this file is handed is OpenAI-shaped and has nowhere
    to keep a signature, so it is kept here, beside the conversation, and put
    back on the way out.
    """
    from google.genai import types as gtypes

    system_bits: list[str] = []
    contents: list = []
    for m in messages or ():
        role = (m.get("role") or "").lower()

        if role == "system":
            text = _text_of(m)
            if text:
                system_bits.append(text)

        elif role == "user":
            # Speech that has not been transcribed yet goes to the model AS
            # speech. Measured: recognising it first and then sending the words
            # costs 1.9 seconds and a second request, and this model reads the
            # audio itself for about 25 tokens a second — a four-second command
            # is a hundredth of a cent. The transcript still arrives, in the
            # background, for the activity log and the conversation window.
            pcm = m.get("audio")
            if pcm and not _text_of(m):
                from .stt_gemini import _wav
                contents.append(gtypes.Content(role="user", parts=[
                    gtypes.Part.from_bytes(data=_wav(pcm, int(m.get("rate") or 16000)),
                                           mime_type="audio/wav")]))
                continue
            text = _text_of(m)
            if text:
                contents.append(gtypes.Content(
                    role="user", parts=[gtypes.Part.from_text(text=text)]))

        elif role == "assistant":
            parts = []
            text = _text_of(m)
            if text:
                parts.append(gtypes.Part.from_text(text=text))
            for call in (m.get("tool_calls") or ()):
                fn = (call or {}).get("function") or {}
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except Exception:
                    args = {}
                name = fn.get("name") or ""
                part = gtypes.Part(
                    function_call=gtypes.FunctionCall(name=name, args=args))
                sig = (signatures or {}).get(_sig_key(call.get("id"), name, args))
                if sig:
                    part.thought_signature = sig
                parts.append(part)
            if parts:
                contents.append(gtypes.Content(role="model", parts=parts))

        elif role == "tool":
            # A tool result is a turn the USER side of the conversation carries;
            # Gemini has no separate role for it.
            contents.append(gtypes.Content(role="user", parts=[
                gtypes.Part.from_function_response(
                    name=m.get("name") or "tool",
                    response={"result": _text_of(m)})]))

    return "\n\n".join(system_bits), contents


class GeminiLLM:
    def __init__(self, cfg: dict):
        self.model = (cfg.get("model") or _DEFAULT_MODEL).strip()
        self.temperature = float(cfg.get("temperature") or 0.4)
        self._key = (cfg.get("api_key") or "").strip()
        self.caps = dict(DEFAULT_CAPS)
        self.caps.update({"vision": True, "tools": True, "tool_format": "gemini",
                          "max_tools": None,
                          # This model reads speech directly, so the engine can
                          # skip the recognition step and its 1.9 seconds.
                          "audio_in": True})
        self.last_usage: dict = {}
        # Thought signatures, by call. Gemini 3 will not accept its own
        # function call back as history without the signature it issued.
        self._signatures: dict[str, Any] = {}

    def _client(self):
        from core import gemini
        return gemini.client(key=self._key or gemini.api_key())

    def _stream(self, messages: list[dict], tools: Optional[list], out: queue.Queue) -> None:
        """Runs in a thread: the SDK's streaming call is synchronous."""
        from google.genai import types as gtypes
        try:
            system, contents = _to_contents(messages, self._signatures)
            config = gtypes.GenerateContentConfig(
                temperature=self.temperature,
                system_instruction=system or None,
                tools=[{"function_declarations": list(tools)}] if tools else None,
            )
            # The client is held in a NAMED local for the whole stream. Built
            # inline, its last reference dies with the expression, the SDK
            # closes its transport on collection, and the next chunk raises
            # "Cannot send a request, as the client has been closed" — which is
            # what happened the first time this file ran.
            client = self._client()
            stream = client.models.generate_content_stream(
                model=self.model, contents=contents, config=config)
            for chunk in stream:
                usage = getattr(chunk, "usage_metadata", None)
                if usage is not None:
                    # Kept so the cheap lane can PROVE its cache hits instead of
                    # assuming them; the credit meter reads this.
                    self.last_usage = {
                        "prompt": getattr(usage, "prompt_token_count", 0) or 0,
                        "cached": getattr(usage, "cached_content_token_count", 0) or 0,
                        "output": getattr(usage, "candidates_token_count", 0) or 0,
                    }
                for candidate in (getattr(chunk, "candidates", None) or ()):
                    parts = getattr(getattr(candidate, "content", None), "parts", None) or ()
                    for part in parts:
                        call = getattr(part, "function_call", None)
                        if call is not None and getattr(call, "name", ""):
                            call_id = getattr(call, "id", "") or f"call_{call.name}"
                            args = dict(call.args or {})
                            # Kept so the NEXT turn can hand it back; see
                            # `_to_contents`. Stored under both routes because
                            # the id may be the one invented downstream.
                            sig = getattr(part, "thought_signature", None)
                            if sig:
                                self._signatures[_sig_key(call_id, call.name, args)] = sig
                                self._signatures[_sig_key(None, call.name, args)] = sig
                            out.put({"tool_calls": [FunctionCall(
                                id=call_id, name=call.name, args=args)]})
                        text = getattr(part, "text", "")
                        if text:
                            out.put({"text": text})
        except Exception as exc:
            out.put({"error": f"{type(exc).__name__}: {exc}"})
        finally:
            out.put(_SENTINEL)

    async def chat(self, messages: list[dict], tools: list[dict] | None = None,
                   *, stream: bool = True) -> AsyncIterator[dict]:
        out: queue.Queue = queue.Queue()
        worker = threading.Thread(target=self._stream, args=(messages, tools, out),
                                  daemon=True)
        worker.start()
        loop = asyncio.get_running_loop()
        while True:
            item = await loop.run_in_executor(None, out.get)
            if item is _SENTINEL:
                break
            if "error" in item:
                raise RuntimeError(item["error"])
            yield item


def _make(cfg: dict) -> GeminiLLM:
    return GeminiLLM(cfg)


def available() -> bool:
    try:
        from core import gemini
        return bool(gemini.api_key())
    except Exception:
        return False


PROVIDER = {
    "id": "gemini_llm",
    "kind": "llm",
    "label": "Gemini (text)",
    "hint": "The cheap lane's brain — repeated instructions are billed at 10%",
    "requires": (),
    "catalog": (
        {"id": _DEFAULT_MODEL, "label": "Gemini 3.5 Flash Lite",
         "note": "$0.30 / 1M in · $0.03 cached · fastest", "local": False,
         "values": {"model": _DEFAULT_MODEL}},
        {"id": "gemini-3.5-flash", "label": "Gemini 3.5 Flash",
         "note": "$1.50 / 1M in · for harder requests", "local": False,
         "values": {"model": "gemini-3.5-flash"}},
    ),
    "needs": (
        Field("model", "Model", kind="choice", choices=_MODELS,
              default=_DEFAULT_MODEL, required=False),
        Field("temperature", "Temperature", kind="number", default=0.4,
              required=False, advanced=True),
        Field("api_key", "API key", kind="secret", default="", required=False,
              hint="Leave empty to use the assistant's own Gemini key",
              advanced=True),
    ),
    "caps": {"vision": True, "tools": True, "tool_format": "gemini", "max_tools": None},
    "factory": _make,
}
