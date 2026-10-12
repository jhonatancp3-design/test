"""
The second way to hold the conversation: speech in, a model, speech out.

WHAT IT IS
    Gemini Live is one socket carrying audio in, audio out, transcripts and tool
    calls together, with the turn boundary decided by the server. This is the
    same conversation assembled from three separate pieces — an STT, an LLM and
    a TTS — behind an object that emits the events main.py already reads.

    Everything upstream of `receive()` is unchanged: the avatar, the lip-sync,
    the self-echo guard, push-to-talk, the wake word, undo, the confirmation
    gate, memory, and every action and plugin. They cannot tell which engine is
    running, and that is the whole design.

WHAT IT HONESTLY COSTS
    Live answers in roughly 0.3-0.6s because there is no separate recognition
    step, no separate synthesis step and no round trip to find the end of a
    turn. Here there are all three, in series. On ordinary hardware expect
    1.5-4s, and a turn that calls a tool costs TWO model round trips rather than
    one, because a stateless chat API has to be asked again with the result.

    So this is not a faster assistant. It is the assistant continuing to work
    when the quota is gone, the network is down, or the microphone is not
    allowed to leave the building.

THE DEADLOCK THIS FILE IS SHAPED AROUND
    main.py calls `send_tool_response()` from INSIDE its `async for` over
    `receive()` — see the tool_call branch of `_receive_audio`. If that call
    waited for the generation it triggers, it would be waiting for events that
    only the loop it is blocking can consume. The whole turn would stop, once,
    on the first tool call, and look exactly like a model that hung.

    So every send method here does the same thing: record what happened, kick
    off work as a task, return. Nothing that main.py awaits ever waits for the
    model.

NO LANGUAGE LOGIC ANYWHERE IN THIS FILE
    Not a keyword, not a greeting, not a locale check. What language the
    assistant speaks is decided by the system prompt and the model, exactly as
    it is on Gemini. The STT is asked to detect rather than told what to expect,
    and the transcript is passed through without being inspected.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import re
import sys
import time
from typing import Any, AsyncIterator, Optional

from . import schema
from .base import DEFAULT_CAPS, FunctionCall, Msg


def _carried_over() -> list[dict]:
    """The turns the other engine already heard, or nothing.

    Imported here rather than at module scope on purpose: `core.conversation`
    belongs to the app, this package is meant to survive being deleted, and a
    provider that cannot import it should still run — with an empty history,
    which is exactly how it behaved before the shared window existed.
    """
    try:
        from core import conversation
        return list(conversation.recent())
    except Exception:
        return []


def _meter(kind: str) -> None:
    """Record a turn against the allowance. A cheap-lane turn costs nothing —
    it is recorded so the app can show what it did, not to charge for it."""
    try:
        from core import credits
        credits.spend(kind)
    except Exception:
        pass


def _remember(role: str, text: str) -> None:
    """Put one side of a turn into the shared window. Same import discipline,
    same reason: a missing app is not an error a provider should raise."""
    try:
        from core import conversation
        conversation.add(role, text, lane="cheap")
    except Exception:
        pass

# How much conversation to carry. Gemini Live compresses its own context with a
# sliding window server-side; a stateless chat API has no such thing, so the
# window is kept here.
#
# Counted in characters rather than tokens on purpose: a real tokenizer would
# mean a dependency and a model-specific vocabulary, and the number it produces
# would still be an estimate for every model but one. Four characters per token
# is close enough for a budget whose only job is to stay well inside a context
# window.
_DEFAULT_HISTORY_CHARS = 24_000
# What one image costs in the history budget (its base64 is not the measure).
_IMAGE_CHARS = 2_000


def _is_image(part) -> bool:
    return isinstance(part, dict) and part.get("type") == "image_url"


def _size(message: dict) -> int:
    content = message.get("content")
    if isinstance(content, list):
        images = sum(1 for p in content if _is_image(p))
        rest = dict(message, content=[p for p in content if not _is_image(p)])
        return len(json.dumps(rest, ensure_ascii=False, default=lambda _o: '<audio>')) \
            + images * _IMAGE_CHARS
    return len(json.dumps(message, ensure_ascii=False, default=lambda _o: '<audio>'))


def _older_images_dropped(messages: list[dict]) -> list[dict]:
    """Only the newest image is sent again; earlier ones become a note. Every
    screenshot of a conversation re-sent on every turn is megabytes to the
    server for pictures of a screen that has changed since."""
    newest = max((i for i, m in enumerate(messages)
                  if isinstance(m.get("content"), list) and any(_is_image(p) for p in m["content"])),
                 default=-1)
    out = []
    for i, m in enumerate(messages):
        content = m.get("content")
        if i != newest and isinstance(content, list) and any(_is_image(p) for p in content):
            words = [p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"]
            m = dict(m, content="\n".join([*filter(None, words), "[an earlier image, no longer shown]"]))
        out.append(m)
    return out

# ── Finding the end of a turn ────────────────────────────────────────────────
# Gemini Live decides this on the server: it hears the user stop and closes the
# turn. Nothing here does that for us, so it is decided from the audio.
#
# Deliberately energy and silence, not a model. A neural voice-activity detector
# would be better at it and would also mean a second download, a second thing to
# warm up and a second thing that behaves differently per platform. Push-to-talk
# remains the exact answer for anybody who wants one; this is what makes the
# engine usable without it.
#
# The threshold is not a constant. A fixed one is wrong in both directions — it
# never triggers in a noisy room and never stops in a silent one — so the noise
# floor is measured continuously from the quietest recent frames and speech is
# whatever stands well above it.
_VAD_SILENCE_MS = 700      # quiet for this long after speech = the turn ended
# With a speech model deciding what "quiet" is (providers/vad.py), noise no
# longer reads as talking, so the end of a turn can be called sooner. 550 ms is
# the end-of-turn silence this project already tuned for Live's own detector
# (memory/config_manager.get_turn_tuning).
_VAD_SILENCE_MS_MODEL = 550
_VAD_SPEECH_PROB = 0.5     # Silero: speech at or above this probability
# No lower 'keep talking' threshold: the silence must be unbroken for the
# whole end-of-turn wait, so a dip between words cannot end a turn anyway, and
# a lower bar only delayed the start of that wait (measured ~40 ms: the
# probability takes ~100-130 ms to fall after the last word).
_VAD_SPEECH_KEEP = _VAD_SPEECH_PROB
# ...but the answer is started, unseen, much earlier. See `_start_speculation`.
_VAD_SPECULATE_MS = 300    # quiet this long: recognise and answer, holding the output
_VAD_RESUME_MS = 120       # speech this long during that wait: it was a pause, drop it
_VAD_MIN_SPEECH_MS = 250   # shorter than this is a cough, a click, a door
_VAD_MAX_TURN_S = 30.0     # a runaway stream is flushed rather than kept for ever
_VAD_FLOOR_MARGIN = 3.5    # speech is this many times the measured noise floor
_VAD_ABSOLUTE_MIN = 0.006  # ...and never below this, so a silent room's own
                           # floor of near-zero cannot make hiss into speech

# ── Stopping a model that will not stop ──────────────────────────────────────
# Gemini Live ends a turn itself. A chat API does not: every tool result is a
# fresh request, and if the model answers it with the same tool call again, the
# loop has no floor.
#
# It happened on the first real attempt. "open whatsapp" opened WhatsApp four
# times — the tool worked, returned "Opened WhatsApp", and a small model asked
# for it again rather than replying. Every repeat launched the application
# again, on top of the copy already running.
#
# Two limits, because they catch different faults. A REPEATED call is a model
# that did not understand the result it got, and re-running the tool is
# actively harmful — so it is answered from what the tool already returned and
# never reaches the app twice. A long chain of DIFFERENT calls is a model
# wandering, which is not harmful but is not converging either, so it is capped
# and told to answer with what it has.
_MAX_TOOL_ROUNDS = 4

# A reply that says it is doing something but carries no tool call ("sending it
# now", and nothing is sent) - seen in the beta, about 1 turn in 16 on a task.
# Asking the model with its tools whether it missed one was answered "OK"
# 19/30; a separate yes/no reading of the reply, then a nudge with the tools,
# 29/30, with no call made on answers, questions or small talk (2026-10-12).
_JUDGE_SYSTEM = ("You read one exchange between a user and a voice assistant that acts on "
                 "the user's computer by calling tools. Answer with one word, YES or NO.")
_JUDGE_ASK = ("User: {user}\nAssistant: {said}\n\nDoes the assistant's reply say it is doing, "
              "or is about to do, an action on the computer or in an app (open, close, send, "
              "write, search, play, click...) - rather than only answering, asking or chatting?")
_NUDGE = ("[SYSTEM] You said that but did not call the tool, so nothing happened. "
          "Call the tool now, with no text.")


class PipelineEngine:
    """STT -> LLM -> TTS, wearing the Live session's interface."""

    def __init__(self, llm, stt=None, tts=None, *, config=None,
                 log=None, history_chars: int = _DEFAULT_HISTORY_CHARS):
        self._llm = llm
        self._stt = stt
        self._tts = tts
        self._log = log or (lambda m: None)
        self._history_chars = history_chars

        self.caps = dict(DEFAULT_CAPS)
        llm_caps = getattr(llm, "caps", {}) or {}
        self.caps.update({
            "tools": bool(llm_caps.get("tools")),
            "tool_format": llm_caps.get("tool_format"),
            "max_tools": llm_caps.get("max_tools"),
            "max_tool_chars": llm_caps.get("max_tool_chars"),
            "vision": bool(llm_caps.get("vision")),
            "audio_out": tts is not None,
            # A local pipeline has no server deciding when the user has started
            # talking over it. Until an engine can do that, `interrupt()` stops
            # playback and this stays false rather than claiming otherwise.
            "barge_in": False,
            "non_blocking": False,
            "resumption": False,
        })

        # The system prompt and the tool declarations come from the config
        # main.py already built for the Live session — same prompt, same tools,
        # same identity, same memory. Nothing is composed twice.
        self._system = ""
        self._tools: Optional[list] = None
        self._declarations: list = []
        if config is not None:
            self._read_config(config)

        # Seeded from the conversation both lanes share, so an engine that
        # starts mid-conversation does not start mid-conversation AMNESIC.
        # Gemini Live keeps its context on the server; this side has none, and
        # without this the user asking "make it louder" right after a lane
        # change is asking about nothing.
        self._history: list[dict] = _carried_over()
        self._queue: asyncio.Queue = asyncio.Queue()
        self._gen: Optional[asyncio.Task] = None
        # The generation a tool result started, and whether anything has been
        # emitted since (see _queue_generation).
        self._tool_gen: Optional[asyncio.Task] = None
        self._gen_spoke = True
        # Speech goes straight to a model that can read it; see
        # `flush_microphone`. False for a text-only model, which still needs
        # the recogniser in front of it.
        self._audio_native = bool(llm_caps.get("audio_in"))
        # Background work that must not be garbage collected mid-flight.
        self._tasks: set = set()
        self._closed = False
        # Tool calls handed out and not yet answered. A model that emits two
        # calls in one turn must have BOTH answered before the next generation,
        # or the second result arrives with no call to attach to and several
        # servers reject the whole request for it.
        self._awaiting: dict[str, str] = {}
        # What has already been run this turn: signature -> the result the tool
        # gave. Cleared when the user speaks again.
        self._done_calls: dict[str, str] = {}
        self._pending_sigs: dict[str, str] = {}
        self._rounds = 0
        self._last_said = ""
        # Something arrived while a generation was running and still needs an
        # answer once that one is finished.
        self._followup = False
        self._mic: list[bytes] = []
        self._mic_bytes = 0
        # Turn detection state. `auto_turn` is off unless something switches it
        # on, so push-to-talk keeps behaving exactly as it does today.
        self.auto_turn = False
        self._on_turn_end = None          # set by main.py: what to do on a flush
        # A turn answered ahead of time, its output held until the silence
        # proves the user really finished. None when nothing is in flight.
        self.speculate = True
        self._vad = None           # providers/vad.SileroVAD, loaded on first use
        self._vad_tried = False
        self._vad_prob = 0.0
        self._vad_in_speech = False
        self._spec: Optional[dict] = None
        # What the assistant last said out loud, and until when hearing it
        # again counts as its own echo (see _is_own_echo).
        self._echo_text = ""
        self._echo_until = 0.0
        self._spec_resume_ms = 0.0
        self._vad_floor = _VAD_ABSOLUTE_MIN
        self._vad_speech_ms = 0.0
        self._vad_quiet_ms = 0.0
        self._vad_started = False
        # Per-turn stage timings, printed to the console when the turn ends.
        self._timing: dict = {}
        # Set by main.py: offered every failure first, returns True when it is
        # something the app explains itself (a subscription out of allowance,
        # a computer over the plan's limit). Then the engine stays quiet about
        # it instead of blaming "the language model" for a refusal.
        self.on_failure = None

    # ── configuration ────────────────────────────────────────────────────────
    def _read_config(self, config) -> None:
        si = getattr(config, "system_instruction", None)
        if si:
            self._system = si if isinstance(si, str) else str(si)

        decls: list = []
        for group in (getattr(config, "tools", None) or []):
            fd = (group.get("function_declarations") if isinstance(group, dict)
                  else getattr(group, "function_declarations", None))
            if fd:
                decls.extend(fd)

        self._declarations = schema.budget(
            decls,
            max_tools=self.caps.get("max_tools"),
            max_chars=self.caps.get("max_tool_chars"),
        )
        fmt = self.caps.get("tool_format")
        if not self.caps.get("tools") or not self._declarations:
            self._tools = None
        elif fmt == "prompt":
            # No native tool calling: the declarations become part of the
            # instructions and the reply is parsed. Less reliable than a real
            # tool API, and the engine says so through caps rather than
            # pretending the difference is not there.
            self._system = (self._system + "\n\n"
                            + schema.to_prompt(self._declarations)).strip()
            self._tools = None
        else:
            self._tools = schema.convert(self._declarations, fmt)

        kept, total = len(self._declarations), len(decls)
        if kept < total:
            print(f"[Pipeline] {kept} of {total} tools declared "
                      f"({schema.payload_size(self._declarations):,} chars) — "
                      f"this model's budget.")

    def tool_names(self) -> list[str]:
        """What the model was actually offered. main.py uses this to decide what
        the assistant should be told it cannot do."""
        out = []
        for d in self._declarations:
            n = d.get("name") if isinstance(d, dict) else getattr(d, "name", "")
            if n:
                out.append(n)
        return out

    # ── history ──────────────────────────────────────────────────────────────
    def _messages(self) -> list[dict]:
        msgs = [{"role": "system", "content": self._system}] if self._system else []
        return msgs + self._language_hint(_older_images_dropped(self._trimmed()))

    def _note_spoken(self, text: str) -> None:
        """What the user just SAID, and the language it was heard in."""
        self._spoken = (text, getattr(self._stt, "detected_language", "") or "")

    def _language_hint(self, messages: list[dict]) -> list[dict]:
        """The newest user message, when it was spoken, carries the language it
        was heard in — for this request only, never kept in the history.

        The prompt says to answer in the language of the latest message, and
        after a conversation in one language the model still answered another
        in it: measured on the subscription's model, Turkish history then
        English questions, 6 of 9 answers in Turkish; with the language named
        in the message, 0 of 9 (2026-10-10). Spoken turns only: speech
        recognition knows the language; typed text and the app's own
        messages are left as they are."""
        text, code = getattr(self, "_spoken", ("", ""))
        if not text or not code:
            return messages
        for i in range(len(messages) - 1, -1, -1):
            m = messages[i]
            if m.get("role") != "user":
                continue
            if m.get("content") != text:
                return messages
            try:
                from core.i18n import english_name
                name = english_name(code)
            except Exception:
                name = ""
            if not name or name.lower() == code.lower():
                # Not one of the interface's languages: its ISO code, which the
                # model reads as well as a name.
                name = f"the language with ISO code '{code}'"
            out = list(messages)
            hint = f"[The user is speaking {name}. Reply in {name}.]"
            out[i] = dict(m, content=text + "\n\n" + hint)
            return out
        return messages

    def _trimmed(self) -> list[dict]:
        """The tail of the conversation that fits the budget.

        Trimmed from the front, and never through the middle of an exchange: a
        tool result whose call has been trimmed away is an orphan that several
        servers reject outright, and a model that keeps the result without the
        request reads it as something the user said.

        An image counts as _IMAGE_CHARS, not as the length of its base64: a
        screenshot is hundreds of kilobytes of text, and counted that way it
        alone was over the budget, everything was cut — the question with it —
        and the server refused a conversation with no user message in it
        ("No user query found in messages", found live 2026-10-10). The
        newest user message is never cut for the same reason.
        """
        total, cut = 0, 0
        for i in range(len(self._history) - 1, -1, -1):
            total += _size(self._history[i])
            if total > self._history_chars:
                cut = i + 1
                break
        last_user = max((i for i, m in enumerate(self._history) if m.get("role") == "user"),
                        default=len(self._history))
        cut = min(cut, last_user)
        while cut < len(self._history) and self._history[cut].get("role") == "tool":
            cut += 1
        return self._history[cut:]

    # ── the four methods main.py calls ───────────────────────────────────────
    async def send_realtime_input(self, *, audio) -> None:
        """Microphone PCM. Buffered until the turn ends; recognised then.

        Nothing is sent anywhere while the user is still speaking. With no STT
        configured the bytes are dropped rather than accumulated, so text-only
        operation does not grow a buffer nobody will ever read.
        """
        if self._stt is None or self._closed:
            return
        data = _pcm_of(audio)
        if not data:
            return
        self._mic.append(data)
        self._mic_bytes += len(data)
        if self.auto_turn:
            await self._watch_for_turn_end(data)

    async def _watch_for_turn_end(self, frame: bytes) -> None:
        """Decide from the audio whether the user has finished speaking."""
        from . import audio as _a

        samples = _a.to_float32(frame)
        if samples.size == 0:
            return
        import numpy as np

        rms = float(np.sqrt(np.mean(samples * samples)))
        ms = samples.size / _a.MIC_RATE * 1000.0

        if not self._vad_tried:
            self._vad_tried = True
            from . import vad as _vad
            self._vad = _vad.load()
            if self._vad is not None:
                print("[Pipeline] end of speech is judged by a speech model (Silero VAD).")

        # The floor tracks downward quickly and upward slowly: a room that goes
        # quiet should be recognised at once, while one loud moment must not
        # raise the bar so far that the next sentence is inaudible to it.
        if rms < self._vad_floor:
            self._vad_floor = rms
        else:
            self._vad_floor += (rms - self._vad_floor) * 0.002

        threshold = max(self._vad_floor * _VAD_FLOOR_MARGIN, _VAD_ABSOLUTE_MIN)
        silence_ms = _VAD_SILENCE_MS
        if self._vad is not None:
            probs = self._vad.feed(samples)
            if probs:
                self._vad_prob = max(probs)
            cut = _VAD_SPEECH_KEEP if self._vad_in_speech else _VAD_SPEECH_PROB
            self._vad_in_speech = self._vad_prob >= cut
            speaking = self._vad_in_speech
            silence_ms = _VAD_SILENCE_MS_MODEL
        else:
            speaking = rms >= threshold

        if speaking:
            self._vad_speech_ms += ms
            self._vad_quiet_ms = 0.0
            if self._vad_speech_ms >= _VAD_MIN_SPEECH_MS:
                self._vad_started = True
            if self._spec is not None:
                self._spec_resume_ms += ms
                if self._spec_resume_ms >= _VAD_RESUME_MS:
                    self._cancel_speculation()
            return

        if not self._vad_started:
            # Nothing said yet. Do not let a long quiet stretch accumulate into
            # a buffer that will be transcribed as silence — keep only enough to
            # carry the leading consonant of whatever comes next.
            keep = int(_a.MIC_RATE * 0.3) * _a.SAMPLE_WIDTH
            while self._mic_bytes > keep and len(self._mic) > 1:
                self._mic_bytes -= len(self._mic.pop(0))
            self._vad_speech_ms = 0.0
            return

        self._vad_quiet_ms += ms
        self._spec_resume_ms = 0.0
        max_bytes = int(_a.MIC_RATE * _VAD_MAX_TURN_S) * _a.SAMPLE_WIDTH
        if self._vad_quiet_ms >= silence_ms or self._mic_bytes >= max_bytes:
            self._reset_vad()
            if self._spec is not None:
                await self._confirm_speculation()
            else:
                await self.flush_microphone()
            return
        if (self.speculate and self._spec is None and self._vad_quiet_ms >= _VAD_SPECULATE_MS
                and not self._audio_native and (self._gen is None or self._gen.done())):
            self._start_speculation()

    def _reset_vad(self) -> None:
        self._vad_speech_ms = 0.0
        self._vad_quiet_ms = 0.0
        self._vad_started = False

    async def send_client_content(self, *, turns, turn_complete: bool = True) -> None:
        """A user turn: text, an image, or both. Starts a generation."""
        if self._closed:
            return
        parts = (turns or {}).get("parts") if isinstance(turns, dict) else None
        content = self._to_content(parts or [])
        if content in ("", []):
            return

        # NOT emitted as "what the user said" — see `heard_speech`.
        #
        # `send_client_content` is how main.py injects everything: a system
        # alert about GPU load, a captured screenshot, the morning briefing, a
        # proactive check-in. All of them are `role: user`, because that is the
        # only way to put text into a conversation.
        #
        # Emitting an input transcript for each of them put them in the activity
        # log as though the person had said them. A real one read:
        #
        #     You: heyyy open whatsapp [SYSTEM_ALERT] GPU load is at 96%. ...
        #
        # Gemini Live does not do that: its `input_transcription` reports what
        # the MICROPHONE heard and nothing else, so injected text never appears
        # there. The engine matches it — the transcript is emitted where speech
        # is actually recognised, in `flush_microphone`.
        self._history.append({"role": "user", "content": content})
        _remember("user", content)
        self._done_calls.clear()
        self._rounds = 0
        self._last_said = ""
        self._queue_generation()

    async def send_tool_response(self, *, function_responses) -> None:
        """Tool results. Returns immediately — see the module docstring."""
        if self._closed:
            return
        started = self._timing.get("tool_at")
        if started:
            self._tick("tools", time.monotonic() - started)
            self._timing["tool_names"] = ", ".join(
                sorted(set(self._awaiting.values()))) or ""
        for fr in function_responses or []:
            call_id = str(getattr(fr, "id", "") or "")
            name = str(getattr(fr, "name", "") or self._awaiting.get(call_id, ""))
            payload = getattr(fr, "response", None)
            if isinstance(payload, dict) and "result" in payload:
                payload = payload["result"]
            sig = getattr(self, "_pending_sigs", {}).pop(call_id, "")
            if sig:
                self._done_calls[sig] = _stringify(payload)
            self._history.append({
                "role": "tool",
                "tool_call_id": call_id or name,
                "name": name,
                "content": _stringify(payload),
            })
            self._awaiting.pop(call_id, None)

        # Only ask the model again once every call it made has an answer.
        if not self._awaiting:
            self._start_generation()

    async def receive(self) -> AsyncIterator[Msg]:
        while True:
            msg = await self._queue.get()
            if msg is None:
                return
            yield msg

    async def close(self) -> None:
        self._closed = True
        if self._gen and not self._gen.done():
            self._gen.cancel()
        await self._queue.put(None)

    async def warmup(self) -> None:
        """Get every part ready before the user speaks.

        Measured on the speech model: without this the FIRST thing anybody says
        is answered after 8.9 seconds and everything after it after 1.2. The
        eight seconds are a one-off cost of bringing the recogniser up, and the
        worst possible moment to spend them is the first sentence — a person who
        waits that long assumes it is broken and says it again, which produces
        two overlapping turns.
        """
        # Both ends, concurrently: the speech model and the voice each have a
        # one-off cost, and they do not need each other.
        jobs = []
        for part in (self._stt, self._tts):
            warm = getattr(part, "warmup", None)
            if warm is not None:
                jobs.append(warm(self._log))
        if jobs:
            await asyncio.gather(*jobs, return_exceptions=True)

    # ── interruption ─────────────────────────────────────────────────────────
    def interrupt(self) -> None:
        """Stop the current generation. Called when the user talks over it.

        The generation is cancelled but the history is left as it is: whatever
        was already said was heard, and deleting it would make the assistant
        deny having said something the user is replying to.

        The turn is then ENDED, as Live's server ends it. main.py drops audio
        from the moment of an interruption until the next turn_complete; a
        cancelled generation never sends one, so the flag stayed up and the
        NEXT answer was the one thrown away — its sound, its log line and its
        place in memory.
        """
        if self._spec is not None:
            self._cancel_speculation()
        if self._gen and not self._gen.done():
            self._gen.cancel()
            self._followup = False
        # Always, also when the answer was already written and only still
        # PLAYING: main.py is waiting for this to stop discarding sound.
        if not self._closed:
            self._queue.put_nowait(Msg.turn_end())

    # ── generation ───────────────────────────────────────────────────────────
    def _queue_generation(self) -> None:
        """Answer this, after whatever is already being answered.

        The first version cancelled the generation in flight. That is right for
        a person talking over the assistant — and `interrupt()` is what does
        that — but `send_client_content` is ALSO how a background monitor drops
        a GPU alert into the conversation, and it fired while the assistant was
        halfway through answering something else. The answer in progress was
        cancelled and replaced by a reply about the graphics card.

        Nothing that arrives this way is urgent enough to destroy an answer the
        user is waiting for. It waits its turn.
        """
        if self._gen and not self._gen.done():
            if self._gen is self._tool_gen and not self._gen_spoke:
                # The answer to a tool result has not said a word yet, and
                # this belongs to it: main.py sends a screenshot right after
                # screen_process's result (_flush_pending_vision). Answered
                # separately, that was two spoken replies (found live
                # 2026-10-10); one answer that has both is what Live gives.
                self._start_generation()
                return
            self._followup = True
            return
        self._gen = asyncio.create_task(self._generate())

    def _start_generation(self) -> None:
        """Generate now, replacing anything running. Only used where the
        conversation genuinely moved on — a tool result completing the turn."""
        if self._gen and not self._gen.done():
            self._gen.cancel()
        self._gen = self._tool_gen = asyncio.create_task(self._generate())
        self._gen_spoke = False

    async def _generate(self) -> None:
        said: list[str] = []
        calls: list[FunctionCall] = []
        t0 = time.monotonic()
        first_word = None
        thinking = _ReasoningFilter()
        pending = ""            # visible text not yet a whole sentence
        voice_buf = ""          # whole sentences not spoken yet (see _SPEAK_SENTENCES)
        buf_n = 0               # how many sentences are in voice_buf
        cut = False             # stopped at _MAX_SENTENCES
        stream = self._llm.chat(self._messages(), self._tools)
        try:
            async for ev in stream:
                if ev.get("text"):
                    visible = thinking.feed(ev["text"])
                    if visible and first_word is None:
                        first_word = time.monotonic() - t0
                    pending += visible
                    # Emit and speak a SENTENCE at a time — never a fragment.
                    #
                    # main.py joins transcript pieces with a space, because
                    # Gemini Live sends whole phrases. A chat API sends tokens,
                    # so emitting them as they arrive produced
                    # "Hel lo ! How c an I assis t y ou today?" in the activity
                    # log. Sentences join back together correctly, and they are
                    # The voice gets them together, not one by one: each sentence
                    # spoken on its own came out with its own flat "first
                    # sentence" intonation, and the reply sounded read out, not
                    # said. Measured on the subscription's model, waiting for
                    # the whole of an ordinary reply costs 0.16 s (median; 0.57
                    # s for a three-sentence explanation). A longer answer is
                    # not held back for that: its first _SPEAK_SENTENCES are
                    # spoken while the rest is still being written.
                    #
                    # And no answer is longer than _MAX_SENTENCES: this is a
                    # voice, heard in real time. The prompt asks for two (five
                    # when the user wants detail); past five the generation is
                    # stopped, which also frees the GPU at once.
                    ready, pending = _take_sentences(pending)
                    for sentence in ready:
                        if len(said) >= _MAX_SENTENCES:
                            cut = True
                            break
                        said.append(sentence)
                        await self._emit(Msg.said(sentence))
                        if voice_buf and len(voice_buf) + len(sentence) > _SPEAK_MAX:
                            await self._speak(voice_buf)
                            voice_buf, buf_n = "", 0
                        voice_buf = f"{voice_buf} {sentence}".strip()
                        buf_n += 1
                    if cut:
                        pending = ""
                        break
                    if buf_n >= _SPEAK_SENTENCES:
                        await self._speak(voice_buf)
                        voice_buf, buf_n = "", 0
                if ev.get("tool_calls"):
                    calls.extend(ev["tool_calls"])

            if cut:
                print(f"[Pipeline] the answer was stopped at {_MAX_SENTENCES} sentences.")
                with contextlib.suppress(Exception):
                    await stream.aclose()           # the server stops generating
            pending += thinking.flush() if not cut else ""
            if pending.strip() and len(said) >= _MAX_SENTENCES:
                pending = ""
            if pending.strip():
                said.append(pending.strip())
                if first_word is None:
                    first_word = time.monotonic() - t0
                await self._emit(Msg.said(pending.strip()))
                voice_buf = f"{voice_buf} {pending.strip()}".strip()
            if voice_buf:
                await self._speak(voice_buf)
                voice_buf = ""
            if thinking.hid:
                self._timing["thought"] = thinking.hid

            text = "".join(said)

            # A prompt-mode model answers with JSON instead of a tool-call field.
            if not calls and self.caps.get("tool_format") == "prompt":
                parsed = _parse_prompt_call(text, set(self.tool_names()))
                if parsed:
                    calls, text = [parsed], ""

            if (not calls and text and self._rounds == 0 and self._tools
                    and self.caps.get("tool_format") != "prompt"):
                calls = await self._missed_call(text)

            if calls:
                self._rounds += 1

                # A call this turn has already made is answered from what it
                # returned the first time. It never reaches main.py, so the
                # application is never launched twice — which is the difference
                # between a wasted round trip and four copies of WhatsApp.
                fresh, repeats = [], []
                for c in calls:
                    if _signature(c) in self._done_calls:
                        repeats.append(c)
                    else:
                        fresh.append(c)

                if repeats:
                    names = ", ".join(sorted({c.name for c in repeats}))
                    print(f"[Pipeline] {names} was already done this turn - "
                              f"reusing the result instead of running it again.")
                    for c in repeats:
                        self._history.append({
                            "role": "assistant", "content": None,
                            "tool_calls": [{"id": c.id, "type": "function",
                                            "function": {"name": c.name,
                                                         "arguments": json.dumps(c.args)}}]})
                        self._history.append({
                            "role": "tool", "tool_call_id": c.id, "name": c.name,
                            "content": self._done_calls[_signature(c)]})

                if self._rounds > _MAX_TOOL_ROUNDS:
                    # Wandering rather than converging. Stop asking for tools
                    # and make it answer with what it has.
                    print(f"[Pipeline] stopping after {_MAX_TOOL_ROUNDS} tool "
                              f"rounds - answering with what it has.")
                    self._history.append({
                        "role": "user",
                        "content": ("Do not call any more tools. Answer now, "
                                    "briefly, using what the tools already "
                                    "returned.")})
                    fresh = []

                if not fresh:
                    # Everything was a repeat, or the cap was reached: go
                    # straight back to the model rather than emitting nothing
                    # and leaving the turn open for ever.
                    self._start_generation()
                    return

                calls = fresh
                for c in calls:
                    self._done_calls.setdefault(_signature(c), "")

                entry: dict[str, Any] = {"role": "assistant", "content": text or None,
                                         "tool_calls": [
                                             {"id": c.id, "type": "function",
                                              "function": {"name": c.name,
                                                           "arguments": json.dumps(c.args)}}
                                             for c in calls]}
                self._history.append(entry)
                self._awaiting = {c.id: c.name for c in calls}
                self._pending_sigs = {c.id: _signature(c) for c in calls}
                # Say the acknowledgement before running the tool.
                #
                # A model that is about to use a tool usually says something
                # first — "one moment", "opening it now" — and this used to
                # return without speaking it. The words reached the log and the
                # avatar's mouth and never reached the speakers, so the
                # assistant appeared to mouth a sentence in silence and then go
                # quiet for however long the tool took. That short reply is the
                # whole point of the instant-acknowledgement behaviour; it is
                # worth more here than anywhere else, because a tool call is
                # exactly when the wait is long enough to need one.
                self._tick("llm", time.monotonic() - t0, first_word, text)
                # Already spoken, sentence by sentence, as it arrived.
                self._timing["tool_at"] = time.monotonic()
                await self._emit(Msg.calls(calls))
                # No turn_end here: the turn continues once the results arrive.
                # Emitting one would make main.py log a half-finished answer as
                # if it were the whole reply.
                return

            self._tick("llm", time.monotonic() - t0, first_word, text)
            if text:
                self._history.append({"role": "assistant", "content": text})
                # Until the sound has played out, and a few seconds more for
                # the room: what the microphone hears of these words then is
                # the assistant's own voice, not the user.
                self._echo_text = text
                self._echo_until = time.monotonic() + float(self._timing.get("spoke_s") or 0.0) + _ECHO_WINDOW_S
            spec = self._spec
            if spec is not None and not spec["confirmed"] and spec.get("gen") is asyncio.current_task():
                # Answered ahead of time and not yet confirmed: what the rest
                # of the app keeps (the shared window, the meter, the log)
                # waits for confirmation, so a cancelled attempt leaves none.
                spec["after"] = (text, time.monotonic() - t0)
            else:
                self._after_reply(text, time.monotonic() - t0)
            await self._emit(Msg.turn_end())
            if self._followup:
                # Something was injected while this answer was being produced.
                self._followup = False
                self._gen = asyncio.create_task(self._generate())

        except asyncio.CancelledError:
            raise
        except Exception as e:
            # A failed generation must still end the turn. Without this the UI
            # sits in THINKING for ever and the stall watchdog reconnects an
            # engine whose only problem was one bad request.
            if not self._report("the language model", self._llm, e):
                await self._emit(Msg.said(_failure_line("the language model", e, self._llm)))
            await self._emit(Msg.turn_end())

    async def _missed_call(self, said: str) -> list:
        """The call a reply announced and did not make (see _JUDGE_SYSTEM), or
        nothing. What the nudge writes is not spoken: the reply already was."""
        user = next((m.get("content") for m in reversed(self._history)
                     if m.get("role") == "user"), "")
        # Only after the user's own words. The app's notes ("[SCREEN_AGENT_SAYS]
        # ...", alerts, a screenshot) come in as user turns too, and a nudge
        # there would start again what is already running.
        if not isinstance(user, str) or not user.strip() or user.lstrip().startswith("["):
            return []
        try:
            verdict = _ReasoningFilter()
            seen = ""
            async for ev in self._llm.chat(
                    [{"role": "system", "content": _JUDGE_SYSTEM},
                     {"role": "user", "content": _JUDGE_ASK.format(user=user[-1500:], said=said[-800:])}],
                    None):
                seen += verdict.feed(ev.get("text") or "")
            seen += verdict.flush()
            if "YES" not in seen.strip().upper()[:12]:
                return []
            calls: list = []
            async for ev in self._llm.chat(self._messages() + [
                    {"role": "assistant", "content": said}, {"role": "user", "content": _NUDGE}],
                    self._tools):
                calls.extend(ev.get("tool_calls") or [])
            print(f"[Pipeline] the reply announced an action without the call - "
                  f"{'asked again: ' + ', '.join(c.name for c in calls) if calls else 'still no call'}")
            return calls
        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"[Pipeline] could not check the reply for a missed call: {type(e).__name__}")
            return []

    async def _speak(self, text: str) -> None:
        if self._tts is None:
            return
        if len(text) > _SPEAK_MAX:
            for piece in _pieces(text):
                await self._speak(piece)
            return
        text = _speakable(text)
        if not text.strip():
            return
        from . import audio as _a

        # A breath between sentences. The voice is made one sentence at a
        # time and the pieces were played back to back, so a three-sentence
        # answer came out as one breathless run. Never before the first
        # sentence of a turn: that one is about speed.
        if self._last_said:
            gap = _pause_after(self._last_said)
            if gap > 0:
                await self._emit(Msg.audio(bytes(int(_a.SPEAKER_RATE * gap) * _a.SAMPLE_WIDTH)))
        self._last_said = text
        # Said calmly: an exclamation mark made the voice model excited one
        # sentence and level the next. The words on screen keep it.
        text = _calm(text)

        t0 = time.monotonic()
        first = None
        total = 0
        # The language the user was heard speaking, handed to a voice that has
        # one per language. A voice that does not take the argument never sees
        # it — this is the only place that knows both halves.
        kwargs = {}
        language = getattr(self._stt, "detected_language", "")
        if language and _accepts_language(self._tts):
            kwargs["language"] = language
        try:
            async for pcm in self._tts.speak(text, **kwargs):
                if pcm:
                    if first is None:
                        first = time.monotonic() - t0
                    total += len(pcm)
                    await self._emit(Msg.audio(pcm))
        except asyncio.CancelledError:
            raise
        except Exception as e:
            # Speech failing is NOT the model failing, and saying so was wrong
            # in a way that cost real time: with a broken voice the assistant
            # reported "I could not reach the model", which sent the search to
            # the language model — the one part that was working.
            self._report("the voice", self._tts, e)
            return

        # Accumulated rather than assigned: a turn that calls a tool speaks
        # twice, and the report should show what the voice cost for the whole
        # turn rather than for whichever half finished last.
        t = self._timing
        t["tts"] = t.get("tts", 0.0) + (time.monotonic() - t0)
        if t.get("tts_first") is None:
            t["tts_first"] = first
        # Seconds of sound produced, from the byte count: 16-bit mono at the
        # speaker's rate. Worth printing next to the time it took, because
        # "1.5s to make 2.1s of speech" and "1.5s to make 0.2s" are opposite
        # situations that a duration on its own cannot tell apart.
        t["spoke_s"] = t.get("spoke_s", 0.0) + (
            total / float(_a.SPEAKER_RATE * _a.SAMPLE_WIDTH))

    # ── where the seconds went ───────────────────────────────────────────────
    # A local pipeline is three services in series, and when a turn takes
    # nineteen seconds the only useful question is which of them took them. From
    # the outside all three look identical: silence.
    #
    # Printed to the console, never to the activity log — it is diagnostic
    # output, and the log is something the user reads. Set MARK_NO_TIMING=1 to
    # silence it.
    def _tick(self, key: str, value, first=None, text: str = "") -> None:
        t = self._timing
        t[key] = t.get(key, 0.0) + float(value or 0.0)
        if first is not None and t.get(f"{key}_first") is None:
            t[f"{key}_first"] = first
        if text:
            t["chars"] = t.get("chars", 0) + len(text)

    def _print_timing(self) -> None:
        import os

        if os.environ.get("MARK_NO_TIMING"):
            self._timing = {}
            return
        t, self._timing = self._timing, {}
        if not t:
            return

        def row(label: str, secs, detail: str) -> str:
            bar = "#" * min(30, int((secs or 0) * 2))
            return f"[Pipeline]   {label:<7}{(secs or 0):>7.2f}s  {bar:<30}  {detail}"

        lines = ["[Pipeline] -- turn " + "-" * 52]
        total = 0.0
        if "stt" in t:
            total += t["stt"]
            lines.append(row("hear", t["stt"],
                             f"{t.get('heard_s', 0):.1f}s of speech"
                             f"  |  {_name_of(self._stt)}"))
        if "llm" in t:
            total += t["llm"]
            first = t.get("llm_first")
            detail = (f"first word {first:.2f}s  |  " if first else "")
            hidden = t.get("thought", 0)
            lines.append(row("think", t["llm"],
                             f"{detail}{t.get('chars', 0)} chars"
                             + (f"  |  {hidden} thought (hidden)" if hidden else "")
                             + f"  |  {_name_of(self._llm)}"))
        if "tools" in t:
            total += t["tools"]
            lines.append(row("tools", t["tools"], t.get("tool_names", "")))
        if "tts" in t:
            total += t["tts"]
            first = t.get("tts_first")
            detail = (f"first sound {first:.2f}s  |  " if first else "")
            lines.append(row("speak", t["tts"],
                             f"{detail}{t.get('spoke_s', 0):.1f}s of audio"
                             f"  |  {_name_of(self._tts)}"))
        lines.append("[Pipeline]   " + "-" * 68)
        lines.append(f"[Pipeline]   {'total':<7}{total:>7.2f}s")
        _console("\n".join(lines))

    # ── saying what actually broke ───────────────────────────────────────────
    def _report(self, what: str, provider, exc: Exception) -> bool:
        """Name the part, name the address, and put the traceback on the console.

        Two faults are being fixed here at once, both found by trying to debug a
        real failure from the activity log alone:

        THE LOG DID NOT SAY WHICH PART. Three services can fail in a local
        pipeline — recognition, the model, the voice — and every one of them
        produced the same "ConnectError: All connection attempts failed". That
        sentence names no host, so it cannot be acted on: it is equally
        consistent with Ollama being down and with ElevenLabs being unreachable.

        THE CONSOLE SAID NOTHING AT ALL. Everything here went to the activity
        log and nowhere else, so the terminal — where a person debugging this
        actually looks, and where main.py prints every other failure with a
        traceback — stayed empty.
        """
        import traceback

        if self.on_failure is not None:
            try:
                if self.on_failure(exc):
                    return True
            except Exception:
                pass
        where = _target_of(provider)
        detail = " ".join(str(exc).split())[:160] or exc.__class__.__name__
        self._log(f"SYS: {what} failed{f' ({where})' if where else ''} - "
                  f"{type(exc).__name__}: {detail}")
        _console(f"[Pipeline] {what} failed"
                 f"{f' at {where}' if where else ''}: "
                 f"{type(exc).__name__}: {detail}")
        traceback.print_exception(type(exc), exc, exc.__traceback__)
        return False

    async def _emit(self, msg: Msg) -> None:
        if self._closed:
            return
        self._gen_spoke = True
        spec = self._spec
        if spec is not None and not spec["confirmed"]:
            spec["gate"].append(msg)          # held until the turn is confirmed
            return
        await self._queue.put(msg)

    # ── answering before the silence is over ─────────────────────────────────
    # WHY
    #     The end of a turn is only known once the user has been quiet for
    #     _VAD_SILENCE_MS. Waiting that out before even starting recognition
    #     put the whole wait in front of every answer. So at _VAD_SPECULATE_MS
    #     of quiet the turn is recognised and answered straight away — and
    #     NOTHING it produces leaves the engine: words, sound and tool calls
    #     are held. When the silence completes, what is ready is released at
    #     once and the rest follows live; if the user speaks again first, it
    #     was a pause mid-sentence, and the attempt is cancelled and rolled
    #     back as if it never happened.
    #
    # WHAT IT CAN NEVER DO
    #     Run a tool. A tool call is an event like any other and is held with
    #     the rest; main.py only ever sees it after confirmation, so a
    #     cancelled attempt cannot have opened, sent or deleted anything.
    def _start_speculation(self) -> None:
        from . import audio as _a

        pcm = b"".join(self._mic)
        spec = {"gate": [], "confirmed": False, "hist": len(self._history), "text": None,
                "failed": False, "gen": None, "task": None, "pcm": pcm}
        self._spec = spec
        self._spec_resume_ms = 0.0
        self._timing = {"heard_s": _a.duration_s(pcm, _a.MIC_RATE)}
        spec["task"] = self._keep(asyncio.create_task(self._speculate(spec)))

    async def _speculate(self, spec: dict) -> None:
        t0 = time.monotonic()
        try:
            text = (await self._stt.transcribe(spec["pcm"], 16000)).strip()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            if spec["confirmed"]:
                self._report("speech recognition", self._stt, e)
            else:
                spec["failed"] = True         # the real end of turn tries again
            return
        if self._spec is not spec and not spec["confirmed"]:
            return                            # cancelled while recognising
        self._timing["stt"] = time.monotonic() - t0
        if text and self._is_own_echo(text):
            print("[Pipeline] heard the assistant's own voice - not taken as a question.")
            text = ""
        spec["text"] = text
        if not text:
            return
        await self._emit(Msg.heard(text))
        self._note_spoken(text)
        content = text
        self._history.append({"role": "user", "content": content})
        if spec["confirmed"]:
            _remember("user", content)
        self._done_calls.clear()
        self._rounds = 0
        self._last_said = ""
        self._gen = asyncio.create_task(self._generate())
        spec["gen"] = self._gen

    def _is_own_echo(self, heard: str) -> bool:
        """Is this the assistant's own last answer coming back through the
        microphone? On speakers the room carries the voice after the echo
        guard has let go, and a few words of it, recognised, became a new
        "question" that was answered again — the reply heard twice. Judged by
        the words themselves, in any language: a short recognition made mostly
        of the LAST words just said (the microphone is closed while the voice
        plays, so only its end can come back), within the time they played."""
        if not self._echo_text or time.monotonic() > self._echo_until:
            return False
        got = _words(heard)
        if not got or len(got) > _ECHO_MAX_WORDS:
            return False
        tail = set(_words(self._echo_text)[-_ECHO_TAIL_WORDS:])
        return sum(w in tail for w in got) / len(got) >= _ECHO_OVERLAP

    def _after_reply(self, text: str, took: float) -> None:
        if text:
            _remember("assistant", text)
            _meter("cheap_turn")
        print(f"[Pipeline] replied in {took:.1f}s")
        self._print_timing()

    def _cancel_speculation(self) -> None:
        spec, self._spec = self._spec, None
        if spec is None:
            return
        if spec["task"] is not None and not spec["task"].done():
            spec["task"].cancel()
        if spec["gen"] is not None and not spec["gen"].done():
            spec["gen"].cancel()
        del self._history[spec["hist"]:]
        self._awaiting, self._pending_sigs = {}, {}
        self._done_calls.clear()
        self._rounds = 0
        self._last_said = ""
        self._timing = {}

    async def _confirm_speculation(self) -> None:
        spec, self._spec = self._spec, None
        if spec is None:
            return
        if spec["failed"] or (spec["task"].done() and spec["text"] is None):
            # Recognition failed ahead of time: do it now, the ordinary way.
            await self.flush_microphone()
            return
        self._mic, self._mic_bytes = [], 0     # this audio is the turn being answered
        spec["confirmed"] = True
        if spec["text"]:
            _remember("user", spec["text"])
        if spec.get("after") is not None:
            self._after_reply(*spec["after"])
        for msg in spec["gate"]:
            if not self._closed:
                await self._queue.put(msg)

    # ── conversion ───────────────────────────────────────────────────────────
    def _to_content(self, parts: list) -> Any:
        from .llm_openai import parts_to_openai
        content = parts_to_openai(parts)
        if isinstance(content, list) and not self.caps.get("vision"):
            # The model cannot see. Keep the words rather than sending an image
            # the endpoint will reject, and say what was dropped — silently
            # losing a screenshot makes the assistant look like it ignored the
            # question.
            kept = [p["text"] for p in content if p.get("type") == "text"]
            dropped = len(content) - len(kept)
            if dropped:
                kept.append(f"[an image was attached; this model cannot see "
                            f"images, so it was not included]")
            return "\n".join(kept)
        return content

    # ── speech recognition, at the end of a turn ─────────────────────────────
    def _keep(self, task: asyncio.Task) -> asyncio.Task:
        """Hold a reference until it finishes. A task nobody keeps can be
        collected while it is still running, which is how background work
        disappears silently."""
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    async def _transcribe_later(self, slot: int, pcm: bytes, rate: int) -> None:
        """The words behind audio the model is already answering.

        Nothing waits for this. It fills in the history entry the audio went
        into — so the next turn carries text rather than a second copy of the
        sound — and gives the activity log the sentence the user said.
        """
        try:
            text = (await self._stt.transcribe(pcm, rate)).strip()
        except Exception as exc:
            self._report("speech recognition", self._stt, exc)
            text = ""
        if 0 <= slot < len(self._history):
            entry = self._history[slot]
            if entry.get("audio") is not None:
                entry.pop("audio", None)
                entry.pop("rate", None)
                # Something must stand in the history where the speech was,
                # or the turn reads as though the user said nothing at all.
                entry["content"] = text or "…"
        if text:
            await self._emit(Msg.heard(text))
            _remember("user", text)

    async def flush_microphone(self, sample_rate: int = 16000) -> str:
        """Recognise what has been buffered and start a turn with it.

        Called by whatever decides a turn has ended — push-to-talk releasing,
        or voice activity detection. Kept separate from `send_realtime_input`
        because those two decisions belong to different parts of the app, and
        the engine should not be the thing guessing when someone stopped
        talking.
        """
        if not self._mic or (self._stt is None and not self._audio_native):
            return ""
        pcm, self._mic, self._mic_bytes = b"".join(self._mic), [], 0
        from . import audio as _a

        self._timing = {"heard_s": _a.duration_s(pcm, sample_rate)}

        # A model that reads speech is asked to read the speech. Recognising it
        # first and then sending the words is two round trips for one sentence:
        # measured end to end, 1.9s of recognition in front of 1.3s of
        # thinking, for a command whose answer is "Tamam." The words are still
        # wanted — for the activity log, the shared conversation window and the
        # history this turn leaves behind — so they are fetched alongside,
        # where nobody is waiting for them.
        if self._audio_native:
            self._history.append({"role": "user", "content": "",
                                  "audio": pcm, "rate": sample_rate})
            slot = len(self._history) - 1
            if self._stt is not None:
                self._keep(asyncio.create_task(self._transcribe_later(slot, pcm, sample_rate)))
            self._start_generation()
            return ""

        t0 = time.monotonic()
        try:
            text = (await self._stt.transcribe(pcm, sample_rate)).strip()
        except Exception as e:
            self._report("speech recognition", self._stt, e)
            return ""
        self._timing["stt"] = time.monotonic() - t0
        if text and self._is_own_echo(text):
            print("[Pipeline] heard the assistant's own voice - not taken as a question.")
            self._timing = {}
            return ""
        if text:
            # THIS is what the user said, and the only thing that should reach
            # the activity log as "You:". Emitted here rather than in
            # `send_client_content`, which is also how the app injects system
            # alerts and screenshots — see the comment there.
            #
            # It also clears main.py's stall watchdog immediately, so a local
            # model taking its time never looks like a dead connection.
            await self._emit(Msg.heard(text))
            self._note_spoken(text)
            await self.send_client_content(
                turns={"role": "user", "parts": [{"text": text}]})
        return text


# ─────────────────────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────────────────────
class _ReasoningFilter:
    """Removes a reasoning model's private thinking from what gets said.

    WHY THIS IS NOT OPTIONAL FOR LOCAL MODELS
        The models people actually run at home — qwen3, deepseek-r1, and every
        distill of them — think out loud before answering, inside
        `<think>...</think>`. Several servers pass that straight through as
        ordinary content.

        Without this the assistant reads its own reasoning aloud: half a minute
        of "the user is greeting me, I should respond warmly, let me consider..."
        spoken in full, before the greeting. It also lands in the activity log,
        the session summary and the avatar's mouth, and it is stored in the
        conversation history where it costs context on every later turn.

    WHY IT IS A FILTER AND NOT A REGEX
        The text arrives in stream chunks, and a tag is regularly split across
        two of them — "<thi" then "nk>". A pattern applied per chunk sees
        neither. So the tail of each chunk is held back until it is long enough
        to be sure it is not the beginning of a tag.
    """

    _OPEN = ("<think>", "<thinking>", "<reasoning>")
    _CLOSE = ("</think>", "</thinking>", "</reasoning>")
    _HOLD = max(len(t) for t in _OPEN + _CLOSE)

    def __init__(self) -> None:
        self._buf = ""
        self._inside = False
        self.hid = 0            # characters removed, for the timing report

    def feed(self, chunk: str) -> str:
        """Whatever of `chunk` is safe to say out loud."""
        self._buf += chunk
        out = []
        while True:
            if self._inside:
                idx, tag = _first_of(self._buf, self._CLOSE)
                if idx < 0:
                    break
                self.hid += idx + len(tag)
                self._buf = self._buf[idx + len(tag):]
                self._inside = False
                continue
            idx, tag = _first_of(self._buf, self._OPEN)
            if idx < 0:
                break
            out.append(self._buf[:idx])
            self._buf = self._buf[idx + len(tag):]
            self._inside = True

        # Hold back enough characters to recognise a tag that is still arriving,
        # WHETHER OR NOT we are inside a block.
        #
        # The first version cleared the buffer outright while inside, on the
        # reasoning that all of it was hidden anyway. It is — but the tail of it
        # may be the first half of the CLOSING tag, and dropping that means the
        # block never ends and the entire rest of the answer is swallowed.
        # Measured against a stream split character by character, the whole reply
        # disappeared.
        keep = self._HOLD - 1
        if self._inside:
            if len(self._buf) > keep:
                self.hid += len(self._buf) - keep
                self._buf = self._buf[-keep:]
            return "".join(out)

        if len(self._buf) > keep:
            out.append(self._buf[:-keep])
            self._buf = self._buf[-keep:]
        return "".join(out)

    def flush(self) -> str:
        """The tail, once the stream has ended."""
        if self._inside:
            # An unterminated block. Dropping it is right: it is reasoning that
            # never reached a conclusion, and saying it is worse than silence.
            self.hid += len(self._buf)
            self._buf = ""
            return ""
        tail, self._buf = self._buf, ""
        return tail


def _first_of(text: str, tags) -> tuple:
    """The earliest of several tags in `text`, as (index, tag)."""
    best, found = -1, ""
    for tag in tags:
        i = text.find(tag)
        if i >= 0 and (best < 0 or i < best):
            best, found = i, tag
    return best, found


# The same two rules the voice uses to split — a space is required after `.!?`
# so version numbers survive, and not after the CJK, Arabic and Devanagari
# stops, which are not written with one.
# Own-echo guard (PipelineEngine._is_own_echo): a recognition of at most
# _ECHO_MAX_WORDS words, at least _ECHO_OVERLAP of them from the answer that was
# just spoken, within its playing time + _ECHO_WINDOW_S, is the assistant
# hearing itself. A real reply ("yes", "no, the other one") rarely repeats the
# answer's own words that closely; a long one is never judged an echo.
_ECHO_WINDOW_S = 6.0
_ECHO_MAX_WORDS = 10
_ECHO_OVERLAP = 0.7
_ECHO_TAIL_WORDS = 8
_WORD = re.compile(r"\w+", re.UNICODE)


def _words(text: str) -> list:
    return [w.casefold() for w in _WORD.findall(str(text or ""))]


# A spoken answer is at most this many sentences (the prompt asks for two,
# up to five when the user wants detail); the generation stops there.
_MAX_SENTENCES = 5
# Up to this many sentences go to the voice in one piece; a longer answer
# starts speaking once it has them, and the rest follows.
_SPEAK_SENTENCES = 3
# A safety cap per request, not a target: five short sentences are 300-450
# characters, a sentence without a full stop can be longer.
_SPEAK_MAX = 600


def _pieces(text: str, limit: int = _SPEAK_MAX) -> list:
    """`text` cut at spaces into pieces of at most `limit` characters (a
    sentence without a full stop can be longer than the voice takes)."""
    out, cur = [], ""
    for word in text.split():
        if cur and len(cur) + 1 + len(word) > limit:
            out.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}" if cur else word
        while len(cur) > limit:                 # one "word" longer than the limit
            out.append(cur[:limit])
            cur = cur[limit:]
    if cur:
        out.append(cur)
    return out

# Seconds of silence after a sentence, by how it ended: a question or an
# exclamation leaves a little more room than a statement, as a person does.
_PAUSE_STATEMENT = 0.18
_PAUSE_QUESTION = 0.26


_EXCLAIM = re.compile(r"[!！]+")
_EXCLAIM_OPEN = re.compile(r"¡")


def _calm(text: str) -> str:
    """The text the voice reads: exclamations as plain statements."""
    return _EXCLAIM_OPEN.sub("", _EXCLAIM.sub(".", text))


def _pause_after(sentence: str) -> float:
    end = sentence.rstrip().rstrip("\"'”’»)]").rstrip()[-1:]
    if end in "?!？！؟":
        return _PAUSE_QUESTION
    return _PAUSE_STATEMENT


_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+|(?<=[。！？؟۔।])\s*")


# Characters that belong in writing but not in speech.
#
# A model ends a friendly sentence with an emoji, and espeak — which is what
# Piper and every system voice phonemise with — has a NAME for each one. So
# "How can I help? 😊" is read out as "How can I help? smiling face with
# smiling eyes", which is both absurd and, at three seconds, longer than the
# sentence it decorates.
#
# Removed on the way to the voice ONLY. The emoji stays in the activity log and
# in the conversation history, because the model did write it and the person
# reading the log should see what was written.
#
# Ranges, not a list: a list of emoji is out of date the year it is written, and
# this has to work for whatever the model reaches for.
# EVERY RANGE HERE IS NARROW ON PURPOSE, AND THAT IS THE WHOLE LESSON.
#
# The first version ended with the range U+24C2-U+1F251, which is what most
# "strip emoji" snippets use. It spans a hundred and twenty thousand code
# points - including all of CJK. Tested with one Japanese sentence, it removed
# the sentence:
#
#     '日本語も大丈夫です。'  ->  ''
#
# An assistant that falls silent in Japanese, Chinese and Korean, in the name of
# not reading an emoji aloud. Nothing but a multilingual test would have caught
# it, and a range copied without checking is exactly how a project written in
# one language quietly breaks in every other.
_UNSPEAKABLE = re.compile(
    "["
    "🀀-🫿"     # emoji, pictographs, flags, skin tones
    "☀-➿"     # misc symbols and dingbats
    "←-⇿"     # arrows
    "⬀-⯿"     # shapes and more arrows
    "︀-️"     # variation selectors
    "‍"                 # zero-width joiner, which welds them together
    "™ℹ⁉‼"   # tm, info, !?, !!
    "]+", flags=re.UNICODE)


def _signature(call) -> str:
    """What makes two tool calls the same request.

    Name plus arguments, with the arguments sorted so key order cannot make an
    identical call look new. The call id is deliberately NOT part of it — the
    model issues a fresh id every time it repeats itself, which is exactly the
    case this exists to catch."""
    try:
        args = json.dumps(call.args or {}, sort_keys=True, ensure_ascii=False)
    except Exception:
        args = str(getattr(call, "args", ""))
    return f"{call.name}({args})"


def _speakable(text: str) -> str:
    """What is left of a reply once the parts a voice cannot say are gone."""
    cleaned = _UNSPEAKABLE.sub(" ", text or "")
    # Collapse the gaps the removal leaves, without touching line structure.
    return re.sub(r"[ 	]{2,}", " ", cleaned).strip()


def _take_sentences(buf: str) -> tuple:
    """(complete sentences, whatever is left over).

    Only complete ones are returned, so a sentence is never emitted twice and a
    half-formed one is never spoken. A buffer that has grown long without any
    punctuation at all — which happens, in several languages — is released at
    the last space rather than held for ever."""
    out = []
    while True:
        m = _SENTENCE_END.search(buf)
        if not m:
            break
        piece = buf[:m.end()].strip()
        if piece:
            out.append(piece)
        buf = buf[m.end():]
    if len(buf) > 320:
        cut = buf.rfind(" ", 0, 320)
        if cut > 0:
            out.append(buf[:cut].strip())
            buf = buf[cut:].lstrip()
    return out, buf


def _accepts_language(tts) -> bool:
    """Whether this voice can be told what language to speak.

    Asked rather than assumed, because the three voices differ and all three are
    valid: ElevenLabs detects it from the text itself, the system voice cannot
    change language at all, and Piper has one model per language and needs to be
    told. Introspection keeps that a property of each provider rather than a
    list kept here that goes stale."""
    import inspect
    try:
        return "language" in inspect.signature(tts.speak).parameters
    except Exception:
        return False


def _console(text: str) -> None:
    """Print, on a console that may not be able to spell.

    THIS PROJECT HAS ALREADY BEEN KILLED BY THIS ONCE. A status line carrying an
    emoji raised UnicodeEncodeError on a non-UTF-8 console - cp1254 on a Turkish
    Windows, cp1251 on a Russian one, cp932 on a Japanese one - and because the
    print sat outside the tool's own try/except it escaped into the receive loop
    and took the session down. Mark LV fixed that occurrence.

    The first version of the timing report brought it straight back, with
    box-drawing characters, and made it worse: this print sits INSIDE
    `_generate`'s try, so the failure was caught and reported as "the language
    model failed". A diagnostic that blames the wrong component is worse than no
    diagnostic, and it was the tests that noticed rather than a user.

    So the report is ASCII, and this is the second line of defence - an
    exception's own text can contain any character at all.
    """
    try:
        print(text)
    except Exception:
        try:
            enc = getattr(sys.stdout, "encoding", None) or "ascii"
            print(text.encode(enc, "replace").decode(enc, "replace"))
        except Exception:
            pass                # a message is never worth an exception


def _name_of(provider) -> str:
    """What to call a provider in a timing line — the model or voice actually in
    use, because "think 16.2s" is only actionable once you know which model."""
    if provider is None:
        return "none"
    # `last_voice` first: a language-following voice picks per turn, and the
    # report has to name what actually spoke rather than what was configured.
    for attr in ("last_voice", "model", "model_name", "voice"):
        value = getattr(provider, attr, "")
        if value:
            extra = ""
            if getattr(provider, "device", ""):
                extra = f"/{provider.device}"
            return f"{value}{extra}"
    engine = getattr(provider, "_engine", "")
    return str(engine) if engine else provider.__class__.__name__


def _target_of(provider) -> str:
    """Where a provider was pointed, for an error message that can be acted on.

    "All connection attempts failed" is httpx's wording and it names no host, so
    on its own it cannot tell anybody whether their local server is down or a
    hosted one is unreachable. The provider knows; it just was not being asked.
    """
    for attr in ("base", "base_url", "endpoint"):
        value = getattr(provider, attr, "")
        if value:
            return str(value)
    for attr in ("voice_id", "model", "model_name"):
        value = getattr(provider, attr, "")
        if value:
            return str(value)
    return provider.__class__.__name__ if provider is not None else ""


# The parts a failure can name, as the user hears them (core/i18n keys).
_PART_KEYS = {"the language model": "voice.part.language_model",
              "speech recognition": "voice.part.speech_recognition",
              "the voice": "voice.part.voice"}


def _ours(provider) -> bool:
    """The part that failed is the subscription's own server (the relay), not
    one the user set up: core/service's address."""
    try:
        from core import service
        base = service.base_url().rstrip("/")
    except Exception:
        return False
    return bool(base) and provider is not None and _target_of(provider).rstrip("/").startswith(base)


def _failure_line(what: str, exc: Exception, provider=None) -> str:
    """What the assistant SAYS when a part of itself is broken.

    Spoken out loud, so it is a sentence rather than a stack trace — but it says
    which part, because "please try again" on a server that is not running is
    advice that cannot work and wastes the listener's time."""
    name = type(exc).__name__.lower()
    unreachable = "connect" in name or "timeout" in name
    # Spoken, so in the user's language when the app is here to translate it
    # (core.i18n). Which part broke stays in the log, in English, where it is
    # read by whoever fixes it. Without the app, the English line as before.
    try:
        from core.i18n import t
        # A subscriber cannot fix an address or start a server: when it is
        # OUR server that failed, tell them whose problem it is. Our own
        # servers' faults reach the owner by themselves (backend/gpu_watch.py).
        if _ours(provider):
            return t("voice.failure_service_unreachable" if unreachable else "voice.failure_service")
        part = t(_PART_KEYS[what]) if what in _PART_KEYS else what
        return t("voice.failure_unreachable" if unreachable else "voice.failure_other", part=part)
    except Exception:
        pass
    if unreachable:
        return (f"I could not reach {what}. It looks like the server is not "
                f"running or the address is wrong — the details are in the log.")
    return f"Something went wrong with {what}. The details are in the log."


def _pcm_of(audio: Any) -> bytes:
    """The raw samples out of whatever `send_realtime_input` was handed.

    THIS IS WHERE THE ENGINE FIRST MET THE REAL CALLER AND LOST.
        main.py sends `audio=types.Blob(data=..., mime_type=...)` — an SDK
        object, because that is what the Live API wants. Every test here had
        passed a plain dict, so the engine read a dict, got a Blob, took `len()`
        of it, and raised.

        That exception is not survivable where it happens: `_send_realtime` is a
        member of the session's TaskGroup, so one TypeError cancels the receive
        loop, the microphone, playback and every monitor with it, and the connect
        loop starts over. The symptom is an assistant that reconnects every few
        seconds, reloads its speech model each time, and comes back asleep — with
        nothing in the log to say why.

        The lesson is not "handle Blob". It is that a contract written against
        the caller has to be tested against the caller.
    """
    if audio is None:
        return b""
    if isinstance(audio, (bytes, bytearray, memoryview)):
        return bytes(audio)
    if isinstance(audio, dict):
        return bytes(audio.get("data") or b"")
    data = getattr(audio, "data", None)          # types.Blob and anything like it
    if isinstance(data, (bytes, bytearray, memoryview)):
        return bytes(data)
    if isinstance(data, str):
        # Some SDK paths carry base64 rather than raw bytes.
        import base64
        try:
            return base64.b64decode(data)
        except Exception:
            return b""
    return b""


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        return "\n".join(p.get("text", "") for p in content
                         if isinstance(p, dict) and p.get("type") == "text").strip()
    return ""


def _stringify(value: Any) -> str:
    """A tool's result, as the string a chat API expects."""
    if isinstance(value, str):
        return value
    if value is None:
        return "ok"
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        return str(value)


def _parse_prompt_call(text: str, known: set[str]) -> Optional[FunctionCall]:
    """A tool call out of a prompt-mode model's reply.

    The model was asked for {"tool": ..., "args": {...}} and nothing else, and
    it will often wrap it in a code fence or a sentence anyway. The name is
    checked against the tools actually declared: a model that invents one is
    answering, not calling, and the reply should reach the user as words.
    """
    if not text or "{" not in text:
        return None
    chunk = text[text.find("{"):text.rfind("}") + 1]
    try:
        obj = json.loads(chunk)
    except Exception:
        return None
    if not isinstance(obj, dict):
        return None
    name = str(obj.get("tool") or obj.get("name") or "").strip()
    if name not in known:
        return None
    args = obj.get("args") if isinstance(obj.get("args"), dict) else {}
    return FunctionCall(id=f"call_{name}", name=name, args=args)


def _make(cfg: dict) -> PipelineEngine:
    """Built from the ids stored in the panel, resolved through the registry."""
    from memory.config_manager import get_provider_config
    from . import registry

    def part(slot: str):
        pid = str(cfg.get(slot) or "").strip()
        return registry.build(pid, get_provider_config(pid)) if pid else None

    llm = part("llm")
    if llm is None:
        raise RuntimeError("the pipeline engine needs a language model")
    return PipelineEngine(llm, part("stt"), part("tts"),
                          config=cfg.get("live_config"), log=cfg.get("log"))


PROVIDER = {
    "id": "pipeline",
    "kind": "engine",
    "label": "Local pipeline",
    "hint": "Speech recognition, a language model and speech synthesis, assembled",
    "requires": (),
    "needs": (),        # its parts are chosen in the panel, not typed in
    "caps": {},         # taken from whatever fills its slots
    "factory": _make,
}
