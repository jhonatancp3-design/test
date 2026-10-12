"""Short sentences the assistant says over and over, kept instead of rebought.

WHY
    An assistant that does things says the same handful of sentences all day:
    "Tamam", "Açtım", "Gönderildi", "Bir saniye". Measured on the hosted voices,
    each one of those costs half a second of waiting and a fraction of a cent,
    every single time, for audio that is identical to the last time.

    Kept on disk, the second one arrives in about a millisecond and costs
    nothing. Measured here: 0.009s on the first read, 0.001s after that,
    against 0.51s from ElevenLabs and 2.5 to 4.5s from Gemini's voice.

WHAT IS AND IS NOT KEPT
    Only short lines — a long answer is unique, and keeping it would fill a
    disk to buy nothing. The key includes the provider, the model and the
    voice, so changing any of them produces new entries rather than playing
    back the old voice. The store is capped and the least recently used entries
    go first.

WHY IT IS NOT IN EITHER PROVIDER
    Both of them want it and neither of them owns it. A copy in each is two
    caches with one bug fixed in one of them.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from core.paths import APP_ID

# What counts as a repeatable line. Longer than this is an answer, not a
# confirmation.
MAX_CHARS = 120
MAX_FILES = 400


def folder() -> Path:
    """Where the lines are kept (not created). "Delete my data" removes it
    too (core/footprint.sweep): the lines are things the assistant said."""
    return Path(os.getenv("LOCALAPPDATA") or tempfile.gettempdir()) / f"{APP_ID}VoiceCache"


def _dir() -> Path:
    d = folder()
    d.mkdir(parents=True, exist_ok=True)
    return d


def path_for(text: str, *parts: str, enabled: bool = True) -> Path | None:
    """Where this line would be kept, or None when it should not be."""
    said = (text or "").strip()
    if not enabled or not said or len(said) > MAX_CHARS:
        return None
    key = hashlib.sha256("|".join([*parts, said]).encode("utf-8")).hexdigest()[:32]
    return _dir() / f"{key}.pcm"


def read(path: Path | None) -> bytes | None:
    """The kept audio, or None. Touches the file so the tidy-up keeps the
    lines that are actually being used."""
    if path is None or not path.exists():
        return None
    try:
        data = path.read_bytes()
        os.utime(path, None)
        return data or None
    except Exception:
        return None          # a broken entry is a miss, never an error


def write(path: Path | None, audio: bytes) -> None:
    if path is None or not audio:
        return
    try:
        path.write_bytes(audio)
        _tidy()
    except Exception:
        pass                 # a cache that cannot be written is still a cache


def _tidy() -> None:
    try:
        files = sorted(_dir().glob("*.pcm"), key=lambda p: p.stat().st_atime)
        for old in files[:-MAX_FILES]:
            old.unlink(missing_ok=True)
    except Exception:
        pass
