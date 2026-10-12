"""
Is someone speaking? Silero VAD (MIT), on onnxruntime, on the CPU.

WHY A MODEL AND NOT LOUDNESS
    The pipeline engine first decided "speech" from loudness against a moving
    noise floor. That is cheap and language-free, and it cannot tell a voice
    from a door, a cough, a keyboard or music — anything loud enough counts,
    and a quiet speaker in a noisy room never stops "talking". A turn boundary
    built on it has to wait long to be safe.

    Silero tells speech from non-speech in any language (it was trained on
    many), so the end of a turn can be called sooner without cutting people
    off. Measured here: 0.19 ms of CPU per 32 ms of audio, a 2.3 MB model
    shipped in models/vad/, and onnxruntime is already what the wake word runs
    on — nothing new to install.

WHEN IT IS NOT THERE
    `load()` returns None (no onnxruntime, no model file, a model that will not
    load) and the engine keeps its loudness rule. A missing optional piece
    degrades the assistant; it never breaks it.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

RATE = 16_000
CHUNK = 512            # samples per decision at 16 kHz (32 ms) — the model's own size
CONTEXT = 64           # samples of the previous chunk the model expects in front


def _model_path() -> Path:
    try:
        from core.paths import resource
        return Path(resource("models/vad/silero_vad.onnx"))
    except Exception:
        return Path(__file__).resolve().parents[1] / "models" / "vad" / "silero_vad.onnx"


class SileroVAD:
    """Speech probability per 32 ms, fed any amount of 16 kHz PCM16 at a time."""

    def __init__(self, path: Path):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        # One thread: this runs beside audio playback and an avatar, and one
        # core does it in a fraction of a millisecond.
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self._s = ort.InferenceSession(str(path), sess_options=opts,
                                       providers=["CPUExecutionProvider"])
        self._sr = np.array(RATE, dtype=np.int64)
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros(CONTEXT, dtype=np.float32)
        self._pending = np.zeros(0, dtype=np.float32)

    def feed(self, samples: np.ndarray) -> list[float]:
        """Probabilities for every whole 32 ms chunk now available."""
        buf = np.concatenate([self._pending, samples.astype(np.float32, copy=False)])
        out = []
        n = len(buf) // CHUNK
        for i in range(n):
            chunk = buf[i * CHUNK:(i + 1) * CHUNK]
            x = np.concatenate([self._context, chunk])[None, :]
            prob, self._state = self._s.run(None, {"input": x, "state": self._state, "sr": self._sr})
            self._context = chunk[-CONTEXT:]
            out.append(float(prob[0][0]))
        self._pending = buf[n * CHUNK:]
        return out


def load() -> Optional[SileroVAD]:
    """A ready detector, or None to keep the loudness rule."""
    try:
        from importlib.util import find_spec
        path = _model_path()
        if find_spec("onnxruntime") is None or not path.is_file():
            return None
        return SileroVAD(path)
    except Exception as e:
        print(f"[VAD] Silero unavailable ({type(e).__name__}: {e}) - using loudness")
        return None
