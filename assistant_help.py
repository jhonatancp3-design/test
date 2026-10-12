"""
assistant_help — the assistant knows what it is, and can look at itself.

WHY
    Asked "what are you?", "how do I connect my phone?" or "why don't you
    answer?", a model on its own says it is "just an AI assistant" — it knows
    nothing of this product. This tool gives it the product's own manual
    (manual/<lang>.md, shipped with the program) and, on request, a live look at
    this computer: the connection to the subscription's servers, the plan and
    what is left of it, the phone link, Remote Mode, the wake word, the plugins
    ON and OFF, the most recent errors.

    The manual is not put into the system prompt: it is long, it would be read
    on every turn, and it would push the rules that matter out of attention.
    The model fetches the sections a question needs.

NOTHING SECRET leaves through it: no key, no licence, no token, no address of
the user's — and the recent errors are the log's own lines, which never carry
those (core/logfile).
"""
from __future__ import annotations

import re
from pathlib import Path

from core.paths import resource

_WORD = re.compile(r"[\w']+", re.UNICODE)
_MAX_SECTIONS = 3


def manual_text(lang: str = "") -> str:
    """The manual in `lang` when there is one, else English; {product} filled in."""
    folder = resource("manual")
    path = folder / f"{(lang or 'en').split('-')[0].lower()}.md"
    if not path.exists():
        path = folder / "en.md"
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return ""
    try:
        from core import edition
        product = edition.name()
    except Exception:
        product = "the assistant"
    return text.replace("{product}", product)


def sections(text: str) -> list[tuple[str, str]]:
    """[(heading, body)] — one per "## " heading."""
    out = []
    for block in re.split(r"(?m)^## ", text)[1:]:
        head, _, body = block.partition("\n")
        out.append((head.strip(), body.strip()))
    return out


def find(topic: str, text: str, limit: int = _MAX_SECTIONS) -> list[tuple[str, str]]:
    """The sections that share the most words with `topic` (the heading counts
    three times). An empty topic: what the assistant is."""
    secs = sections(text)
    words = {w.lower() for w in _WORD.findall(topic or "") if len(w) > 2}
    if not words:
        return secs[:2]
    scored = []
    for i, (head, body) in enumerate(secs):
        hw = {w.lower() for w in _WORD.findall(head)}
        bw = {w.lower() for w in _WORD.findall(body)}
        score = 3 * len(words & hw) + len(words & bw)
        if score:
            scored.append((score, -i, head, body))
    scored.sort(reverse=True)
    return [(h, b) for _s, _i, h, b in scored[:limit]] or secs[:2]


# ── a look at this computer ──────────────────────────────────────────────────

def _line(label: str, fn) -> str:
    try:
        return f"- {label}: {fn()}"
    except Exception as e:
        return f"- {label}: could not be read ({type(e).__name__})"


def _subscription() -> str:
    from core import service
    if not service.linked():
        return "not signed in to a subscription (this install uses its own models and keys)"
    from core import credits
    st = credits.status()
    if st is None:
        return "signed in, but the subscription's servers did NOT answer just now (internet, or our side)"
    return f"signed in, servers answering; plan {st.get('plan') or '?'}; {credits.describe()}"


def _phone() -> str:
    from core import phone
    lk = phone.link()
    if not lk:
        return "no phone connected"
    return f"connected (bot @{lk.get('bot_username') or '?'})"


def _remote() -> str:
    from core import remotemode
    st = remotemode.settings()
    on = lambda k: "ON" if remotemode.allowed(k) else "OFF"       # noqa: E731
    parts = [f"mode {remotemode.panel_mode()}",
             f"phone control (master) {on('phone_control')}",
             f"phone requests {on('accept_commands')}",
             f"screenshots {on('allow_screen')}",
             f"files from the phone {on('files_in')}, to the phone {on('files_out')}",
             f"risky steps from the phone confirmed {remotemode.confirm_count()}x"]
    if remotemode.plug():
        parts.append(f"smart plug: {remotemode.plug().get('kind')}")
    elif st.get("plug_manual"):
        parts.append("smart plug: switched from its own app")
    elif st.get("wake_url"):
        parts.append("smart plug: web link")
    parts.append("can be switched on after a full shutdown" if remotemode.power_on_ready()
                 else "can NOT be switched on from the phone after a full shutdown")
    parts.append("hibernation available" if remotemode.hibernate_available() else "hibernation off")
    return "; ".join(parts)


def _wake() -> str:
    from core import wake_word
    m = wake_word.model()
    return f"'{m['phrase']}' ready" if m else "no wake model installed"


def _plugins() -> str:
    from core import plugin_loader
    reg = plugin_loader.current()
    if reg is None:
        return "not loaded yet"
    rows = [r for r in reg.list_for_ui() if r["valid"]]
    on = sorted(r["label"] for r in rows if r["enabled"])
    off = sorted(r["label"] for r in rows if not r["enabled"])
    broken = [r["name"] for r in reg.list_for_ui() if not r["valid"]]
    out = f"ON: {', '.join(on) or 'none'}. OFF: {', '.join(off) or 'none'}."
    return out + (f" Could not load: {', '.join(broken)}." if broken else "")


def _recent_errors(limit: int = 8) -> str:
    from core import logfile
    path = logfile.path()
    if not path or not Path(path).exists():
        return "no log"
    with open(path, "rb") as f:
        f.seek(0, 2)
        f.seek(max(0, f.tell() - 200_000))
        tail = f.read().decode("utf-8", "replace").splitlines()
    hits = [ln.strip()[:220] for ln in tail
            if re.search(r"\b(failed|error|exception|refused|not reach|unreachable)\b", ln, re.I)]
    return ("\n  " + "\n  ".join(hits[-limit:])) if hits else "none recently"


def diagnose() -> str:
    return "\n".join([
        "LIVE CHECK OF THIS COMPUTER:",
        _line("Subscription", _subscription),
        _line("Phone", _phone),
        _line("Remote Mode", _remote),
        _line("Wake word", _wake),
        _line("Plugins", _plugins),
        _line("Recent errors in the log", _recent_errors),
    ])


def assistant_help(parameters: dict | None = None, **_ctx) -> str:
    p = parameters or {}
    topic = str(p.get("topic") or "").strip()
    try:
        from core import i18n
        lang = i18n.current()
    except Exception:
        lang = "en"
    text = manual_text(lang)
    parts = []
    if text:
        for head, body in find(topic, text):
            parts.append(f"## {head}\n{body}")
    if p.get("diagnose"):
        parts.append(diagnose())
    parts.append("Answer the user from this, in their language, briefly and concretely: name the exact "
                 "button or setting. Do not paste this text.")
    return "\n\n".join(parts)


TOOL = {
    "name": "assistant_help",
    "description": (
        "Your own manual and a live check of yourself. Call it whenever the user asks about YOU — what "
        "you are, who made you, what you can do, how a feature or a plugin is set up, where a setting is, "
        "privacy, the subscription and its limits, the phone, switching the computer on and off remotely — "
        "and whenever they say something does not work (then with diagnose=true). Never answer such "
        "questions from general knowledge or say you are 'just an AI'."),
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "topic": {"type": "STRING",
                      "description": "What the user asks about, in a few English words (e.g. 'connect phone', "
                                     "'microphone not hearing', 'privacy', 'plans and limits')."},
            "diagnose": {"type": "BOOLEAN",
                         "description": "true when something is not working: adds a live check of this "
                                        "computer (servers, plan, phone, Remote Mode, wake word, plugins, "
                                        "recent errors)."},
        },
    },
    "handler": assistant_help,
}
