import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from datetime import timedelta
from pathlib import Path

from core.footprint import launchd_label, task_path, unit_name

_CNW: dict = (
    {"creationflags": subprocess.CREATE_NO_WINDOW}
    if platform.system() == "Windows" else {}
)

def _get_os() -> str:
    _sys = platform.system()
    if _sys == "Darwin":
        return "mac"
    if _sys == "Linux":
        return "linux"
    return "windows"


def _title() -> str:
    """The notification's title: the assistant's own name, as the user set it."""
    try:
        from memory.config_manager import get_assistant_name
        name = (get_assistant_name() or "").strip()
    except Exception:
        name = ""
    return f"{name} — Reminder" if name else "Reminder"


# ── What the OS scheduler runs ────────────────────────────────────────────────
# The program itself with `--notify <payload>` (core/notify.py), not a generated
# Python script: a packaged build has neither Python nor .py files to run, and
# every reminder set from it used to be registered and then never shown.

# A reminder the computer missed (it was off) still shows when it is next on,
# but not forever: past this it is stale, and the task deletes itself instead
# of piling up in the Task Scheduler.
_LATE_LIMIT = timedelta(days=1)


def _schedule_windows(target_dt: datetime, task_name: str,
                      program: str, args: list[str]) -> str:
    from xml.sax.saxutils import escape

    xml_path = _payload_dir() / f"{task_name}.xml"
    arguments = subprocess.list2cmdline(args)
    xml_content = (
        '<?xml version="1.0" encoding="UTF-16"?>\n'
        '<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">\n'
        f'  <RegistrationInfo><Description>{escape(_title())}</Description></RegistrationInfo>\n'
        '  <Triggers><TimeTrigger>\n'
        f'    <StartBoundary>{target_dt.strftime("%Y-%m-%dT%H:%M:%S")}</StartBoundary>\n'
        f'    <EndBoundary>{(target_dt + _LATE_LIMIT).strftime("%Y-%m-%dT%H:%M:%S")}</EndBoundary>\n'
        '    <Enabled>true</Enabled>\n'
        '  </TimeTrigger></Triggers>\n'
        '  <Actions><Exec>\n'
        f'    <Command>{escape(program)}</Command>\n'
        f'    <Arguments>{escape(arguments)}</Arguments>\n'
        '  </Exec></Actions>\n'
        '  <Settings>\n'
        '    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>\n'
        '    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>\n'
        '    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>\n'
        '    <StartWhenAvailable>true</StartWhenAvailable>\n'
        '    <ExecutionTimeLimit>PT5M</ExecutionTimeLimit>\n'
        '    <Enabled>true</Enabled>\n'
        '    <DeleteExpiredTaskAfter>PT0S</DeleteExpiredTaskAfter>\n'
        '  </Settings>\n'
        '  <Principals><Principal>\n'
        '    <LogonType>InteractiveToken</LogonType>\n'
        '    <RunLevel>LeastPrivilege</RunLevel>\n'
        '  </Principal></Principals>\n'
        '</Task>'
    )
    xml_path.write_text(xml_content, encoding="utf-16")

    result = subprocess.run(
        ["schtasks", "/Create", "/TN", task_path(task_name), "/XML", str(xml_path), "/F"],
        capture_output=True, text=True, encoding="utf-8", errors="replace", **_CNW,
    )
    try:
        xml_path.unlink(missing_ok=True)
    except Exception:
        pass

    if result.returncode != 0:
        err = (result.stderr or result.stdout).strip()
        print(f"[Reminder] ❌ schtasks: {err}")
        return ""
    return task_path(task_name)


def _schedule_mac(target_dt: datetime, task_name: str,
                  program: str, args: list[str]) -> str:
    from xml.sax.saxutils import escape

    agents_dir = Path.home() / "Library" / "LaunchAgents"
    agents_dir.mkdir(parents=True, exist_ok=True)

    label      = launchd_label(task_name)
    plist_path = agents_dir / f"{label}.plist"
    argv_xml   = "\n".join(f"    <string>{escape(a)}</string>" for a in (program, *args))

    plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>             <string>{label}</string>
  <key>ProgramArguments</key>
  <array>
{argv_xml}
  </array>
  <key>StartCalendarInterval</key>
  <dict>
    <key>Year</key>   <integer>{target_dt.year}</integer>
    <key>Month</key>  <integer>{target_dt.month}</integer>
    <key>Day</key>    <integer>{target_dt.day}</integer>
    <key>Hour</key>   <integer>{target_dt.hour}</integer>
    <key>Minute</key> <integer>{target_dt.minute}</integer>
  </dict>
  <key>RunAtLoad</key>         <false/>
  <key>StandardOutPath</key>   <string>/dev/null</string>
  <key>StandardErrorPath</key> <string>/dev/null</string>
</dict>
</plist>
"""
    plist_path.write_text(plist_content, encoding="utf-8")
    plist_path.chmod(0o644)

    result = subprocess.run(
        ["launchctl", "load", str(plist_path)],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if result.returncode != 0:
        plist_path.unlink(missing_ok=True)
        print(f"[Reminder] ❌ launchctl: {result.stderr.strip()}")
        return ""
    return label


def _schedule_linux(target_dt: datetime, task_name: str,
                    program: str, args: list[str]) -> str:
    if shutil.which("systemd-run"):
        on_calendar = target_dt.strftime("%Y-%m-%d %H:%M:00")
        result = subprocess.run(
            ["systemd-run", "--user", f"--on-calendar={on_calendar}",
             f"--unit={unit_name(task_name)}", "--", program, *args],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        if result.returncode == 0:
            return task_name
        print(f"[Reminder] ⚠️ systemd-run failed: {result.stderr.strip()}, trying 'at'")

    if shutil.which("at"):
        import shlex
        at_time = target_dt.strftime("%H:%M %Y-%m-%d")
        cmd_str = shlex.join([program, *args]) + "\n"
        result  = subprocess.run(
            ["at", at_time],
            input=cmd_str, capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        if result.returncode == 0:
            return task_name
        print(f"[Reminder] ❌ at: {result.stderr.strip()}")
        return ""

    print("[Reminder] ❌ Neither systemd-run nor at found on this Linux system.")
    return ""


def _payload_dir() -> Path:
    from core.notify import _payload_dir as payload_dir
    return payload_dir()


def reminder(
    parameters: dict,
    response=None,
    player=None,
    session_memory=None,
) -> str:

    date_str = parameters.get("date", "").strip()
    time_str = parameters.get("time", "").strip()
    message  = parameters.get("message", "Reminder").strip()

    if not date_str or not time_str:
        return "I need both a date and a time to set a reminder."

    try:
        target_dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
    except ValueError:
        return "I couldn't parse that date or time. Please use YYYY-MM-DD and HH:MM."

    if target_dt <= datetime.now():
        return "That time has already passed — I can't set a reminder in the past."

    os_name    = _get_os()
    # Named after the product, inside its own scheduler folder
    # (core/footprint), so the uninstaller finds every one of them.
    task_name  = f"reminder-{target_dt.strftime('%Y%m%d_%H%M%S')}"

    # The phone copy is made first and independently: if this machine's
    # scheduler fails, the reminder still reaches the phone, and the other way
    # round. Nothing happens here unless the cloud agent is paired.
    phone = _mirror_to_cloud(message, target_dt)
    # The assistant also says it aloud at that moment when it is running — a
    # routine (core/routines), independent of the system notification below.
    phone += _add_spoken_routine(message, target_dt)

    from core import notify
    try:
        payload = notify.write_payload(_title(), message)
    except Exception as e:
        return f"Could not prepare the reminder: {e}{phone}"
    program, args = notify.command(payload)

    try:
        if os_name == "windows":
            job_id = _schedule_windows(target_dt, task_name, program, args)
        elif os_name == "mac":
            job_id = _schedule_mac(target_dt, task_name, program, args)
        else:
            job_id = _schedule_linux(target_dt, task_name, program, args)
    except Exception as e:
        print(f"[Reminder] ❌ Scheduling exception: {e}")
        job_id = ""
    if not job_id:
        payload.unlink(missing_ok=True)
        return f"I couldn't register the reminder with the system scheduler.{phone}"

    if player:
        player.write_log(f"[Reminder] ✅ {date_str} {time_str} — {message[:40]}")

    friendly_time = target_dt.strftime("%B %d at %I:%M %p")
    return f"Reminder set for {friendly_time}.{phone}"


def _add_spoken_routine(message: str, target_dt: datetime) -> str:
    try:
        from core.routines import desktop
        eng = desktop.engine()
        if eng is None:
            return ""
        eng.add(name=message[:60] or "Reminder", kind="say", text=message,
                trigger={"type": "time", "rule": {"type": "once", "at_ts": target_dt.timestamp()}},
                source="reminder")
        return " It will also be said aloud at that time while the assistant is running."
    except Exception as e:
        print(f"[Reminder] spoken copy not set: {e}")
        return ""


def _mirror_to_cloud(message: str, target_dt: datetime) -> str:
    """Also schedule the reminder on the cloud body, so it reaches the phone
    over Telegram even if this computer is off at that time.

    Deterministic on purpose: which of two reminder tools a model happens to
    pick must not decide whether the phone hears about it. Returns a sentence
    to append to the result, or "" when no cloud is paired — in which case this
    costs one config read and changes nothing."""
    try:
        from memory.config_manager import get_plugin_config, get_plugin_enabled
        if not (str(get_plugin_config("cloud_agent").get("pair_code") or "").strip()
                and get_plugin_enabled("cloud_agent")):
            return ""
        agent = sys.modules.get("plugins.cloud_agent")
        if agent is None or not hasattr(agent, "mirror_reminder"):
            return ""
        ok, detail = agent.mirror_reminder(message, target_dt.strftime("%Y-%m-%d %H:%M"))
    except Exception as e:
        ok, detail = False, str(e)
    if ok:
        return " It will also be sent to the user's phone over Telegram at that time."
    return f" The phone copy could not be set ({detail}) — only this computer will remind them."


# ── Tool declaration (auto-discovered by core/action_loader.py) ──────────────
TOOL = {
    "name": "reminder",
    # Written out rather than left as the one line it used to be — "Sets a
    # timed reminder using Task Scheduler" — because a tool is chosen from its
    # description and that line says what it USES, not what it is FOR.
    # Measured, five runs of "yarın saat 3'te beni toplantı için uyar" against
    # the strong model: four of them called `system_status` and one called
    # `web_search`. Nothing was wrong with the code behind this tool; the model
    # simply could not tell that this was the tool for being reminded.
    "description": (
        "Sets a ONE-OFF reminder for a date and time: the assistant speaks it "
        "and a system notification appears. Use it whenever the user asks to "
        "be reminded, woken, alerted, warned or called at some moment — "
        "\"remind me at three\", \"in forty minutes\", \"tomorrow morning\", "
        "\"don't let me forget the meeting\", and the same in any language. "
        "Work the absolute date and time out yourself from the current date "
        "and time you were given. For something that should repeat every day "
        "or every week, use `routines` instead."),
    "brief": ("Sets a one-off reminder at a date and time, with a system "
              "notification. Repeating ones belong to `routines`."),
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "date": {
                "type": "STRING",
                "description": "Date in YYYY-MM-DD format"
            },
            "time": {
                "type": "STRING",
                "description": "Time in HH:MM format (24h)"
            },
            "message": {
                "type": "STRING",
                "description": "Reminder message text"
            }
        },
        "required": [
            "date",
            "time",
            "message"
        ]
    },
    "handler": reminder,
}
