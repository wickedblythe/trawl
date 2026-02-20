# /// script
# requires-python = ">=3.10"
# dependencies = ["rich"]
# ///
"""treader - Claude Code transcript reader.

Read and render Claude Code conversation transcripts, including team
sessions with multiple subagents.

Usage:
    treader                          # list all sessions
    treader <id>                     # view main conversation
    treader <id> --agents            # list subagents in a team session
    treader <id> --agent <agent-id>  # view a specific subagent conversation
    treader <id> --team              # interleaved team timeline
    treader <id> --raw               # dump raw JSON
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from rich.box import ASCII as ASCII_BOX, ROUNDED, SIMPLE
from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

USE_ASCII = False

def _detect_ascii() -> bool:
    """Return True if terminal likely can't handle Unicode box drawing."""
    encoding = getattr(sys.stdout, "encoding", "") or ""
    if encoding.lower().replace("-", "") not in ("utf8", "utf16", "utf32"):
        return True
    lang = os.environ.get("LANG", "") + os.environ.get("LC_ALL", "")
    if lang and "utf" not in lang.lower():
        return True
    return False

def box_style():
    """Return the box style to use for panels/tables."""
    return ASCII_BOX if USE_ASCII else ROUNDED

def table_box():
    """Return the box style to use for tables."""
    if USE_ASCII:
        return ASCII_BOX
    # Import Rich's default heavy-head table box
    from rich.box import HEAVY_HEAD
    return HEAVY_HEAD

console = Console()

# ── Paths ──────────────────────────────────────────────────────────────

def config_dir() -> Path:
    """Return the Claude config directory."""
    env = os.environ.get("CLAUDE_CONFIG_DIR")
    if env:
        return Path(env)
    return Path.home() / ".config" / "claude"


def projects_dir() -> Path:
    return config_dir() / "projects"


# ── JSONL helpers ──────────────────────────────────────────────────────

def iter_jsonl(path: Path):
    """Yield parsed records from a JSONL file, skipping bad lines."""
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def count_messages(path: Path) -> int:
    """Count user + assistant records in a JSONL file."""
    n = 0
    for rec in iter_jsonl(path):
        if rec.get("type") in ("user", "assistant"):
            n += 1
    return n


def first_user_message(path: Path) -> str:
    """Extract the first real user message text (not meta/tool_result)."""
    for rec in iter_jsonl(path):
        if rec.get("type") != "user":
            continue
        if rec.get("isMeta"):
            continue
        content = rec.get("message", {}).get("content")
        if isinstance(content, str):
            text = content.strip()
        elif isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
            text = " ".join(parts).strip()
        else:
            continue
        if not text:
            continue
        if _is_noise_user_content(text):
            continue
        # Strip XML wrapper tags for cleaner preview
        if text.startswith("<"):
            text = re.sub(r"<[^>]+>", " ", text).strip()
        # Strip ANSI escape codes and collapse to single line
        text = re.sub(r"\x1b\[[0-9;]*m", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:80]
    return ""


# ── Session discovery ──────────────────────────────────────────────────

def discover_sessions() -> list[dict]:
    """Find all sessions across all project directories."""
    pdir = projects_dir()
    if not pdir.exists():
        return []
    sessions = []
    for project in sorted(pdir.iterdir()):
        if not project.is_dir():
            continue
        for f in sorted(project.iterdir()):
            if f.suffix == ".jsonl" and f.is_file():
                sid = f.stem
                subagent_dir = project / sid / "subagents"
                has_agents = subagent_dir.is_dir() and any(subagent_dir.iterdir())
                stat = f.stat()
                sessions.append({
                    "id": sid,
                    "path": f,
                    "project": project.name,
                    "size": stat.st_size,
                    "mtime": stat.st_mtime,
                    "has_agents": has_agents,
                    "subagent_dir": subagent_dir if has_agents else None,
                })
    # Sort newest first
    sessions.sort(key=lambda s: s["mtime"], reverse=True)
    return sessions


def resolve_session(sessions: list[dict], prefix: str) -> dict | None:
    """Find a session by prefix match on ID."""
    matches = [s for s in sessions if s["id"].startswith(prefix)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        console.print(f"[yellow]Ambiguous prefix '{prefix}', matches {len(matches)} sessions:[/]")
        for m in matches[:5]:
            console.print(f"  {m['id']}")
        return None
    return None


def resolve_subagent(sessions: list[dict], prefix: str) -> tuple[dict, dict] | None:
    """Find a subagent by prefix match on agent ID across all sessions.

    Returns (session, agent) tuple or None.
    """
    matches: list[tuple[dict, dict]] = []
    for session in sessions:
        for agent in list_subagents(session):
            if agent["id"].startswith(prefix):
                matches.append((session, agent))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        console.print(f"[yellow]Ambiguous prefix '{prefix}', matches {len(matches)} subagents:[/]")
        for sess, agent in matches[:5]:
            console.print(f"  {agent['id']} in session {short_id(sess['id'])}")
        return None
    return None


def list_subagents(session: dict) -> list[dict]:
    """List subagent files for a team session."""
    if not session.get("subagent_dir"):
        return []
    agents = []
    for f in sorted(session["subagent_dir"].iterdir()):
        if f.suffix == ".jsonl":
            aid = f.stem.removeprefix("agent-")
            agents.append({"id": aid, "path": f, "size": f.stat().st_size})
    return agents


# ── Formatting helpers ─────────────────────────────────────────────────

AGENT_COLORS = [
    "bright_cyan", "bright_green", "bright_magenta", "bright_yellow",
    "bright_blue", "bright_red", "deep_sky_blue1", "spring_green1",
    "orchid1", "gold1", "turquoise2", "hot_pink",
]


def short_id(full_id: str) -> str:
    return full_id[:7]


def format_size(n: int) -> str:
    if n < 1024:
        return f"{n}B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f}K"
    return f"{n / (1024 * 1024):.1f}M"


def parse_ts(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def relative_delta(a: datetime | None, b: datetime | None) -> str:
    """Human-readable delta between two timestamps."""
    if not a or not b:
        return ""
    delta = (b - a).total_seconds()
    if delta < 0:
        return ""
    if delta < 1:
        return ""
    if delta < 60:
        return f"{delta:.0f}s"
    if delta < 3600:
        return f"{delta / 60:.0f}m"
    return f"{delta / 3600:.1f}h"


def compact_json(obj: Any, max_len: int = 120) -> str:
    """Compact JSON representation, truncated."""
    try:
        s = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        s = str(obj)
    if len(s) > max_len:
        s = s[:max_len] + "..."
    return s


def _short_model(model: str) -> str:
    """Extract a readable short model name from API model IDs."""
    if not model:
        return ""
    # claude-opus-4-6 -> opus-4-6, claude-sonnet-4-5-20250929 -> sonnet-4-5
    m = model.removeprefix("claude-")
    # Strip date suffixes like -20250929
    m = re.sub(r"-\d{8}$", "", m)
    return m


def _is_noise_user_content(text: str) -> bool:
    """Return True if user text is noise we should skip rendering."""
    if not text:
        return True
    # System reminders
    if text.startswith("<") and "system-reminder" in text[:80]:
        return True
    # Slash commands alone (/clear, /help, etc.)
    if re.match(r"^/\w+(\s+\w+)?\s*$", text):
        return True
    # XML-wrapped commands: <command-name>/clear</command-name>...
    if "<command-name>" in text[:30]:
        return True
    # Empty local-command-stdout
    if "<local-command-stdout>" in text[:30]:
        cleaned = re.sub(r"<[^>]+>", "", text).strip()
        if not cleaned:
            return True
    # local-command-caveat wrapper with no real content
    if "local-command-caveat" in text[:50] and len(text) < 500:
        cleaned = re.sub(r"<[^>]+>", "", text).strip()
        if not cleaned or cleaned.startswith("Caveat:"):
            return True
    return False


def truncate_lines(text: str, max_lines: int = 3) -> str:
    """Truncate text to max_lines, showing overflow count."""
    lines = text.split("\n")
    if len(lines) <= max_lines:
        return text
    kept = "\n".join(lines[:max_lines])
    remaining = len(lines) - max_lines
    return f"{kept}\n... ({remaining} more lines)"


# ── Message deduplication ─────────────────────────────────────────────

def _content_hash(text: str) -> str:
    """Hash message content for dedup, normalizing whitespace."""
    normalized = re.sub(r"\s+", " ", text.strip())
    return hashlib.md5(normalized.encode()).hexdigest()[:16]


def _extract_teammate_msg(text: str) -> tuple[str, str] | None:
    """Extract (sender, body) from a <teammate-message> wrapper. None if not one."""
    m = re.match(r'<teammate-message\s+teammate_id="([^"]+)"[^>]*>\s*', text)
    if not m:
        return None
    sender = m.group(1)
    body = text[m.end():]
    body = re.sub(r"</teammate-message>\s*$", "", body).strip()
    return sender, body


def _extract_send_key(block: dict) -> str | None:
    """Extract a dedup key from a SendMessage tool_use block."""
    if block.get("name") != "SendMessage":
        return None
    inp = block.get("input", {})
    content = inp.get("content", "")
    if not content:
        return None
    return _content_hash(content)


class MessageDedup:
    """Track seen inter-agent messages for dedup in merged views."""

    def __init__(self):
        self._seen: set[str] = set()

    def check_send(self, rec: dict) -> None:
        """Register SendMessage tool_use blocks from an assistant record."""
        content = rec.get("message", {}).get("content", [])
        if not isinstance(content, list):
            return
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                key = _extract_send_key(block)
                if key:
                    self._seen.add(key)

    def is_duplicate_receive(self, rec: dict) -> bool:
        """Return True if this user record is a duplicate teammate-message."""
        content = rec.get("message", {}).get("content")
        if not isinstance(content, str):
            return False
        parsed = _extract_teammate_msg(content)
        if not parsed:
            return False
        _sender, body = parsed
        key = _content_hash(body)
        if key in self._seen:
            return True
        # First receive of this content — keep it, block subsequent copies
        self._seen.add(key)
        return False


# ── Time filtering ────────────────────────────────────────────────────

GAP_THRESHOLD_SECS = 30 * 60  # 30 minutes → segment separator

def parse_time_arg(val: str) -> datetime:
    """Parse a time argument: ISO datetime, date, or relative (1h, 30m, 2d)."""
    # Relative: 1h, 30m, 2d, 1w
    m = re.match(r"^(\d+)([smhdw])$", val)
    if m:
        n, unit = int(m.group(1)), m.group(2)
        deltas = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}
        return datetime.now(timezone.utc) - timedelta(seconds=n * deltas[unit])
    # ISO datetime
    try:
        dt = datetime.fromisoformat(val)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        pass
    raise ValueError(f"Cannot parse time: {val!r} (use ISO format, date, or relative like 1h/30m/2d)")


def filter_by_time(records: list, after: datetime | None, before: datetime | None) -> list:
    """Filter (label, color, record) tuples by timestamp range."""
    if not after and not before:
        return records
    filtered = []
    for item in records:
        rec = item[2] if isinstance(item, tuple) else item
        ts = parse_ts(rec.get("timestamp"))
        if not ts:
            continue
        if after and ts < after:
            continue
        if before and ts > before:
            continue
        filtered.append(item)
    return filtered


# ── Session segmentation ──────────────────────────────────────────────

def _is_compact_marker(rec: dict) -> bool:
    """Return True if this record indicates a compaction event."""
    rtype = rec.get("type")
    if rtype == "user":
        content = rec.get("message", {}).get("content", "")
        if isinstance(content, str):
            if "<command-name>/compact</command-name>" in content:
                return True
            if "PreCompact" in content and "compaction" in content.lower():
                return True
    if rtype == "summary":
        return True
    return False


def print_segment_separator(ts: datetime | None, reason: str = ""):
    """Print a visual segment separator."""
    label = reason
    if ts:
        label = f"{reason} — {ts.strftime('%Y-%m-%d %H:%M')}" if reason else ts.strftime("%Y-%m-%d %H:%M")
    separator = f"{'─' * 20 if not USE_ASCII else '-' * 20}  {label}  {'─' * 20 if not USE_ASCII else '-' * 20}"
    console.print(Text(separator, style="bold dim"))


# ── Rendering ──────────────────────────────────────────────────────────

def render_text_block(text: str) -> Markdown:
    """Render a text block as rich Markdown."""
    return Markdown(text)


def render_tool_use(block: dict) -> Text:
    """Render a tool_use block as a compact line."""
    name = block.get("name", "?")
    inp = block.get("input", {})
    compact = compact_json(inp, max_len=100)
    t = Text()
    t.append("  [tool] ", style="yellow")
    t.append(name, style="bold yellow")
    t.append(f"({compact})", style="dim yellow")
    return t


def render_tool_result(block: dict) -> Text:
    """Render a tool_result block as truncated dim text."""
    content = block.get("content", "")
    if isinstance(content, list):
        # Sometimes content is a list of text blocks
        parts = []
        for c in content:
            if isinstance(c, dict):
                parts.append(c.get("text", str(c)))
            else:
                parts.append(str(c))
        content = "\n".join(parts)
    content = str(content)
    truncated = truncate_lines(content, max_lines=3)
    t = Text()
    t.append("  [result] ", style="dim")
    t.append(truncated, style="dim")
    return t


def _will_render(rec: dict) -> bool:
    """Return True if this record would produce visible output."""
    rtype = rec.get("type")
    if rtype in ("file-history-snapshot", "queue-operation"):
        return False
    if rtype == "progress":
        ptype = rec.get("data", {}).get("type", "")
        if ptype in ("hook_progress", "agent_progress", ""):
            return False
    if rtype == "system":
        return True
    if rtype == "user":
        if rec.get("isMeta"):
            return False
        content = rec.get("message", {}).get("content")
        if isinstance(content, str):
            return not _is_noise_user_content(content.strip())
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    return True
                if isinstance(block, str) and not _is_noise_user_content(block.strip()):
                    return True
                if isinstance(block, dict) and block.get("type") == "text":
                    if not _is_noise_user_content(block.get("text", "").strip()):
                        return True
            return False
    if rtype == "assistant":
        return True
    return False


def render_record(rec: dict, prev_ts: datetime | None, agent_label: str | None = None, agent_color: str | None = None) -> datetime | None:
    """Render a single transcript record. Returns its timestamp."""
    rtype = rec.get("type")
    ts = parse_ts(rec.get("timestamp"))
    msg = rec.get("message", {})
    content = msg.get("content")

    # Only show timestamp delta if this record will actually render something
    if _will_render(rec):
        delta_str = relative_delta(prev_ts, ts)
        if delta_str:
            time_text = Text(f"  +{delta_str}", style="dim italic")
            if agent_label:
                time_text = Text()
                time_text.append(f"  [{agent_label}] ", style=agent_color or "dim")
                time_text.append(f"+{delta_str}", style="dim italic")
            console.print(time_text)

    if rtype == "user":
        if rec.get("isMeta"):
            return ts
        if isinstance(content, str):
            text = content.strip()
            if _is_noise_user_content(text):
                return ts
            # Handle teammate messages: extract sender and show cleanly
            title = "User"
            border = "cyan"
            tm = re.match(r'<teammate-message\s+teammate_id="([^"]+)">\s*', text)
            if tm:
                sender = tm.group(1)
                text = text[tm.end():]
                text = re.sub(r"</teammate-message>\s*$", "", text).strip()
                title = f"From {sender}"
                border = "magenta"
            if agent_label:
                title = f"{agent_label} > {title}"
            console.print(Panel(
                Markdown(text) if len(text) < 5000 else Text(truncate_lines(text, 20)),
                title=title, title_align="left",
                border_style=border, width=min(console.width, 120), padding=(0, 1), box=box_style(),
            ))
        elif isinstance(content, list):
            text_parts = []
            tool_results = []
            for block in content:
                if isinstance(block, str):
                    text_parts.append(block)
                elif isinstance(block, dict):
                    bt = block.get("type")
                    if bt == "text":
                        text_parts.append(block.get("text", ""))
                    elif bt == "tool_result":
                        tool_results.append(block)
            if text_parts:
                joined = "\n".join(text_parts).strip()
                if not _is_noise_user_content(joined):
                    title = "User"
                    if agent_label:
                        title = f"{agent_label} > User"
                    console.print(Panel(
                        Markdown(joined) if len(joined) < 5000 else Text(truncate_lines(joined, 20)),
                        title=title, title_align="left",
                        border_style="cyan", width=min(console.width, 120), padding=(0, 1), box=box_style(),
                    ))
            for tr in tool_results:
                console.print(render_tool_result(tr))

    elif rtype == "assistant":
        model = msg.get("model", "")
        model_short = _short_model(model)
        if isinstance(content, list):
            text_parts = []
            tool_uses = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                bt = block.get("type")
                if bt == "text":
                    t = block.get("text", "").strip()
                    if t:
                        text_parts.append(t)
                elif bt == "tool_use":
                    tool_uses.append(block)

            if text_parts:
                joined = "\n\n".join(text_parts)
                title = f"Assistant ({model_short})" if model_short else "Assistant"
                if agent_label:
                    title = f"{agent_label} > {title}"
                console.print(Panel(
                    Markdown(joined) if len(joined) < 8000 else Text(truncate_lines(joined, 30)),
                    title=title, title_align="left",
                    border_style="green", width=min(console.width, 120), padding=(0, 1), box=box_style(),
                ))

            for tu in tool_uses:
                name = tu.get("name", "")
                inp = tu.get("input", {})
                # Highlight SendMessage calls
                if name == "SendMessage":
                    recipient = inp.get("recipient", "?")
                    msg_type = inp.get("type", "message")
                    msg_content = inp.get("content", "")
                    title = f">> {msg_type} to {recipient}"
                    if agent_label:
                        title = f"{agent_label} >> {msg_type} to {recipient}"
                    console.print(Panel(
                        Text(msg_content, style="magenta"),
                        title=title, title_align="left",
                        border_style="magenta", width=min(console.width, 120), padding=(0, 1), box=box_style(),
                    ))
                elif name == "Task":
                    task_desc = inp.get("description", "")
                    task_prompt = inp.get("prompt", "")[:80]
                    t = Text()
                    if agent_label:
                        t.append(f"  [{agent_label}] ", style=agent_color or "dim")
                    t.append("  [task] ", style="bold blue")
                    t.append(f"Task({task_desc})", style="blue")
                    if task_prompt:
                        t.append(f" - {task_prompt}", style="dim blue")
                    console.print(t)
                else:
                    t = render_tool_use(tu)
                    if agent_label:
                        prefix = Text()
                        prefix.append(f"  [{agent_label}] ", style=agent_color or "dim")
                        console.print(prefix, end="")
                    console.print(t)

    elif rtype == "system":
        subtype = rec.get("subtype", "")
        t = Text()
        if agent_label:
            t.append(f"  [{agent_label}] ", style=agent_color or "dim")
        t.append(f"  [system:{subtype}]", style="dim")
        console.print(t)

    elif rtype == "progress":
        # Only show meaningful progress (skip hooks and agent_progress noise)
        data = rec.get("data", {})
        ptype = data.get("type", "")
        if ptype not in ("hook_progress", "agent_progress", ""):
            t = Text()
            if agent_label:
                t.append(f"  [{agent_label}] ", style=agent_color or "dim")
            t.append(f"  [progress] {ptype}", style="dim")
            console.print(t)

    return ts


# ── Commands ───────────────────────────────────────────────────────────

def cmd_list(sessions: list[dict]):
    """List all sessions in a table."""
    w = min(console.width, 120)
    # Fixed cols: ID(7) + Project(10) + Date(16) + Size(6) + Msgs(4) + T(1) + padding/borders ~ 60
    msg_width = max(20, w - 62)
    table = Table(title="Claude Code Sessions", show_lines=False, padding=(0, 1), width=w, box=table_box())
    table.add_column("ID", style="bold cyan", no_wrap=True, width=7)
    table.add_column("Project", style="dim", no_wrap=True, width=10)
    table.add_column("Date", style="green", no_wrap=True, width=16)
    table.add_column("Size", justify="right", no_wrap=True, width=6)
    table.add_column("Msgs", justify="right", no_wrap=True, width=4)
    table.add_column("T", justify="center", style="bold magenta", width=1)
    table.add_column("First Message", no_wrap=True, overflow="ellipsis", width=msg_width)

    for s in sessions:
        dt = datetime.fromtimestamp(s["mtime"])
        date_str = dt.strftime("%Y-%m-%d %H:%M")
        msg_count = count_messages(s["path"])
        team_marker = "T" if s["has_agents"] else ""
        preview = first_user_message(s["path"])
        project_short = s["project"].removeprefix("-home-sisyphus-") or s["project"]

        table.add_row(
            short_id(s["id"]),
            project_short,
            date_str,
            format_size(s["size"]),
            str(msg_count),
            team_marker,
            escape(preview),
        )

    console.print(table)
    console.print(f"\n[dim]{len(sessions)} sessions found[/]")


def cmd_view(session: dict, agent_path: Path | None = None, after: datetime | None = None, before: datetime | None = None):
    """Render a conversation from a JSONL transcript."""
    path = agent_path or session["path"]
    prev_ts = None
    for rec in iter_jsonl(path):
        ts = parse_ts(rec.get("timestamp"))
        if after and ts and ts < after:
            continue
        if before and ts and ts > before:
            continue
        # Segment separator on large time gaps
        if ts and prev_ts:
            gap = (ts - prev_ts).total_seconds()
            if gap > GAP_THRESHOLD_SECS:
                print_segment_separator(ts, reason="gap")
        if _is_compact_marker(rec):
            print_segment_separator(ts, reason="compact")
        prev_ts = render_record(rec, prev_ts) or prev_ts


def cmd_agents(session: dict):
    """List subagents in a team session."""
    agents = list_subagents(session)
    if not agents:
        console.print("[yellow]No subagents found for this session.[/]")
        return

    w = min(console.width, 120)
    msg_w = max(20, w - 32)
    table = Table(title=f"Subagents for {short_id(session['id'])}", show_lines=False, padding=(0, 1), box=table_box(), width=w)
    table.add_column("Agent", style="bold cyan", no_wrap=True, width=7)
    table.add_column("Size", justify="right", no_wrap=True, width=6)
    table.add_column("Msgs", justify="right", no_wrap=True, width=4)
    table.add_column("First Message", no_wrap=True, overflow="ellipsis", width=msg_w)

    for a in agents:
        msg_count = count_messages(a["path"])
        preview = first_user_message(a["path"])
        table.add_row(
            a["id"],
            format_size(a["size"]),
            str(msg_count),
            escape(preview),
        )

    console.print(table)
    console.print(f"\n[dim]{len(agents)} subagents[/]")


def cmd_team(session: dict, after: datetime | None = None, before: datetime | None = None):
    """Render interleaved team timeline from main + all subagent transcripts."""
    # Collect all records with source labels
    all_records: list[tuple[str, str, dict]] = []  # (label, color, record)

    # Main transcript
    for rec in iter_jsonl(session["path"]):
        all_records.append(("main", "bold white", rec))

    # Subagent transcripts
    agents = list_subagents(session)
    for i, agent in enumerate(agents):
        color = AGENT_COLORS[i % len(AGENT_COLORS)]
        label = agent["id"]
        for rec in iter_jsonl(agent["path"]):
            all_records.append((label, color, rec))

    # Sort by timestamp
    all_records.sort(key=lambda item: item[2].get("timestamp", "") or "9999")

    # Apply time filter
    all_records = filter_by_time(all_records, after, before)

    console.print(Panel(
        f"Team timeline: {short_id(session['id'])} - main + {len(agents)} subagents",
        style="bold magenta", box=box_style(),
    ))

    # Legend
    legend_parts = [Text("main", style="bold white")]
    for i, agent in enumerate(agents):
        color = AGENT_COLORS[i % len(AGENT_COLORS)]
        legend_parts.append(Text(agent["id"], style=color))
    console.print("Agents: ", end="")
    for i, lp in enumerate(legend_parts):
        if i > 0:
            console.print(" | ", end="")
        console.print(lp, end="")
    console.print("\n")

    dedup = MessageDedup()
    prev_ts = None
    for label, color, rec in all_records:
        # Register sends for dedup
        if rec.get("type") == "assistant":
            dedup.check_send(rec)
        # Skip duplicate receives
        if rec.get("type") == "user" and dedup.is_duplicate_receive(rec):
            continue
        # Segment separator on large time gaps
        ts = parse_ts(rec.get("timestamp"))
        if prev_ts and ts:
            gap = (ts - prev_ts).total_seconds()
            if gap > GAP_THRESHOLD_SECS:
                print_segment_separator(ts, reason="gap")
        if _is_compact_marker(rec):
            print_segment_separator(ts, reason="compact")
        prev_ts = render_record(rec, prev_ts, agent_label=label, agent_color=color) or prev_ts


def _is_flow_record(rec: dict) -> bool:
    """Return True if this record is interesting for flow view.

    Flow view shows the orchestration story: text conversations,
    inter-agent messages, task spawns, task assignments - but skips
    tool_use/tool_result noise.
    """
    rtype = rec.get("type")

    if rtype == "user":
        if rec.get("isMeta"):
            return False
        content = rec.get("message", {}).get("content")
        if isinstance(content, str):
            text = content.strip()
            if _is_noise_user_content(text):
                return False
            # Show teammate messages and real user text
            return True
        if isinstance(content, list):
            # Skip pure tool_result messages (the noisy ones)
            has_text = False
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    continue
                if isinstance(block, str) and not _is_noise_user_content(block):
                    has_text = True
                if isinstance(block, dict) and block.get("type") == "text":
                    if not _is_noise_user_content(block.get("text", "")):
                        has_text = True
            return has_text
        return False

    if rtype == "assistant":
        content = rec.get("message", {}).get("content", [])
        if not isinstance(content, list):
            return False
        for block in content:
            if not isinstance(block, dict):
                continue
            bt = block.get("type")
            # Show text blocks (actual conversation)
            if bt == "text" and block.get("text", "").strip():
                return True
            # Show SendMessage, Task spawns (orchestration)
            if bt == "tool_use":
                name = block.get("name", "")
                if name in ("SendMessage", "Task", "TaskCreate", "TaskUpdate", "TeamCreate"):
                    return True
        return False

    return False


def render_flow_record(rec: dict, prev_ts: datetime | None, agent_label: str | None = None, agent_color: str | None = None) -> datetime | None:
    """Render a record for flow view - only orchestration-relevant content."""
    rtype = rec.get("type")
    ts = parse_ts(rec.get("timestamp"))
    msg = rec.get("message", {})
    content = msg.get("content")

    delta_str = relative_delta(prev_ts, ts)
    if delta_str:
        time_text = Text()
        if agent_label:
            time_text.append(f"  [{agent_label}] ", style=agent_color or "dim")
        time_text.append(f"+{delta_str}", style="dim italic")
        console.print(time_text)

    if rtype == "user":
        if isinstance(content, str):
            text = content.strip()
            title = "User"
            border = "cyan"
            tm = re.match(r'<teammate-message\s+teammate_id="([^"]+)">\s*', text)
            if tm:
                sender = tm.group(1)
                text = text[tm.end():]
                text = re.sub(r"</teammate-message>\s*$", "", text).strip()
                title = f"From {sender}"
                border = "magenta"
            if agent_label:
                title = f"{agent_label} > {title}"
            console.print(Panel(
                Markdown(text) if len(text) < 5000 else Text(truncate_lines(text, 20)),
                title=title, title_align="left",
                border_style=border, width=min(console.width, 120), padding=(0, 1), box=box_style(),
            ))
        elif isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, str) and not _is_noise_user_content(block):
                    text_parts.append(block)
                elif isinstance(block, dict) and block.get("type") == "text":
                    t = block.get("text", "").strip()
                    if t and not _is_noise_user_content(t):
                        text_parts.append(t)
            if text_parts:
                joined = "\n".join(text_parts).strip()
                title = "User"
                if agent_label:
                    title = f"{agent_label} > User"
                console.print(Panel(
                    Markdown(joined) if len(joined) < 5000 else Text(truncate_lines(joined, 20)),
                    title=title, title_align="left",
                    border_style="cyan", width=min(console.width, 120), padding=(0, 1), box=box_style(),
                ))

    elif rtype == "assistant":
        model = msg.get("model", "")
        model_short = _short_model(model)
        if isinstance(content, list):
            text_parts = []
            flow_tools = []
            for block in content:
                if not isinstance(block, dict):
                    continue
                bt = block.get("type")
                if bt == "text":
                    t = block.get("text", "").strip()
                    if t:
                        text_parts.append(t)
                elif bt == "tool_use":
                    name = block.get("name", "")
                    if name in ("SendMessage", "Task", "TaskCreate", "TaskUpdate", "TeamCreate"):
                        flow_tools.append(block)

            if text_parts:
                joined = "\n\n".join(text_parts)
                title = f"Assistant ({model_short})" if model_short else "Assistant"
                if agent_label:
                    title = f"{agent_label} > {title}"
                console.print(Panel(
                    Markdown(joined) if len(joined) < 8000 else Text(truncate_lines(joined, 30)),
                    title=title, title_align="left",
                    border_style="green", width=min(console.width, 120), padding=(0, 1), box=box_style(),
                ))

            for tu in flow_tools:
                name = tu.get("name", "")
                inp = tu.get("input", {})
                if name == "SendMessage":
                    recipient = inp.get("recipient", "?")
                    msg_type = inp.get("type", "message")
                    msg_content = inp.get("content", "")
                    title = f">> {msg_type} to {recipient}"
                    if agent_label:
                        title = f"{agent_label} >> {msg_type} to {recipient}"
                    console.print(Panel(
                        Text(msg_content, style="magenta"),
                        title=title, title_align="left",
                        border_style="magenta", width=min(console.width, 120), padding=(0, 1), box=box_style(),
                    ))
                elif name == "Task":
                    desc = inp.get("description", "")
                    prompt = inp.get("prompt", "")[:120]
                    t = Text()
                    if agent_label:
                        t.append(f"  [{agent_label}] ", style=agent_color or "dim")
                    t.append("  [spawn] ", style="bold blue")
                    t.append(f"Task({desc})", style="blue")
                    if prompt:
                        t.append(f" - {prompt}", style="dim blue")
                    console.print(t)
                elif name in ("TaskCreate", "TaskUpdate"):
                    subject = inp.get("subject", "")
                    status = inp.get("status", "")
                    task_id = inp.get("taskId", "")
                    t = Text()
                    if agent_label:
                        t.append(f"  [{agent_label}] ", style=agent_color or "dim")
                    t.append(f"  [{name}] ", style="bold yellow")
                    if subject:
                        t.append(subject, style="yellow")
                    if task_id:
                        t.append(f" #{task_id}", style="dim yellow")
                    if status:
                        color = "green" if status == "completed" else "yellow"
                        t.append(f" -> {status}", style=color)
                    console.print(t)
                elif name == "TeamCreate":
                    team_name = inp.get("team_name", "?")
                    t = Text()
                    if agent_label:
                        t.append(f"  [{agent_label}] ", style=agent_color or "dim")
                    t.append(f"  [team] ", style="bold blue")
                    t.append(f"TeamCreate({team_name})", style="blue")
                    console.print(t)

    return ts


def cmd_flow(session: dict, after: datetime | None = None, before: datetime | None = None):
    """Render the orchestration flow - text, messages, and task management only."""
    all_records: list[tuple[str, str, dict]] = []

    for rec in iter_jsonl(session["path"]):
        all_records.append(("main", "bold white", rec))

    agents = list_subagents(session)
    for i, agent in enumerate(agents):
        color = AGENT_COLORS[i % len(AGENT_COLORS)]
        label = agent["id"]
        for rec in iter_jsonl(agent["path"]):
            all_records.append((label, color, rec))

    all_records.sort(key=lambda item: item[2].get("timestamp", "") or "9999")

    # Apply time filter before flow filtering (cheaper to filter early)
    all_records = filter_by_time(all_records, after, before)

    # Filter to flow-relevant records
    flow_records = [(l, c, r) for l, c, r in all_records if _is_flow_record(r)]

    console.print(Panel(
        f"Flow view: {short_id(session['id'])} - {len(flow_records)} events from main + {len(agents)} agents",
        style="bold magenta", box=box_style(),
    ))

    dedup = MessageDedup()
    prev_ts = None
    for label, color, rec in flow_records:
        # Register sends for dedup
        if rec.get("type") == "assistant":
            dedup.check_send(rec)
        # Skip duplicate receives
        if rec.get("type") == "user" and dedup.is_duplicate_receive(rec):
            continue
        # Segment separator on large time gaps
        ts = parse_ts(rec.get("timestamp"))
        if prev_ts and ts:
            gap = (ts - prev_ts).total_seconds()
            if gap > GAP_THRESHOLD_SECS:
                print_segment_separator(ts, reason="gap")
        if _is_compact_marker(rec):
            print_segment_separator(ts, reason="compact")
        prev_ts = render_flow_record(rec, prev_ts, agent_label=label, agent_color=color) or prev_ts


def cmd_raw(session: dict, agent_path: Path | None = None):
    """Dump raw JSON records."""
    path = agent_path or session["path"]
    for rec in iter_jsonl(path):
        rtype = rec.get("type")
        if rtype in ("user", "assistant", "system"):
            console.print_json(json.dumps(rec))


# ── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        prog="treader",
        description="Claude Code transcript reader",
    )
    parser.add_argument("session", nargs="?", help="Session ID (prefix match)")
    parser.add_argument("--agents", action="store_true", help="List subagents in a team session")
    parser.add_argument("--agent", metavar="ID", help="View a specific subagent transcript")
    parser.add_argument("--team", action="store_true", help="Interleaved team timeline (everything)")
    parser.add_argument("--flow", action="store_true", help="Orchestration flow (text + messages + tasks only)")
    parser.add_argument("--raw", action="store_true", help="Dump raw JSON")
    parser.add_argument("--project", "-p", metavar="NAME", help="Filter by project name (substring match)")
    parser.add_argument("--after", metavar="TIME", help="Show records after TIME (ISO, date, or relative: 1h/30m/2d)")
    parser.add_argument("--before", metavar="TIME", help="Show records before TIME (ISO, date, or relative: 1h/30m/2d)")
    parser.add_argument("--ascii", action="store_true", help="Force ASCII output (no Unicode box drawing)")
    parser.add_argument("--color", action="store_true", help="Force color output (for piping to less -R)")
    args = parser.parse_args()

    global USE_ASCII, console
    if args.ascii or _detect_ascii():
        USE_ASCII = True
    if args.color:
        console = Console(force_terminal=True)

    # Parse time filters
    time_after = parse_time_arg(args.after) if args.after else None
    time_before = parse_time_arg(args.before) if args.before else None

    sessions = discover_sessions()

    if args.project:
        sessions = [s for s in sessions if args.project in s["project"]]

    if not args.session:
        cmd_list(sessions)
        return

    session = resolve_session(sessions, args.session)
    if not session:
        # Fall back to subagent ID matching
        result = resolve_subagent(sessions, args.session)
        if not result:
            console.print(f"[red]No session or subagent matching '{args.session}'[/]")
            sys.exit(1)
        parent, agent = result
        console.print(Panel(
            f"Agent {agent['id']} in session {short_id(parent['id'])}",
            style="bold cyan", box=box_style(),
        ))
        if args.raw:
            cmd_raw(parent, agent["path"])
        else:
            cmd_view(parent, agent["path"], after=time_after, before=time_before)
        return

    if args.agents:
        cmd_agents(session)
    elif args.agent:
        agents = list_subagents(session)
        matches = [a for a in agents if a["id"].startswith(args.agent)]
        if len(matches) == 1:
            if args.raw:
                cmd_raw(session, matches[0]["path"])
            else:
                console.print(Panel(
                    f"Agent {matches[0]['id']} in session {short_id(session['id'])}",
                    style="bold cyan", box=box_style(),
                ))
                cmd_view(session, matches[0]["path"], after=time_after, before=time_before)
        elif len(matches) > 1:
            console.print(f"[yellow]Ambiguous agent prefix '{args.agent}':[/]")
            for m in matches:
                console.print(f"  {m['id']}")
            sys.exit(1)
        else:
            console.print(f"[red]No agent matching '{args.agent}'[/]")
            cmd_agents(session)
            sys.exit(1)
    elif args.flow:
        cmd_flow(session, after=time_after, before=time_before)
    elif args.team:
        cmd_team(session, after=time_after, before=time_before)
    elif args.raw:
        cmd_raw(session)
    else:
        cmd_view(session, after=time_after, before=time_before)


if __name__ == "__main__":
    main()
