"""Session discovery, record parsing, and core data model."""
from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator


# ── Paths ─────────────────────────────────────────────────────────────

def config_dir() -> Path:
    env = os.environ.get("CLAUDE_CONFIG_DIR")
    if env:
        return Path(env)
    # Check both common locations
    xdg = Path.home() / ".config" / "claude"
    if xdg.exists():
        return xdg
    dot = Path.home() / ".claude"
    if dot.exists():
        return dot
    return xdg  # default


def projects_dir() -> Path:
    return config_dir() / "projects"


# ── JSONL helpers ─────────────────────────────────────────────────────

def iter_jsonl(path: Path) -> Iterator[dict]:
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


# ── Record ────────────────────────────────────────────────────────────

class Record:
    """Thin wrapper over a raw JSONL dict with typed property accessors."""

    __slots__ = ("raw",)

    def __init__(self, raw: dict):
        self.raw = raw

    @property
    def type(self) -> str:
        return self.raw.get("type", "")

    @property
    def timestamp(self) -> datetime | None:
        return parse_ts(self.raw.get("timestamp"))

    @property
    def message(self) -> dict:
        return self.raw.get("message", {})

    @property
    def content(self) -> Any:
        return self.message.get("content")

    @property
    def model(self) -> str:
        return short_model(self.message.get("model", ""))

    @property
    def model_raw(self) -> str:
        return self.message.get("model", "")

    @property
    def is_meta(self) -> bool:
        return bool(self.raw.get("isMeta"))

    @property
    def is_sidechain(self) -> bool:
        return bool(self.raw.get("isSidechain"))

    @property
    def uuid(self) -> str:
        return self.raw.get("uuid", "")

    @property
    def parent_uuid(self) -> str:
        return self.raw.get("parentUuid", "")

    @property
    def subtype(self) -> str:
        return self.raw.get("subtype", "")

    @property
    def usage(self) -> dict | None:
        return self.raw.get("usage") or self.message.get("usage")

    @property
    def content_text(self) -> str:
        c = self.content
        if isinstance(c, str):
            return c
        if isinstance(c, list):
            parts = []
            for block in c:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
            return "\n".join(parts)
        return ""

    @property
    def tool_uses(self) -> list[dict]:
        c = self.content
        if not isinstance(c, list):
            return []
        return [b for b in c if isinstance(b, dict) and b.get("type") == "tool_use"]

    @property
    def tool_results(self) -> list[dict]:
        c = self.content
        if not isinstance(c, list):
            return []
        return [b for b in c if isinstance(b, dict) and b.get("type") == "tool_result"]

    @property
    def thinking_blocks(self) -> list[dict]:
        c = self.content
        if not isinstance(c, list):
            return []
        return [b for b in c if isinstance(b, dict) and b.get("type") == "thinking"]

    @property
    def progress_data(self) -> dict:
        return self.raw.get("data", {})


# ── Agent ─────────────────────────────────────────────────────────────

class Agent:
    __slots__ = ("id", "path", "size")

    def __init__(self, id: str, path: Path, size: int):
        self.id = id
        self.path = path
        self.size = size

    def records(self, after: datetime | None = None, before: datetime | None = None) -> Iterator[Record]:
        for raw in iter_jsonl(self.path):
            rec = Record(raw)
            ts = rec.timestamp
            if after and ts and ts < after:
                continue
            if before and ts and ts > before:
                continue
            yield rec

    def peek_model(self) -> str:
        """Return model from first assistant record, or ''."""
        for raw in iter_jsonl(self.path):
            rec = Record(raw)
            if rec.type == "assistant" and rec.model:
                return rec.model
        return ""


# ── Session ───────────────────────────────────────────────────────────

class Session:
    __slots__ = ("id", "path", "project", "size", "mtime", "has_agents", "subagent_dir")

    def __init__(self, id: str, path: Path, project: str, size: int,
                 mtime: float, has_agents: bool, subagent_dir: Path | None):
        self.id = id
        self.path = path
        self.project = project
        self.size = size
        self.mtime = mtime
        self.has_agents = has_agents
        self.subagent_dir = subagent_dir

    def records(self, after: datetime | None = None, before: datetime | None = None) -> Iterator[Record]:
        for raw in iter_jsonl(self.path):
            rec = Record(raw)
            ts = rec.timestamp
            if after and ts and ts < after:
                continue
            if before and ts and ts > before:
                continue
            yield rec

    def agents(self) -> list[Agent]:
        if not self.subagent_dir or not self.subagent_dir.is_dir():
            return []
        agents = []
        for f in sorted(self.subagent_dir.iterdir()):
            if f.suffix == ".jsonl":
                aid = f.stem.removeprefix("agent-")
                agents.append(Agent(aid, f, f.stat().st_size))
        return agents

    def task_spawns(self) -> list[dict]:
        """Extract Task tool_use calls from this session, in order."""
        spawns = []
        for raw in iter_jsonl(self.path):
            rec = Record(raw)
            for tu in rec.tool_uses:
                if tu.get("name") == "Task":
                    inp = tu.get("input", {})
                    spawns.append({
                        "subagent_type": inp.get("subagent_type", ""),
                        "description": inp.get("description", ""),
                        "name": inp.get("name", ""),
                    })
        return spawns

    def all_records(self, after: datetime | None = None, before: datetime | None = None) -> Iterator[tuple[str, Record]]:
        """Interleaved main + agent records, sorted by timestamp."""
        all_recs: list[tuple[str, Record]] = []
        for rec in self.records(after=after, before=before):
            all_recs.append(("main", rec))
        for agent in self.agents():
            for rec in agent.records(after=after, before=before):
                all_recs.append((agent.id, rec))
        all_recs.sort(key=lambda x: x[1].raw.get("timestamp", "") or "9999")
        return iter(all_recs)


# ── Discovery ─────────────────────────────────────────────────────────

def discover_sessions() -> list[Session]:
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
                sessions.append(Session(
                    id=sid,
                    path=f,
                    project=project.name,
                    size=stat.st_size,
                    mtime=stat.st_mtime,
                    has_agents=has_agents,
                    subagent_dir=subagent_dir if has_agents else None,
                ))
    sessions.sort(key=lambda s: s.mtime, reverse=True)
    return sessions


def resolve_session(sessions: list[Session], prefix: str) -> Session | None:
    matches = [s for s in sessions if s.id.startswith(prefix)]
    if len(matches) == 1:
        return matches[0]
    return None


def resolve_session_verbose(sessions: list[Session], prefix: str, console) -> Session | None:
    """Resolve with user-facing error messages."""
    matches = [s for s in sessions if s.id.startswith(prefix)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        console.print(f"[yellow]Ambiguous prefix '{prefix}', matches {len(matches)} sessions:[/]")
        for m in matches[:5]:
            console.print(f"  {m.id}")
    return None


def resolve_subagent(sessions: list[Session], prefix: str) -> tuple[Session, Agent] | None:
    """Find a subagent by prefix match across all sessions."""
    matches: list[tuple[Session, Agent]] = []
    for session in sessions:
        for agent in session.agents():
            if agent.id.startswith(prefix):
                matches.append((session, agent))
    if len(matches) == 1:
        return matches[0]
    return None


def resolve_subagent_verbose(sessions: list[Session], prefix: str, console) -> tuple[Session, Agent] | None:
    matches: list[tuple[Session, Agent]] = []
    for session in sessions:
        for agent in session.agents():
            if agent.id.startswith(prefix):
                matches.append((session, agent))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        console.print(f"[yellow]Ambiguous prefix '{prefix}', matches {len(matches)} subagents:[/]")
        for sess, agent in matches[:5]:
            console.print(f"  {agent.id} in session {short_id(sess.id)}")
    return None


# ── Formatting helpers ────────────────────────────────────────────────

AGENT_COLORS = [
    "bright_cyan", "bright_green", "bright_magenta", "bright_yellow",
    "bright_blue", "bright_red", "deep_sky_blue1", "spring_green1",
    "orchid1", "gold1", "turquoise2", "hot_pink",
]


def short_id(full_id: str) -> str:
    return full_id[:8]


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


_B64_RE = re.compile(r'[A-Za-z0-9+/]{200,}={0,2}')


def collapse_b64(text: str) -> str:
    """Replace long base64 blobs with a size summary."""
    def _repl(m):
        raw = m.group(0)
        size = len(raw) * 3 // 4  # approximate decoded size
        return f"[base64 ~{format_size(size)}]"
    return _B64_RE.sub(_repl, text)


def compact_json(obj: Any, max_len: int = 120) -> str:
    try:
        s = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    except (TypeError, ValueError):
        s = str(obj)
    s = collapse_b64(s)
    if len(s) > max_len:
        s = s[:max_len] + "..."
    return s


def short_model(model: str) -> str:
    if not model:
        return ""
    m = model.removeprefix("claude-")
    m = re.sub(r"-\d{8}$", "", m)
    return m


def truncate_lines(text: str, max_lines: int = 3) -> str:
    lines = text.split("\n")
    if len(lines) <= max_lines:
        return text
    kept = "\n".join(lines[:max_lines])
    remaining = len(lines) - max_lines
    return f"{kept}\n... ({remaining} more lines)"


# ── Noise detection ───────────────────────────────────────────────────

def is_noise_user_content(text: str) -> bool:
    if not text:
        return True
    if text.startswith("<") and "system-reminder" in text[:80]:
        return True
    if re.match(r"^/\w+(\s+\w+)?\s*$", text):
        return True
    if "<command-name>" in text[:30]:
        return True
    if "<local-command-stdout>" in text[:30]:
        cleaned = re.sub(r"<[^>]+>", "", text).strip()
        if not cleaned:
            return True
    if "local-command-caveat" in text[:50] and len(text) < 500:
        cleaned = re.sub(r"<[^>]+>", "", text).strip()
        if not cleaned or cleaned.startswith("Caveat:"):
            return True
    return False


def first_user_message(path: Path) -> str:
    """Extract the first real user message text."""
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
        if is_noise_user_content(text):
            continue
        if text.startswith("<"):
            text = re.sub(r"<[^>]+>", " ", text).strip()
        text = re.sub(r"\x1b\[[0-9;]*m", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:80]
    return ""


def count_messages(path: Path) -> int:
    n = 0
    for rec in iter_jsonl(path):
        if rec.get("type") in ("user", "assistant"):
            n += 1
    return n


# ── Time filtering ────────────────────────────────────────────────────

def parse_time_arg(val: str) -> datetime:
    """Parse a time argument: ISO datetime, date, or relative (1h, 30m, 2d)."""
    m = re.match(r"^(\d+)([smhdw])$", val)
    if m:
        n, unit = int(m.group(1)), m.group(2)
        deltas = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}
        return datetime.now(timezone.utc) - timedelta(seconds=n * deltas[unit])
    try:
        dt = datetime.fromisoformat(val)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        pass
    raise ValueError(f"Cannot parse time: {val!r} (use ISO format, date, or relative like 1h/30m/2d)")


# ── Message deduplication ─────────────────────────────────────────────

GAP_THRESHOLD_SECS = 30 * 60


def content_hash(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text.strip())
    return hashlib.md5(normalized.encode()).hexdigest()[:16]


def extract_teammate_msg(text: str) -> tuple[str, str] | None:
    m = re.match(r'<teammate-message\s+teammate_id="([^"]+)"[^>]*>\s*', text)
    if not m:
        return None
    sender = m.group(1)
    body = text[m.end():]
    body = re.sub(r"</teammate-message>\s*$", "", body).strip()
    return sender, body


def _extract_send_key(block: dict) -> str | None:
    if block.get("name") != "SendMessage":
        return None
    inp = block.get("input", {})
    content = inp.get("content", "")
    if not content:
        return None
    return content_hash(content)


class MessageDedup:
    """Track seen inter-agent messages for dedup in merged views."""

    def __init__(self):
        self._seen: set[str] = set()

    def check_send(self, rec: Record) -> None:
        for tu in rec.tool_uses:
            key = _extract_send_key(tu)
            if key:
                self._seen.add(key)

    def is_duplicate_receive(self, rec: Record) -> bool:
        content = rec.content
        if not isinstance(content, str):
            return False
        parsed = extract_teammate_msg(content)
        if not parsed:
            return False
        _sender, body = parsed
        key = content_hash(body)
        if key in self._seen:
            return True
        self._seen.add(key)
        return False


# ── Segment detection ─────────────────────────────────────────────────

def is_compact_marker(rec: Record) -> bool:
    if rec.type == "user":
        content = rec.content
        if isinstance(content, str):
            if "<command-name>/compact</command-name>" in content:
                return True
            if "PreCompact" in content and "compaction" in content.lower():
                return True
    if rec.type == "summary":
        return True
    return False


def will_render(rec: Record) -> bool:
    """Return True if this record would produce visible output."""
    rtype = rec.type
    if rtype in ("file-history-snapshot", "queue-operation"):
        return False
    if rtype == "progress":
        ptype = rec.progress_data.get("type", "")
        if ptype in ("hook_progress", "agent_progress", ""):
            return False
    if rtype == "system":
        return True
    if rtype == "user":
        if rec.is_meta:
            return False
        content = rec.content
        if isinstance(content, str):
            return not is_noise_user_content(content.strip())
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    return True
                if isinstance(block, str) and not is_noise_user_content(block.strip()):
                    return True
                if isinstance(block, dict) and block.get("type") == "text":
                    if not is_noise_user_content(block.get("text", "").strip()):
                        return True
            return False
    if rtype == "assistant":
        return True
    return False


def is_flow_record(rec: Record) -> bool:
    """Return True if this record is interesting for flow view."""
    rtype = rec.type
    if rtype == "user":
        if rec.is_meta:
            return False
        content = rec.content
        if isinstance(content, str):
            text = content.strip()
            if is_noise_user_content(text):
                return False
            return True
        if isinstance(content, list):
            has_text = False
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    continue
                if isinstance(block, str) and not is_noise_user_content(block):
                    has_text = True
                if isinstance(block, dict) and block.get("type") == "text":
                    if not is_noise_user_content(block.get("text", "")):
                        has_text = True
            return has_text
        return False
    if rtype == "assistant":
        content = rec.content
        if not isinstance(content, list):
            return False
        for block in content:
            if not isinstance(block, dict):
                continue
            bt = block.get("type")
            if bt == "text" and block.get("text", "").strip():
                return True
            if bt == "tool_use":
                name = block.get("name", "")
                if name in ("SendMessage", "Task", "TaskCreate", "TaskUpdate", "TeamCreate"):
                    return True
        return False
    return False
