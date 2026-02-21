"""Human formatter — Rich terminal output."""
from __future__ import annotations

import os
import re
import sys
from datetime import datetime

from rich.box import ASCII as ASCII_BOX, ROUNDED, SIMPLE
from rich.console import Console
from rich.markdown import Markdown
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from trawl.session import (
    Session, Record, Agent, MessageDedup,
    short_id, format_size, parse_ts, relative_delta, compact_json,
    is_noise_user_content, truncate_lines, extract_teammate_msg,
    is_compact_marker, will_render, is_flow_record,
    GAP_THRESHOLD_SECS, AGENT_COLORS, iter_jsonl, count_messages,
    first_user_message,
)

# ── Module state ──────────────────────────────────────────────────────

USE_ASCII = False
console = Console()


def init(ascii_mode: bool = False, force_color: bool = False):
    global USE_ASCII, console
    USE_ASCII = ascii_mode
    if force_color:
        console = Console(force_terminal=True)
    else:
        console = Console()


def detect_ascii() -> bool:
    encoding = getattr(sys.stdout, "encoding", "") or ""
    if encoding.lower().replace("-", "") not in ("utf8", "utf16", "utf32"):
        return True
    lang = os.environ.get("LANG", "") + os.environ.get("LC_ALL", "")
    if lang and "utf" not in lang.lower():
        return True
    return False


def box_style():
    return ASCII_BOX if USE_ASCII else ROUNDED


def table_box():
    if USE_ASCII:
        return ASCII_BOX
    from rich.box import HEAVY_HEAD
    return HEAVY_HEAD


# ── Find formatter ────────────────────────────────────────────────────

def format_find(data: list[dict]) -> None:
    w = min(console.width, 120)
    msg_width = max(20, w - 62)
    table = Table(title="Claude Code Sessions", show_lines=False,
                  padding=(0, 1), width=w, box=table_box())
    table.add_column("ID", style="bold cyan", no_wrap=True, width=7)
    table.add_column("Project", style="dim", no_wrap=True, width=10)
    table.add_column("Date", style="green", no_wrap=True, width=16)
    table.add_column("Size", justify="right", no_wrap=True, width=6)
    table.add_column("Msgs", justify="right", no_wrap=True, width=4)
    table.add_column("T", justify="center", style="bold magenta", width=1)
    table.add_column("First Message", no_wrap=True, overflow="ellipsis", width=msg_width)

    for s in data:
        dt = datetime.fromisoformat(s["date"])
        date_str = dt.strftime("%Y-%m-%d %H:%M")
        team_marker = "T" if s["has_agents"] else ""
        project_short = s["project"].removeprefix("-home-sisyphus-").removeprefix("-home-mork-") or s["project"]
        table.add_row(
            short_id(s["id"]),
            project_short,
            date_str,
            format_size(s["size"]),
            str(s["messages"]),
            team_marker,
            escape(s["preview"]),
        )

    console.print(table)
    console.print(f"\n[dim]{len(data)} sessions found[/]")


# ── Agents formatter ──────────────────────────────────────────────────

def format_agents(data: list[dict], session_id: str) -> None:
    if not data:
        console.print("[yellow]No subagents found for this session.[/]")
        return

    w = min(console.width, 120)
    msg_w = max(20, w - 32)
    table = Table(title=f"Subagents for {short_id(session_id)}",
                  show_lines=False, padding=(0, 1), box=table_box(), width=w)
    table.add_column("Agent", style="bold cyan", no_wrap=True, width=7)
    table.add_column("Size", justify="right", no_wrap=True, width=6)
    table.add_column("Msgs", justify="right", no_wrap=True, width=4)
    table.add_column("First Message", no_wrap=True, overflow="ellipsis", width=msg_w)

    for a in data:
        table.add_row(
            a["id"],
            format_size(a["size"]),
            str(a["messages"]),
            escape(a["preview"]),
        )

    console.print(table)
    console.print(f"\n[dim]{len(data)} subagents[/]")


# ── Record rendering ──────────────────────────────────────────────────

def _render_text_block(text: str) -> Markdown:
    return Markdown(text)


def _render_tool_use(block: dict) -> Text:
    name = block.get("name", "?")
    inp = block.get("input", {})
    compact = compact_json(inp, max_len=100)
    t = Text()
    t.append("  [tool] ", style="yellow")
    t.append(name, style="bold yellow")
    t.append(f"({compact})", style="dim yellow")
    return t


def _render_tool_result(block: dict) -> Text:
    content = block.get("content", "")
    if isinstance(content, list):
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


def _print_segment_separator(ts: datetime | None, reason: str = ""):
    label = reason
    if ts:
        label = f"{reason} — {ts.strftime('%Y-%m-%d %H:%M')}" if reason else ts.strftime("%Y-%m-%d %H:%M")
    sep_char = "-" if USE_ASCII else "\u2500"
    separator = f"{sep_char * 20}  {label}  {sep_char * 20}"
    console.print(Text(separator, style="bold dim"))


def render_record(rec: Record, prev_ts: datetime | None,
                  agent_label: str | None = None,
                  agent_color: str | None = None) -> datetime | None:
    """Render a single transcript record. Returns its timestamp."""
    ts = rec.timestamp

    if will_render(rec):
        delta_str = relative_delta(prev_ts, ts)
        if delta_str:
            time_text = Text(f"  +{delta_str}", style="dim italic")
            if agent_label:
                time_text = Text()
                time_text.append(f"  [{agent_label}] ", style=agent_color or "dim")
                time_text.append(f"+{delta_str}", style="dim italic")
            console.print(time_text)

    rtype = rec.type
    content = rec.content

    if rtype == "user":
        if rec.is_meta:
            return ts
        if isinstance(content, str):
            text = content.strip()
            if is_noise_user_content(text):
                return ts
            title = "User"
            border = "cyan"
            tm = extract_teammate_msg(text)
            if tm:
                sender, text = tm
                title = f"From {sender}"
                border = "magenta"
            if agent_label:
                title = f"{agent_label} > {title}"
            console.print(Panel(
                Markdown(text) if len(text) < 5000 else Text(truncate_lines(text, 20)),
                title=title, title_align="left",
                border_style=border, width=min(console.width, 120),
                padding=(0, 1), box=box_style(),
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
                if not is_noise_user_content(joined):
                    title = "User"
                    if agent_label:
                        title = f"{agent_label} > User"
                    console.print(Panel(
                        Markdown(joined) if len(joined) < 5000 else Text(truncate_lines(joined, 20)),
                        title=title, title_align="left",
                        border_style="cyan", width=min(console.width, 120),
                        padding=(0, 1), box=box_style(),
                    ))
            for tr in tool_results:
                console.print(_render_tool_result(tr))

    elif rtype == "assistant":
        model_short = rec.model
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
                    border_style="green", width=min(console.width, 120),
                    padding=(0, 1), box=box_style(),
                ))

            for tu in tool_uses:
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
                        border_style="magenta", width=min(console.width, 120),
                        padding=(0, 1), box=box_style(),
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
                    t = _render_tool_use(tu)
                    if agent_label:
                        prefix = Text()
                        prefix.append(f"  [{agent_label}] ", style=agent_color or "dim")
                        console.print(prefix, end="")
                    console.print(t)

    elif rtype == "system":
        t = Text()
        if agent_label:
            t.append(f"  [{agent_label}] ", style=agent_color or "dim")
        t.append(f"  [system:{rec.subtype}]", style="dim")
        console.print(t)

    elif rtype == "progress":
        ptype = rec.progress_data.get("type", "")
        if ptype not in ("hook_progress", "agent_progress", ""):
            t = Text()
            if agent_label:
                t.append(f"  [{agent_label}] ", style=agent_color or "dim")
            t.append(f"  [progress] {ptype}", style="dim")
            console.print(t)

    return ts


# ── Read formatter (direct rendering path) ────────────────────────────

def format_read(session: Session, agent_path=None,
                after=None, before=None) -> None:
    """Render conversation directly — bypasses canonical IR."""
    path = agent_path or session.path
    prev_ts = None
    for raw in iter_jsonl(path):
        rec = Record(raw)
        ts = rec.timestamp
        if after and ts and ts < after:
            continue
        if before and ts and ts > before:
            continue
        if ts and prev_ts:
            gap = (ts - prev_ts).total_seconds()
            if gap > GAP_THRESHOLD_SECS:
                _print_segment_separator(ts, reason="gap")
        if is_compact_marker(rec):
            _print_segment_separator(ts, reason="compact")
        prev_ts = render_record(rec, prev_ts) or prev_ts


# ── Team formatter (direct rendering path) ────────────────────────────

def format_team(session: Session, after=None, before=None) -> None:
    agents = session.agents()

    console.print(Panel(
        f"Team timeline: {short_id(session.id)} - main + {len(agents)} subagents",
        style="bold magenta", box=box_style(),
    ))

    legend_parts = [("main", "bold white")]
    for i, agent in enumerate(agents):
        color = AGENT_COLORS[i % len(AGENT_COLORS)]
        legend_parts.append((agent.id, color))
    console.print("Agents: ", end="")
    for i, (label, color) in enumerate(legend_parts):
        if i > 0:
            console.print(" | ", end="")
        console.print(Text(label, style=color), end="")
    console.print("\n")

    # Build color map
    color_map: dict[str, str] = {"main": "bold white"}
    for i, agent in enumerate(agents):
        color_map[agent.id] = AGENT_COLORS[i % len(AGENT_COLORS)]

    dedup = MessageDedup()
    prev_ts = None
    for source, rec in session.all_records(after=after, before=before):
        if rec.type == "assistant":
            dedup.check_send(rec)
        if rec.type == "user" and dedup.is_duplicate_receive(rec):
            continue
        ts = rec.timestamp
        if prev_ts and ts:
            gap = (ts - prev_ts).total_seconds()
            if gap > GAP_THRESHOLD_SECS:
                _print_segment_separator(ts, reason="gap")
        if is_compact_marker(rec):
            _print_segment_separator(ts, reason="compact")
        color = color_map.get(source, "dim")
        prev_ts = render_record(rec, prev_ts, agent_label=source, agent_color=color) or prev_ts


# ── Flow formatter (direct rendering path) ────────────────────────────

def _render_flow_record(rec: Record, prev_ts: datetime | None,
                        agent_label: str | None = None,
                        agent_color: str | None = None) -> datetime | None:
    ts = rec.timestamp
    content = rec.content

    delta_str = relative_delta(prev_ts, ts)
    if delta_str:
        time_text = Text()
        if agent_label:
            time_text.append(f"  [{agent_label}] ", style=agent_color or "dim")
        time_text.append(f"+{delta_str}", style="dim italic")
        console.print(time_text)

    if rec.type == "user":
        if isinstance(content, str):
            text = content.strip()
            title = "User"
            border = "cyan"
            tm = extract_teammate_msg(text)
            if tm:
                sender, text = tm
                title = f"From {sender}"
                border = "magenta"
            if agent_label:
                title = f"{agent_label} > {title}"
            console.print(Panel(
                Markdown(text) if len(text) < 5000 else Text(truncate_lines(text, 20)),
                title=title, title_align="left",
                border_style=border, width=min(console.width, 120),
                padding=(0, 1), box=box_style(),
            ))
        elif isinstance(content, list):
            text_parts = []
            for block in content:
                if isinstance(block, str) and not is_noise_user_content(block):
                    text_parts.append(block)
                elif isinstance(block, dict) and block.get("type") == "text":
                    t = block.get("text", "").strip()
                    if t and not is_noise_user_content(t):
                        text_parts.append(t)
            if text_parts:
                joined = "\n".join(text_parts).strip()
                title = "User"
                if agent_label:
                    title = f"{agent_label} > User"
                console.print(Panel(
                    Markdown(joined) if len(joined) < 5000 else Text(truncate_lines(joined, 20)),
                    title=title, title_align="left",
                    border_style="cyan", width=min(console.width, 120),
                    padding=(0, 1), box=box_style(),
                ))

    elif rec.type == "assistant":
        model_short = rec.model
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
                    border_style="green", width=min(console.width, 120),
                    padding=(0, 1), box=box_style(),
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
                        border_style="magenta", width=min(console.width, 120),
                        padding=(0, 1), box=box_style(),
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
                        scolor = "green" if status == "completed" else "yellow"
                        t.append(f" -> {status}", style=scolor)
                    console.print(t)
                elif name == "TeamCreate":
                    team_name = inp.get("team_name", "?")
                    t = Text()
                    if agent_label:
                        t.append(f"  [{agent_label}] ", style=agent_color or "dim")
                    t.append("  [team] ", style="bold blue")
                    t.append(f"TeamCreate({team_name})", style="blue")
                    console.print(t)

    return ts


def format_flow(session: Session, after=None, before=None) -> None:
    agents = session.agents()

    # Build all records with source labels
    all_recs: list[tuple[str, Record]] = []
    for rec in session.records():
        all_recs.append(("main", rec))
    for agent in agents:
        for rec in agent.records():
            all_recs.append((agent.id, rec))
    all_recs.sort(key=lambda x: x[1].raw.get("timestamp", "") or "9999")

    # Time filter
    if after or before:
        filtered = []
        for source, rec in all_recs:
            ts = rec.timestamp
            if after and ts and ts < after:
                continue
            if before and ts and ts > before:
                continue
            filtered.append((source, rec))
        all_recs = filtered

    # Flow filter
    flow_recs = [(s, r) for s, r in all_recs if is_flow_record(r)]

    color_map: dict[str, str] = {"main": "bold white"}
    for i, agent in enumerate(agents):
        color_map[agent.id] = AGENT_COLORS[i % len(AGENT_COLORS)]

    console.print(Panel(
        f"Flow view: {short_id(session.id)} - {len(flow_recs)} events from main + {len(agents)} agents",
        style="bold magenta", box=box_style(),
    ))

    dedup = MessageDedup()
    prev_ts = None
    for source, rec in flow_recs:
        if rec.type == "assistant":
            dedup.check_send(rec)
        if rec.type == "user" and dedup.is_duplicate_receive(rec):
            continue
        ts = rec.timestamp
        if prev_ts and ts:
            gap = (ts - prev_ts).total_seconds()
            if gap > GAP_THRESHOLD_SECS:
                _print_segment_separator(ts, reason="gap")
        if is_compact_marker(rec):
            _print_segment_separator(ts, reason="compact")
        color = color_map.get(source, "dim")
        prev_ts = _render_flow_record(rec, prev_ts, agent_label=source, agent_color=color) or prev_ts
