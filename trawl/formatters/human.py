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
    first_user_message, short_model,
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

def _clean_project(name: str) -> str:
    """Strip common path prefixes from project directory names."""
    # Projects are encoded as -home-user-path, strip the home prefix
    import re as _re
    cleaned = _re.sub(r"^-home-[^-]+-", "", name)
    return cleaned or name


def format_find(data: list[dict]) -> None:
    w = min(console.width, 120)
    table = Table(title="Sessions", show_lines=False,
                  padding=(0, 1), width=w, box=table_box())
    table.add_column("ID", style="bold cyan", no_wrap=True, min_width=8)
    table.add_column("Project", style="dim", no_wrap=True, justify="right", max_width=12)
    table.add_column("Date", style="green", no_wrap=True, justify="right", min_width=16)
    table.add_column("Size", justify="right", no_wrap=True)
    table.add_column("Msgs", justify="right", no_wrap=True)
    table.add_column("Ag", justify="right", style="magenta")
    table.add_column("First Message", no_wrap=True, overflow="ellipsis", ratio=1)

    for s in data:
        dt = datetime.fromisoformat(s["date"])
        date_str = dt.strftime("%Y-%m-%d %H:%M")
        agent_count = s.get("agent_count", 0)
        agents_str = str(agent_count) if agent_count > 0 else ""
        table.add_row(
            short_id(s["id"]),
            _clean_project(s["project"]),
            date_str,
            format_size(s["size"]),
            str(s["messages"]),
            agents_str,
            escape(s["preview"]),
        )

    console.print(table)
    has_agents = any(s.get("agent_count", 0) > 0 for s in data)
    footer = f"\n[dim]{len(data)} sessions[/]"
    if has_agents:
        footer += "\n[dim]Tip: trawl <id> --agents to list subagents, trawl <id> --agent <aid> to read one[/]"
    console.print(footer)


# ── Agents formatter ──────────────────────────────────────────────────

def _short_agent_type(t: str) -> str:
    """Shorten subagent_type for display."""
    # "oh-my-claudecode:executor" -> "executor"
    if ":" in t:
        t = t.split(":", 1)[1]
    # "general-purpose" stays as-is
    return t


def format_agents(data: list[dict], session_id: str) -> None:
    if not data:
        console.print("[yellow]No subagents found for this session.[/]")
        return

    w = min(console.width, 120)
    table = Table(title=f"Subagents for {short_id(session_id)}",
                  show_lines=False, padding=(0, 1), box=table_box(), width=w)
    table.add_column("Agent", style="bold cyan", no_wrap=True, min_width=8)
    table.add_column("Type", style="yellow", no_wrap=True, max_width=18)
    table.add_column("Model", style="dim", no_wrap=True, max_width=12)
    table.add_column("Size", justify="right", no_wrap=True)
    table.add_column("Msgs", justify="right", no_wrap=True)
    table.add_column("First Message", no_wrap=True, overflow="ellipsis", ratio=1)

    for a in data:
        table.add_row(
            a["id"],
            _short_agent_type(a.get("type", "")),
            a.get("model", ""),
            format_size(a["size"]),
            str(a["messages"]),
            escape(a["preview"]),
        )

    console.print(table)
    console.print(f"\n[dim]{len(data)} subagents — trawl {short_id(session_id)} --agent <id> to read one[/]")


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


# ── Stats formatter ───────────────────────────────────────────────────

def format_stats(data: dict) -> None:
    w = min(console.width, 120)

    # Duration
    dur = data.get("duration_secs", 0)
    if dur >= 3600:
        dur_str = f"{dur // 3600}h {(dur % 3600) // 60}m {dur % 60}s"
    elif dur >= 60:
        dur_str = f"{dur // 60}m {dur % 60}s"
    else:
        dur_str = f"{dur}s"

    msgs = data.get("messages", {})
    tokens = data.get("tokens", {})
    total_tok = tokens.get("total", {})
    tools = data.get("tools", {})
    agents_info = data.get("agents", {})

    # Header
    lines = []
    lines.append(f"[bold]Duration:[/] {dur_str}  |  "
                 f"[bold]Messages:[/] {msgs.get('user', 0)} user / {msgs.get('assistant', 0)} assistant")

    # Tokens
    inp_k = total_tok.get("input", 0) / 1000
    out_k = total_tok.get("output", 0) / 1000
    cache_r = total_tok.get("cache_read", 0) / 1000
    cache_c = total_tok.get("cache_creation", 0) / 1000
    tok_line = f"[bold]Tokens:[/] {inp_k:.1f}K in / {out_k:.1f}K out"
    if cache_r > 0:
        tok_line += f" / {cache_r:.1f}K cache-read"
    if cache_c > 0:
        tok_line += f" / {cache_c:.1f}K cache-create"
    lines.append(tok_line)

    # Cost
    cost = data.get("cost_estimate_usd", 0)
    lines.append(f"[bold]Est. cost:[/] ${cost:.2f}")

    # Token breakdown by model
    by_model = tokens.get("by_model", {})
    if by_model:
        model_table = Table(show_header=True, box=box_style(), padding=(0, 1),
                           title="Tokens by Model", width=min(w, 80))
        model_table.add_column("Model", style="cyan")
        model_table.add_column("Input", justify="right")
        model_table.add_column("Output", justify="right")
        model_table.add_column("Cache Read", justify="right")
        model_table.add_column("Cost", justify="right", style="green")
        cost_by_model = data.get("cost_by_model", {})
        for model, mtok in by_model.items():
            model_table.add_row(
                short_model(model),
                f"{mtok.get('input', 0):,}",
                f"{mtok.get('output', 0):,}",
                f"{mtok.get('cache_read', 0):,}",
                f"${cost_by_model.get(model, 0):.2f}",
            )

    # Tools
    tool_names = tools.get("by_name", {})
    top_tools = sorted(tool_names.items(), key=lambda x: x[1], reverse=True)[:10]
    tool_str = ", ".join(f"{n}: {c}" for n, c in top_tools)
    lines.append(f"[bold]Tools:[/] {tools.get('total_calls', 0)} calls ({tool_str})")
    if tools.get("errors", 0) > 0:
        lines.append(f"  [red]Errors: {tools['errors']} ({tools.get('error_rate', 0):.1%})[/]")

    # Agents
    lines.append(f"[bold]Agents:[/] {agents_info.get('count', 0)} subagents  |  "
                 f"Parallelism: {agents_info.get('parallelism_ratio', 0):.2f}")

    console.print(Panel(
        "\n".join(lines),
        title=f"Session {short_id(data.get('session', ''))}",
        border_style="blue", box=box_style(), width=w,
    ))

    if by_model:
        console.print(model_table)


# ── Trace formatter ───────────────────────────────────────────────────

def format_trace(data: dict) -> None:
    # Chains mode
    if "chains" in data:
        _format_chains(data)
        return

    events = data.get("events", [])
    console.print(Panel(
        f"Trace: {short_id(data.get('session', ''))} - {len(events)} events",
        style="bold blue", box=box_style(),
    ))

    prev_t = None
    for ev in events:
        t = ev.get("t", "")
        ts = parse_ts(t)
        delta = relative_delta(prev_t, ts) if prev_t and ts else ""
        prev_t = ts or prev_t

        etype = ev.get("type", "")
        line = Text()

        if delta:
            line.append(f"+{delta:>4s} ", style="dim italic")
        else:
            line.append("      ", style="dim")

        if etype == "user":
            line.append("[user] ", style="bold cyan")
            line.append(ev.get("preview", ""), style="cyan")
        elif etype == "thinking":
            line.append("[think] ", style="bold dim magenta")
            line.append(ev.get("preview", ""), style="dim magenta")
        elif etype == "text":
            line.append("[text] ", style="bold green")
            line.append(ev.get("preview", ""), style="green")
        elif etype == "tool":
            line.append("[tool] ", style="bold yellow")
            line.append(ev.get("name", ""), style="yellow")
            target = ev.get("target", "")
            if target:
                line.append(f" {target}", style="dim yellow")
        elif etype == "spawn":
            line.append("[spawn] ", style="bold blue")
            line.append(ev.get("agent", ""), style="blue")
            prompt = ev.get("prompt", "")
            if prompt:
                line.append(f" - {prompt}", style="dim blue")

        console.print(line)


def _format_chains(data: dict) -> None:
    chains = data.get("chains", [])
    console.print(Panel(
        f"Spawn tree: {short_id(data.get('session', ''))} - {len(chains)} top-level spawns",
        style="bold blue", box=box_style(),
    ))

    def _print_chain(node: dict, depth: int = 0):
        indent = "  " * depth
        prefix = "+-" if depth > 0 else ""
        t = Text()
        t.append(f"{indent}{prefix}", style="dim")
        t.append(node.get("agent", "?"), style="bold blue")
        prompt = node.get("prompt", "")
        if prompt:
            t.append(f" - {prompt}", style="dim")
        console.print(t)
        for child in node.get("children", []):
            _print_chain(child, depth + 1)

    for chain in chains:
        _print_chain(chain)


# ── Shapes formatter ──────────────────────────────────────────────────

def format_shapes(data: dict) -> None:
    # Coverage mode
    if "coverage" in data:
        _format_coverage(data)
        return

    shapes = data.get("shapes", [])
    w = min(console.width, 120)

    table = Table(title=f"Shapes for {short_id(data.get('session', ''))}",
                  show_lines=False, padding=(0, 1), box=table_box(), width=w)
    table.add_column("Fingerprint", style="cyan", no_wrap=True, width=12)
    table.add_column("Type", style="bold", no_wrap=True, width=12)
    table.add_column("Count", justify="right", width=6)
    table.add_column("Keys", overflow="ellipsis")

    for shape in shapes:
        table.add_row(
            shape["fingerprint"],
            shape.get("type", ""),
            str(shape.get("count", 0)),
            shape.get("keys", ""),
        )

    console.print(table)
    console.print(f"\n[dim]{len(shapes)} unique shapes[/]")

    # Deep mode: show paths for each shape
    for shape in shapes:
        paths = shape.get("paths")
        if paths:
            console.print(f"\n[bold cyan]{shape['fingerprint']}[/] ({shape.get('type', '')}):")
            for p in paths:
                console.print(f"  {p['path']}: [dim]{p['type']}[/]")


def _format_coverage(data: dict) -> None:
    cov = data["coverage"]
    lines = [
        f"[bold]Session shapes:[/] {cov['session_shapes']}",
        f"[bold]File shapes:[/] {cov['file_shapes']}",
        f"[bold]Matched:[/] {cov['matched']}",
        f"[bold]Coverage:[/] {cov['coverage_ratio']:.1%}",
    ]
    missing = cov.get("missing_from_file", [])
    if missing:
        lines.append(f"[red]Missing from file:[/] {', '.join(missing)}")
    extra = cov.get("extra_in_file", [])
    if extra:
        lines.append(f"[yellow]Extra in file:[/] {', '.join(extra)}")
    console.print(Panel(
        "\n".join(lines),
        title=f"Coverage: {short_id(data.get('session', ''))}",
        border_style="blue", box=box_style(),
    ))
