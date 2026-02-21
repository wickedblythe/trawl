"""Read command — conversation rendering."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterator

from trawl.session import (
    Session, Record, Agent, iter_jsonl,
    is_flow_record, short_id,
)


def cmd_read(session: Session, agent_path: Path | None = None,
             after: datetime | None = None, before: datetime | None = None) -> dict:
    """Return conversation as canonical dict (for JSON output)."""
    path = agent_path or session.path
    messages = []
    for raw in iter_jsonl(path):
        rec = Record(raw)
        ts = rec.timestamp
        if after and ts and ts < after:
            continue
        if before and ts and ts > before:
            continue
        if rec.type not in ("user", "assistant", "system"):
            continue
        msg: dict = {
            "role": rec.type,
            "timestamp": rec.raw.get("timestamp"),
            "source": "main",
        }
        if rec.type == "assistant":
            if rec.model:
                msg["model"] = rec.model
            text = rec.content_text
            if text:
                msg["content"] = text
            tools = []
            for tu in rec.tool_uses:
                tool_entry: dict = {"name": tu.get("name", "")}
                inp = tu.get("input", {})
                # Extract a short summary of the tool input
                target = (inp.get("file_path") or inp.get("command")
                          or inp.get("pattern") or inp.get("description") or "")
                if target:
                    tool_entry["input_summary"] = str(target)[:120]
                tools.append(tool_entry)
            if tools:
                msg["tools"] = tools
        elif rec.type == "user":
            if rec.is_meta:
                continue
            msg["content"] = rec.content_text
        elif rec.type == "system":
            msg["content"] = rec.content_text
        messages.append(msg)
    return {
        "session": session.id,
        "messages": messages,
    }


def cmd_read_team(session: Session, after: datetime | None = None,
                  before: datetime | None = None) -> dict:
    """Return interleaved team timeline as canonical dict."""
    messages = []
    for source, rec in session.all_records(after=after, before=before):
        if rec.type not in ("user", "assistant", "system"):
            continue
        msg: dict = {
            "role": rec.type,
            "timestamp": rec.raw.get("timestamp"),
            "source": source,
        }
        if rec.type == "assistant":
            if rec.model:
                msg["model"] = rec.model
            text = rec.content_text
            if text:
                msg["content"] = text
        elif rec.type == "user":
            if rec.is_meta:
                continue
            msg["content"] = rec.content_text
        elif rec.type == "system":
            msg["content"] = rec.content_text
        messages.append(msg)
    agents = session.agents()
    return {
        "session": session.id,
        "agents": [a.id for a in agents],
        "messages": messages,
    }
