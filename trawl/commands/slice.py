"""Slice command — time/index windowed extraction."""
from __future__ import annotations

from datetime import datetime

from trawl.session import Session, Record, parse_time_arg


def _parse_index_range(spec: str) -> tuple[int | None, int | None]:
    """Parse 'start:end' index range. Either side optional."""
    parts = spec.split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid index range: {spec!r} (use start:end)")
    start = int(parts[0]) if parts[0].strip() else None
    end = int(parts[1]) if parts[1].strip() else None
    return start, end


def cmd_slice(session: Session,
              after: datetime | None = None,
              before: datetime | None = None,
              index_range: str | None = None) -> dict:
    """Extract a windowed subset of records."""
    records = []
    for i, rec in enumerate(session.records()):
        ts = rec.timestamp
        if after and ts and ts < after:
            continue
        if before and ts and ts > before:
            continue
        if rec.type not in ("user", "assistant", "system"):
            continue
        records.append((i, rec))

    # Apply index range filter
    if index_range:
        start, end = _parse_index_range(index_range)
        filtered = []
        for idx, (orig_idx, rec) in enumerate(records):
            if start is not None and idx < start:
                continue
            if end is not None and idx >= end:
                break
            filtered.append((orig_idx, rec))
        records = filtered

    messages = []
    for orig_idx, rec in records:
        msg: dict = {
            "index": orig_idx,
            "role": rec.type,
            "timestamp": rec.raw.get("timestamp"),
        }
        if rec.type == "assistant":
            if rec.model:
                msg["model"] = rec.model
            text = rec.content_text
            if text:
                msg["content"] = text
            tools = [{"name": tu.get("name", "")} for tu in rec.tool_uses]
            if tools:
                msg["tools"] = tools
        else:
            msg["content"] = rec.content_text
        messages.append(msg)

    return {
        "session": session.id,
        "count": len(messages),
        "messages": messages,
    }
