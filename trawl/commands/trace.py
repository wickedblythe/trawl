"""Trace command — chronological event timeline extraction."""
from __future__ import annotations

from datetime import datetime

from trawl.session import Session, Record, is_noise_user_content, short_model

_TARGET_KEYS = ("file_path", "file", "command", "pattern", "query", "url", "description")


def _tool_target(inp: dict) -> str:
    for key in _TARGET_KEYS:
        val = inp.get(key)
        if val:
            return str(val)
    return ""


def _ts_iso(rec: Record) -> str:
    ts = rec.timestamp
    if ts is None:
        return rec.raw.get("timestamp", "")
    return ts.isoformat()


def cmd_trace(
    session: Session,
    thinking_only: bool = False,
    chains: bool = False,
    after: datetime | None = None,
    before: datetime | None = None,
) -> dict:
    """Extract a chronological event timeline from a session."""

    if chains:
        return _build_chains(session, after=after, before=before)

    events: list[dict] = []

    for source, rec in session.all_records(after=after, before=before):
        t = _ts_iso(rec)

        if rec.type == "user":
            if thinking_only:
                continue
            if rec.is_meta:
                continue
            text = rec.content_text
            if is_noise_user_content(text):
                continue
            ev: dict = {"t": t, "type": "user", "preview": text[:80]}
            if source != "main":
                ev["source"] = source
            events.append(ev)

        elif rec.type == "assistant":
            base: dict = {"t": t}
            if source != "main":
                base["source"] = source

            # thinking blocks
            for block in rec.thinking_blocks:
                thinking_text = block.get("thinking", "")
                events.append({**base, "type": "thinking", "preview": thinking_text[:80]})

            if thinking_only:
                continue

            # text content
            text = rec.content_text
            if text:
                events.append({**base, "type": "text", "preview": text[:80]})

            # tool uses
            for tu in rec.tool_uses:
                name = tu.get("name", "")
                inp = tu.get("input", {}) or {}

                if name == "Task":
                    description = inp.get("description", "")
                    prompt = inp.get("prompt", "")
                    events.append({
                        **base,
                        "type": "spawn",
                        "agent": description,
                        "prompt": prompt[:80],
                    })
                else:
                    target = _tool_target(inp)
                    events.append({
                        **base,
                        "type": "tool",
                        "name": name,
                        "target": target[:120],
                    })

    events.sort(key=lambda e: e["t"])

    return {
        "session": session.id,
        "events": events,
    }


def _build_chains(
    session: Session,
    after: datetime | None = None,
    before: datetime | None = None,
) -> dict:
    """Build a subagent spawn tree from Task tool_uses."""
    roots: list[dict] = []

    for _source, rec in session.all_records(after=after, before=before):
        if rec.type != "assistant":
            continue
        for tu in rec.tool_uses:
            if tu.get("name") != "Task":
                continue
            inp = tu.get("input", {}) or {}
            node: dict = {
                "agent": inp.get("description", ""),
                "prompt": (inp.get("prompt", "") or "")[:80],
                "children": [],
            }
            roots.append(node)

    return {
        "session": session.id,
        "chains": roots,
    }
