"""Session listing and agent listing commands."""
from __future__ import annotations

from datetime import datetime

from trawl.session import Session, count_messages, first_user_message, short_id, format_size


def cmd_find(sessions: list[Session]) -> list[dict]:
    """Return session list as canonical dicts."""
    result = []
    for s in sessions:
        dt = datetime.fromtimestamp(s.mtime)
        result.append({
            "id": s.id,
            "project": s.project,
            "date": dt.isoformat(),
            "size": s.size,
            "messages": count_messages(s.path),
            "has_agents": s.has_agents,
            "preview": first_user_message(s.path),
        })
    return result


def cmd_agents(session: Session) -> list[dict]:
    """Return subagent list as canonical dicts."""
    agents = session.agents()
    result = []
    for a in agents:
        result.append({
            "id": a.id,
            "size": a.size,
            "messages": count_messages(a.path),
            "preview": first_user_message(a.path),
        })
    return result
