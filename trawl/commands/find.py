"""Session listing and agent listing commands."""
from __future__ import annotations

from datetime import datetime

from trawl.session import Session, count_messages, first_user_message, short_id, format_size


def cmd_find(sessions: list[Session], include_empty: bool = False) -> list[dict]:
    """Return session list as canonical dicts."""
    result = []
    for s in sessions:
        dt = datetime.fromtimestamp(s.mtime)
        msgs = count_messages(s.path)
        if not include_empty and msgs == 0:
            continue
        agent_count = len(s.agents()) if s.has_agents else 0
        result.append({
            "id": s.id,
            "project": s.project,
            "date": dt.isoformat(),
            "size": s.size,
            "messages": msgs,
            "has_agents": s.has_agents,
            "agent_count": agent_count,
            "preview": first_user_message(s.path),
        })
    return result


def cmd_agents(session: Session) -> list[dict]:
    """Return subagent list as canonical dicts."""
    agents = session.agents()
    spawns = session.task_spawns()
    result = []
    for i, a in enumerate(agents):
        spawn = spawns[i] if i < len(spawns) else {}
        result.append({
            "id": a.id,
            "model": a.peek_model(),
            "type": spawn.get("subagent_type", ""),
            "size": a.size,
            "messages": count_messages(a.path),
            "preview": first_user_message(a.path),
        })
    return result
