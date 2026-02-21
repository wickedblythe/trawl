"""Session statistics extraction command."""
from __future__ import annotations

from trawl.session import Session, Record
from trawl.pricing import estimate_cost


def cmd_stats(session: Session, aspect: str | None = None) -> dict:
    """Extract statistics from a session and return as a structured dict."""
    # Token aggregation
    total_tokens: dict[str, int] = {
        "input": 0,
        "output": 0,
        "cache_read": 0,
        "cache_creation": 0,
    }
    by_model: dict[str, dict[str, int]] = {}

    # Tool aggregation
    tool_by_name: dict[str, int] = {}
    total_tool_calls = 0
    tool_errors = 0

    # Timing
    first_ts = None
    last_ts = None

    # Message counts
    user_count = 0
    assistant_count = 0

    # Parallelism (sidechain)
    sidechain_count = 0

    # We need to check tool_results in user records that follow assistant records
    # Iterate all records once, collecting tool_result error counts from user records
    for rec in session.records():
        ts = rec.timestamp
        if ts is not None:
            if first_ts is None or ts < first_ts:
                first_ts = ts
            if last_ts is None or ts > last_ts:
                last_ts = ts

        if rec.type == "user":
            user_count += 1
            # Check tool_results for errors
            for tr in rec.tool_results:
                if tr.get("is_error"):
                    tool_errors += 1

        elif rec.type == "assistant":
            assistant_count += 1

            if rec.is_sidechain:
                sidechain_count += 1

            # Token usage
            usage = rec.usage
            if usage:
                inp = usage.get("input_tokens", 0) or 0
                out = usage.get("output_tokens", 0) or 0
                cr = usage.get("cache_read_input_tokens", 0) or 0
                cc = usage.get("cache_creation_input_tokens", 0) or 0

                total_tokens["input"] += inp
                total_tokens["output"] += out
                total_tokens["cache_read"] += cr
                total_tokens["cache_creation"] += cc

                model_key = rec.model_raw or "unknown"
                if model_key not in by_model:
                    by_model[model_key] = {
                        "input": 0,
                        "output": 0,
                        "cache_read": 0,
                        "cache_creation": 0,
                    }
                by_model[model_key]["input"] += inp
                by_model[model_key]["output"] += out
                by_model[model_key]["cache_read"] += cr
                by_model[model_key]["cache_creation"] += cc

            # Tool uses
            for tu in rec.tool_uses:
                name = tu.get("name", "unknown")
                tool_by_name[name] = tool_by_name.get(name, 0) + 1
                total_tool_calls += 1

    # Duration
    if first_ts is not None and last_ts is not None:
        duration_secs = int((last_ts - first_ts).total_seconds())
    else:
        duration_secs = 0

    # Cost
    cost_by_model: dict[str, float] = {}
    for model_key, mtokens in by_model.items():
        cost_by_model[model_key] = estimate_cost(
            model_key,
            mtokens["input"],
            mtokens["output"],
        )
    total_cost = sum(cost_by_model.values())

    # Agents / parallelism
    agents = session.agents()
    agent_count = len(agents)
    parallelism_ratio = (
        sidechain_count / assistant_count if assistant_count > 0 else 0.0
    )

    # Error rate
    error_rate = tool_errors / total_tool_calls if total_tool_calls > 0 else 0.0

    result: dict = {
        "session": session.id,
        "duration_secs": duration_secs,
        "messages": {"user": user_count, "assistant": assistant_count},
        "tokens": {
            "total": total_tokens,
            "by_model": by_model,
        },
        "cost_estimate_usd": total_cost,
        "cost_by_model": cost_by_model,
        "tools": {
            "total_calls": total_tool_calls,
            "by_name": tool_by_name,
            "errors": tool_errors,
            "error_rate": error_rate,
        },
        "agents": {
            "count": agent_count,
            "parallelism_ratio": parallelism_ratio,
        },
    }

    if aspect is None:
        return result

    aspect_map = {
        "tokens": {
            "session": session.id,
            "tokens": result["tokens"],
        },
        "tools": {
            "session": session.id,
            "tools": result["tools"],
        },
        "cost": {
            "session": session.id,
            "cost_estimate_usd": result["cost_estimate_usd"],
            "cost_by_model": result["cost_by_model"],
        },
        "timing": {
            "session": session.id,
            "duration_secs": result["duration_secs"],
        },
    }
    return aspect_map.get(aspect, result)
