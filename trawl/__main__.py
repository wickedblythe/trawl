"""CLI entry point for trawl."""
from __future__ import annotations

import argparse
import sys

from trawl.session import (
    discover_sessions, resolve_session_verbose, resolve_subagent_verbose,
    parse_time_arg, short_id,
)


def _build_parser() -> argparse.ArgumentParser:
    # Shared global options — inherited by every subcommand
    global_opts = argparse.ArgumentParser(add_help=False)
    global_opts.add_argument("--format", "-f", choices=["human", "json", "toon"],
                             default=None, help="Output format (default: auto-detect)")
    global_opts.add_argument("--project", "-p", metavar="NAME",
                             help="Filter by project name (substring match)")
    global_opts.add_argument("--after", metavar="TIME",
                             help="Show records after TIME (ISO, date, or relative: 1h/30m/2d)")
    global_opts.add_argument("--before", metavar="TIME",
                             help="Show records before TIME (ISO, date, or relative: 1h/30m/2d)")
    global_opts.add_argument("--ascii", action="store_true",
                             help="Force ASCII output (no Unicode box drawing)")
    global_opts.add_argument("--color", action="store_true",
                             help="Force color output (for piping to less -R)")

    parser = argparse.ArgumentParser(
        prog="trawl",
        description="Claude Code transcript trawler",
        parents=[global_opts],
    )
    parser.add_argument("--version", action="version", version="trawl 0.1.0")

    sub = parser.add_subparsers(dest="command")

    # find
    sub.add_parser("find", parents=[global_opts], help="List all sessions")

    # read
    p_read = sub.add_parser("read", parents=[global_opts], help="Render a conversation")
    p_read.add_argument("session", help="Session ID (prefix match)")
    p_read.add_argument("--agents", action="store_true", help="List subagents")
    p_read.add_argument("--agent", metavar="ID", help="View a specific subagent")
    p_read.add_argument("--team", action="store_true", help="Interleaved team timeline")
    p_read.add_argument("--flow", action="store_true", help="Orchestration flow view")

    # stats (placeholder for phase 2)
    p_stats = sub.add_parser("stats", parents=[global_opts], help="Session statistics")
    p_stats.add_argument("session", help="Session ID (prefix match)")
    p_stats.add_argument("aspect", nargs="?", choices=["tokens", "tools", "cost", "timing"],
                         help="Show specific aspect")

    # trace (placeholder for phase 3)
    p_trace = sub.add_parser("trace", parents=[global_opts], help="Event timeline")
    p_trace.add_argument("session", help="Session ID (prefix match)")
    p_trace.add_argument("--thinking", action="store_true", help="Thinking blocks only")
    p_trace.add_argument("--chains", action="store_true", help="Subagent spawn tree")

    # shapes (placeholder for phase 4)
    p_shapes = sub.add_parser("shapes", parents=[global_opts], help="Shape fingerprint inventory")
    p_shapes.add_argument("session", help="Session ID (prefix match)")
    p_shapes.add_argument("--deep", action="store_true", help="Deep nested shape walk")
    p_shapes.add_argument("--verify", metavar="FILE", help="Coverage comparison file")

    # slice (placeholder for phase 5)
    p_slice = sub.add_parser("slice", parents=[global_opts], help="Extract time/index window")
    p_slice.add_argument("session", help="Session ID (prefix match)")
    p_slice.add_argument("--index", metavar="RANGE", help="Index range (e.g. 10:20)")

    return parser


def _get_format(args) -> str:
    """Determine output format from args + TTY detection."""
    if args.format:
        return args.format
    if not sys.stdout.isatty():
        return "json"
    return "human"


def main():
    parser = _build_parser()

    # Handle bare `trawl <session-id>` (no subcommand)
    # If first positional looks like a session ID (not a known subcommand), treat as read
    known_commands = {"find", "read", "stats", "trace", "shapes", "slice"}
    argv = sys.argv[1:]
    if argv and argv[0] not in known_commands and not argv[0].startswith("-"):
        argv = ["read"] + argv

    args = parser.parse_args(argv)

    # Init human formatter
    from trawl.formatters.human import init as init_human, detect_ascii
    ascii_mode = args.ascii or detect_ascii()
    init_human(ascii_mode=ascii_mode, force_color=args.color)
    from trawl.formatters.human import console

    fmt = _get_format(args)

    # Parse time filters
    time_after = parse_time_arg(args.after) if args.after else None
    time_before = parse_time_arg(args.before) if args.before else None

    sessions = discover_sessions()

    if args.project:
        sessions = [s for s in sessions if args.project in s.project]

    # No command or explicit find
    if not args.command or args.command == "find":
        from trawl.commands.find import cmd_find
        data = cmd_find(sessions)
        if fmt == "json":
            from trawl.formatters.json import format_json
            format_json(data)
        else:
            from trawl.formatters.human import format_find
            format_find(data)
        return

    # Commands that need a session
    if args.command == "read":
        session = resolve_session_verbose(sessions, args.session, console)
        if not session:
            result = resolve_subagent_verbose(sessions, args.session, console)
            if not result:
                console.print(f"[red]No session or subagent matching '{args.session}'[/]")
                sys.exit(1)
            parent, agent = result
            if fmt == "json":
                from trawl.commands.read import cmd_read
                from trawl.formatters.json import format_json
                data = cmd_read(parent, agent.path, after=time_after, before=time_before)
                format_json(data)
            else:
                from trawl.formatters.human import format_read, box_style
                console.print(Panel(
                    f"Agent {agent.id} in session {short_id(parent.id)}",
                    style="bold cyan", box=box_style(),
                ))
                format_read(parent, agent.path, after=time_after, before=time_before)
            return

        # --agents
        if args.agents:
            from trawl.commands.find import cmd_agents
            data = cmd_agents(session)
            if fmt == "json":
                from trawl.formatters.json import format_json
                format_json(data)
            else:
                from trawl.formatters.human import format_agents
                format_agents(data, session.id)
            return

        # --agent <id>
        if args.agent:
            agents = session.agents()
            matches = [a for a in agents if a.id.startswith(args.agent)]
            if len(matches) == 1:
                agent = matches[0]
                if fmt == "json":
                    from trawl.commands.read import cmd_read
                    from trawl.formatters.json import format_json
                    data = cmd_read(session, agent.path, after=time_after, before=time_before)
                    format_json(data)
                else:
                    from trawl.formatters.human import format_read, box_style
                    console.print(Panel(
                        f"Agent {agent.id} in session {short_id(session.id)}",
                        style="bold cyan", box=box_style(),
                    ))
                    format_read(session, agent.path, after=time_after, before=time_before)
            elif len(matches) > 1:
                console.print(f"[yellow]Ambiguous agent prefix '{args.agent}':[/]")
                for m in matches:
                    console.print(f"  {m.id}")
                sys.exit(1)
            else:
                console.print(f"[red]No agent matching '{args.agent}'[/]")
                from trawl.commands.find import cmd_agents
                from trawl.formatters.human import format_agents
                format_agents(cmd_agents(session), session.id)
                sys.exit(1)
            return

        # --flow
        if args.flow:
            if fmt == "json":
                from trawl.commands.read import cmd_read_team
                from trawl.formatters.json import format_json
                data = cmd_read_team(session, after=time_after, before=time_before)
                format_json(data)
            else:
                from trawl.formatters.human import format_flow
                format_flow(session, after=time_after, before=time_before)
            return

        # --team
        if args.team:
            if fmt == "json":
                from trawl.commands.read import cmd_read_team
                from trawl.formatters.json import format_json
                data = cmd_read_team(session, after=time_after, before=time_before)
                format_json(data)
            else:
                from trawl.formatters.human import format_team
                format_team(session, after=time_after, before=time_before)
            return

        # Default: read
        if fmt == "json":
            from trawl.commands.read import cmd_read
            from trawl.formatters.json import format_json
            data = cmd_read(session, after=time_after, before=time_before)
            format_json(data)
        else:
            from trawl.formatters.human import format_read
            format_read(session, after=time_after, before=time_before)
        return

    # Resolve session for remaining commands
    session = resolve_session_verbose(sessions, args.session, console)
    if not session:
        console.print(f"[red]No session matching '{args.session}'[/]")
        sys.exit(1)

    if args.command == "stats":
        from trawl.commands.stats import cmd_stats
        data = cmd_stats(session, aspect=getattr(args, "aspect", None))
        if fmt == "json":
            from trawl.formatters.json import format_json
            format_json(data)
        elif fmt == "toon":
            from trawl.formatters.toon import format_toon
            format_toon(data)
        else:
            from trawl.formatters.human import format_stats
            format_stats(data)
        return

    if args.command == "trace":
        from trawl.commands.trace import cmd_trace
        data = cmd_trace(
            session,
            thinking_only=getattr(args, "thinking", False),
            chains=getattr(args, "chains", False),
            after=time_after, before=time_before,
        )
        if fmt == "json":
            from trawl.formatters.json import format_json
            format_json(data)
        elif fmt == "toon":
            from trawl.formatters.toon import format_toon
            format_toon(data)
        else:
            from trawl.formatters.human import format_trace
            format_trace(data)
        return

    if args.command == "shapes":
        from trawl.commands.shapes import cmd_shapes
        data = cmd_shapes(
            session,
            deep=getattr(args, "deep", False),
            verify_file=getattr(args, "verify", None),
        )
        if fmt == "json":
            from trawl.formatters.json import format_json
            format_json(data)
        elif fmt == "toon":
            from trawl.formatters.toon import format_toon
            format_toon(data)
        else:
            from trawl.formatters.human import format_shapes
            format_shapes(data)
        return

    if args.command == "slice":
        from trawl.commands.slice import cmd_slice
        data = cmd_slice(
            session,
            after=time_after, before=time_before,
            index_range=getattr(args, "index", None),
        )
        if fmt == "json":
            from trawl.formatters.json import format_json
            format_json(data)
        elif fmt == "toon":
            from trawl.formatters.toon import format_toon
            format_toon(data)
        else:
            # Human: just show the count and delegate to read for rendering
            console.print(f"[dim]Slice: {data['count']} records[/]")
            from trawl.formatters.json import format_json
            format_json(data)
        return


if __name__ == "__main__":
    main()
