# trawl

Drag a net through your Claude Code conversation logs. Pull up whatever you need.

Two consumers, same data: humans get rich terminal output, agents get structured JSON.

## Install

```bash
uv tool install .
```

Or run directly:

```bash
uv run trawl
```

## Commands

### List sessions

```bash
trawl                           # session table
trawl find -p myproject         # filter by project
```

### Read a conversation

```bash
trawl abc1234                   # rendered conversation
trawl abc1234 --flow            # orchestration only (text + messages + task spawns)
trawl abc1234 --team            # interleaved timeline (main + all subagents)
trawl abc1234 --agents          # list subagents
trawl abc1234 --agent <aid>     # view specific subagent
```

### Session stats

```bash
trawl stats abc1234             # full dashboard
trawl stats abc1234 tokens      # token breakdown by model
trawl stats abc1234 tools       # tool usage frequency
trawl stats abc1234 cost        # estimated cost
trawl stats abc1234 timing      # duration
```

### Event trace

```bash
trawl trace abc1234             # chronological timeline
trawl trace abc1234 --thinking  # thinking blocks only
trawl trace abc1234 --chains    # subagent spawn tree
```

### Shape fingerprints

```bash
trawl shapes abc1234            # structural variant inventory
trawl shapes abc1234 --deep     # nested structure walk
trawl shapes abc1234 --verify extract.json  # coverage check
```

### Slice

```bash
trawl slice abc1234 --after 1h --before 30m   # time window
trawl slice abc1234 --index 10:20             # by message index
```

## Output formats

```bash
trawl stats abc1234 --format human   # rich terminal (default)
trawl stats abc1234 --format json    # structured JSON
trawl stats abc1234 --format toon    # JSON piped through toon-cli
```

Piping auto-selects JSON:

```bash
trawl stats abc1234 | jq .cost_estimate_usd
```

## Global options

| Flag | Description |
|------|-------------|
| `--format`, `-f` | Output format: `human`, `json`, `toon` |
| `--project`, `-p` | Filter by project name (substring) |
| `--after` | Show records after TIME (ISO, relative: `1h`, `30m`, `2d`) |
| `--before` | Show records before TIME |
| `--ascii` | Force ASCII box drawing |
| `--color` | Force color (for `\| less -R`) |

## For agents

```bash
# List sessions
trawl find --format json

# Session stats
trawl stats <id> --format json

# Token usage
trawl stats <id> tokens --format json

# Conversation
trawl read <id> --format json

# Shape inventory
trawl shapes <id> --format json
```

## Requirements

- Python 3.10+
- [uv](https://docs.astral.sh/uv/)
