"""Shared fixtures for trawl tests."""
from __future__ import annotations

import json
import pytest
from pathlib import Path
from datetime import datetime, timezone


@pytest.fixture
def tmp_jsonl(tmp_path):
    """Factory: write a list of dicts as a JSONL file, return path."""
    def _make(records: list[dict], name: str = "session.jsonl") -> Path:
        p = tmp_path / name
        with open(p, "w") as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
        return p
    return _make


@pytest.fixture
def sample_records():
    """Minimal transcript records for testing."""
    return [
        {
            "type": "system",
            "subtype": "init",
            "timestamp": "2026-01-01T00:00:00Z",
        },
        {
            "type": "user",
            "timestamp": "2026-01-01T00:00:01Z",
            "message": {"content": "Hello, help me with this code"},
        },
        {
            "type": "assistant",
            "timestamp": "2026-01-01T00:00:05Z",
            "isSidechain": False,
            "message": {
                "model": "claude-opus-4-6",
                "content": [
                    {"type": "thinking", "thinking": "Let me analyze the request"},
                    {"type": "text", "text": "I'll help you with that."},
                    {"type": "tool_use", "name": "Read", "input": {"file_path": "/src/main.py"}},
                ],
            },
            "usage": {
                "input_tokens": 1000,
                "output_tokens": 500,
                "cache_read_input_tokens": 200,
                "cache_creation_input_tokens": 100,
            },
        },
        {
            "type": "user",
            "timestamp": "2026-01-01T00:00:08Z",
            "message": {
                "content": [
                    {"type": "tool_result", "content": "file contents here", "is_error": False},
                ],
            },
        },
        {
            "type": "assistant",
            "timestamp": "2026-01-01T00:00:12Z",
            "isSidechain": False,
            "message": {
                "model": "claude-opus-4-6",
                "content": [
                    {"type": "text", "text": "Here's the fix."},
                    {"type": "tool_use", "name": "Edit", "input": {"file_path": "/src/main.py"}},
                ],
            },
            "usage": {
                "input_tokens": 2000,
                "output_tokens": 800,
                "cache_read_input_tokens": 500,
                "cache_creation_input_tokens": 0,
            },
        },
        {
            "type": "user",
            "timestamp": "2026-01-01T00:00:15Z",
            "message": {
                "content": [
                    {"type": "tool_result", "content": "error: file not found", "is_error": True},
                ],
            },
        },
        {
            "type": "assistant",
            "timestamp": "2026-01-01T00:00:20Z",
            "isSidechain": True,
            "message": {
                "model": "claude-haiku-4-5-20251001",
                "content": [
                    {"type": "text", "text": "Let me try another approach."},
                    {"type": "tool_use", "name": "Bash", "input": {"command": "ls /src"}},
                ],
            },
            "usage": {
                "input_tokens": 500,
                "output_tokens": 200,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
            },
        },
    ]


@pytest.fixture
def sample_session(tmp_jsonl, sample_records, tmp_path):
    """Build a Session object from sample records."""
    from trawl.session import Session
    path = tmp_jsonl(sample_records)
    return Session(
        id="test-session-id-001",
        path=path,
        project="test-project",
        size=path.stat().st_size,
        mtime=path.stat().st_mtime,
        has_agents=False,
        subagent_dir=None,
    )
