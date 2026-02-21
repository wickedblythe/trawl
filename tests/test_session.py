"""Tests for trawl.session — Record, iter_jsonl, helpers."""
from __future__ import annotations

import json
import pytest
from datetime import datetime, timezone, timedelta

from trawl.session import (
    Record, iter_jsonl, parse_ts, parse_time_arg,
    short_id, format_size, relative_delta, short_model,
    is_noise_user_content, content_hash, MessageDedup,
)


class TestRecord:
    def test_type(self):
        rec = Record({"type": "assistant"})
        assert rec.type == "assistant"

    def test_timestamp(self):
        rec = Record({"timestamp": "2026-01-01T00:00:00Z"})
        ts = rec.timestamp
        assert ts is not None
        assert ts.year == 2026

    def test_model(self):
        rec = Record({"message": {"model": "claude-opus-4-6"}})
        assert rec.model == "opus-4-6"
        assert rec.model_raw == "claude-opus-4-6"

    def test_content_text_string(self):
        rec = Record({"message": {"content": "hello world"}})
        assert rec.content_text == "hello world"

    def test_content_text_blocks(self):
        rec = Record({"message": {"content": [
            {"type": "text", "text": "first"},
            {"type": "text", "text": "second"},
        ]}})
        assert "first" in rec.content_text
        assert "second" in rec.content_text

    def test_tool_uses(self):
        rec = Record({"message": {"content": [
            {"type": "tool_use", "name": "Read", "input": {"file_path": "/a"}},
            {"type": "text", "text": "hi"},
            {"type": "tool_use", "name": "Edit", "input": {}},
        ]}})
        assert len(rec.tool_uses) == 2
        assert rec.tool_uses[0]["name"] == "Read"

    def test_tool_results(self):
        rec = Record({"message": {"content": [
            {"type": "tool_result", "content": "ok"},
        ]}})
        assert len(rec.tool_results) == 1

    def test_thinking_blocks(self):
        rec = Record({"message": {"content": [
            {"type": "thinking", "thinking": "hmm"},
            {"type": "text", "text": "ok"},
        ]}})
        assert len(rec.thinking_blocks) == 1

    def test_is_sidechain(self):
        assert Record({"isSidechain": True}).is_sidechain is True
        assert Record({"isSidechain": False}).is_sidechain is False
        assert Record({}).is_sidechain is False

    def test_usage(self):
        rec = Record({"usage": {"input_tokens": 100}})
        assert rec.usage["input_tokens"] == 100

    def test_usage_from_message(self):
        rec = Record({"message": {"usage": {"input_tokens": 200}}})
        assert rec.usage["input_tokens"] == 200


class TestIterJsonl:
    def test_basic(self, tmp_jsonl):
        path = tmp_jsonl([{"a": 1}, {"b": 2}])
        records = list(iter_jsonl(path))
        assert len(records) == 2
        assert records[0] == {"a": 1}

    def test_skips_bad_lines(self, tmp_path):
        path = tmp_path / "bad.jsonl"
        path.write_text('{"ok": true}\nnot json\n{"also": "ok"}\n')
        records = list(iter_jsonl(path))
        assert len(records) == 2

    def test_skips_blank_lines(self, tmp_path):
        path = tmp_path / "blank.jsonl"
        path.write_text('{"a": 1}\n\n\n{"b": 2}\n')
        records = list(iter_jsonl(path))
        assert len(records) == 2


class TestHelpers:
    def test_short_id(self):
        assert short_id("abc12345-full-id") == "abc1234"

    def test_format_size(self):
        assert format_size(500) == "500B"
        assert "K" in format_size(2048)
        assert "M" in format_size(2 * 1024 * 1024)

    def test_parse_ts(self):
        ts = parse_ts("2026-01-01T00:00:00Z")
        assert ts is not None
        assert ts.year == 2026
        assert parse_ts(None) is None
        assert parse_ts("not-a-date") is None

    def test_relative_delta(self):
        a = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        b = datetime(2026, 1, 1, 0, 0, 30, tzinfo=timezone.utc)
        assert relative_delta(a, b) == "30s"
        c = datetime(2026, 1, 1, 0, 5, 0, tzinfo=timezone.utc)
        assert relative_delta(a, c) == "5m"
        assert relative_delta(None, b) == ""

    def test_short_model(self):
        assert short_model("claude-opus-4-6") == "opus-4-6"
        assert short_model("claude-haiku-4-5-20251001") == "haiku-4-5"
        assert short_model("") == ""

    def test_noise_detection(self):
        assert is_noise_user_content("") is True
        assert is_noise_user_content('<system-reminder>blah</system-reminder>') is True
        assert is_noise_user_content("/clear") is True
        assert is_noise_user_content("<command-name>/help</command-name>") is True
        assert is_noise_user_content("Help me fix this bug") is False

    def test_content_hash_deterministic(self):
        h1 = content_hash("hello world")
        h2 = content_hash("hello  world")  # extra space
        assert h1 == h2  # normalizes whitespace

    def test_parse_time_arg_relative(self):
        dt = parse_time_arg("1h")
        assert dt < datetime.now(timezone.utc)
        diff = (datetime.now(timezone.utc) - dt).total_seconds()
        assert 3550 < diff < 3650  # ~1 hour

    def test_parse_time_arg_iso(self):
        dt = parse_time_arg("2026-01-15T10:00:00")
        assert dt.year == 2026
        assert dt.month == 1

    def test_parse_time_arg_invalid(self):
        with pytest.raises(ValueError):
            parse_time_arg("not-a-time")


class TestMessageDedup:
    def test_dedup_basic(self):
        dedup = MessageDedup()
        # Register a send
        send_rec = Record({"type": "assistant", "message": {"content": [
            {"type": "tool_use", "name": "SendMessage",
             "input": {"content": "hello teammate", "recipient": "worker"}},
        ]}})
        dedup.check_send(send_rec)

        # First receive — kept
        recv_rec = Record({"type": "user", "message": {
            "content": '<teammate-message teammate_id="leader">hello teammate</teammate-message>'
        }})
        assert dedup.is_duplicate_receive(recv_rec) is True  # matches the send
