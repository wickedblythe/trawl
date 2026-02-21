"""Tests for trawl.commands.stats and trawl.pricing."""
from __future__ import annotations

from trawl.commands.stats import cmd_stats
from trawl.pricing import estimate_cost


class TestPricing:
    def test_exact_match(self):
        cost = estimate_cost("claude-opus-4-6", 1_000_000, 1_000_000)
        assert cost == 15.0 + 75.0  # $90 per million

    def test_fallback_match(self):
        cost = estimate_cost("claude-opus-4-99-preview", 1_000_000, 0)
        assert cost == 15.0  # matches "opus" substring

    def test_haiku_pricing(self):
        cost = estimate_cost("claude-haiku-4-5-20251001", 1_000_000, 1_000_000)
        assert cost == 0.80 + 4.0

    def test_unknown_model(self):
        assert estimate_cost("gpt-4o", 1000, 1000) == 0.0

    def test_zero_tokens(self):
        assert estimate_cost("claude-opus-4-6", 0, 0) == 0.0


class TestCmdStats:
    def test_full_stats(self, sample_session):
        data = cmd_stats(sample_session)
        assert data["session"] == "test-session-id-001"
        assert data["duration_secs"] > 0
        assert data["messages"]["user"] > 0
        assert data["messages"]["assistant"] > 0
        assert data["tokens"]["total"]["input"] > 0
        assert data["tokens"]["total"]["output"] > 0
        assert data["tools"]["total_calls"] > 0
        assert "Read" in data["tools"]["by_name"]
        assert data["cost_estimate_usd"] > 0

    def test_token_counts(self, sample_session):
        data = cmd_stats(sample_session)
        tokens = data["tokens"]["total"]
        # From sample_records: 1000+2000+500 = 3500 input
        assert tokens["input"] == 3500
        # 500+800+200 = 1500 output
        assert tokens["output"] == 1500
        # cache_read: 200+500+0 = 700
        assert tokens["cache_read"] == 700

    def test_tool_counts(self, sample_session):
        data = cmd_stats(sample_session)
        tools = data["tools"]
        assert tools["by_name"]["Read"] == 1
        assert tools["by_name"]["Edit"] == 1
        assert tools["by_name"]["Bash"] == 1
        assert tools["total_calls"] == 3

    def test_tool_errors(self, sample_session):
        data = cmd_stats(sample_session)
        assert data["tools"]["errors"] == 1
        assert data["tools"]["error_rate"] > 0

    def test_parallelism(self, sample_session):
        data = cmd_stats(sample_session)
        # 1 sidechain out of 3 assistant records
        assert data["agents"]["parallelism_ratio"] == pytest.approx(1/3, rel=0.01)

    def test_aspect_tokens(self, sample_session):
        data = cmd_stats(sample_session, aspect="tokens")
        assert "tokens" in data
        assert "tools" not in data
        assert "session" in data

    def test_aspect_tools(self, sample_session):
        data = cmd_stats(sample_session, aspect="tools")
        assert "tools" in data
        assert "tokens" not in data

    def test_aspect_cost(self, sample_session):
        data = cmd_stats(sample_session, aspect="cost")
        assert "cost_estimate_usd" in data

    def test_aspect_timing(self, sample_session):
        data = cmd_stats(sample_session, aspect="timing")
        assert "duration_secs" in data


# Need pytest imported for approx
import pytest
