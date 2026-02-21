"""Tests for trawl.commands.shapes."""
from __future__ import annotations

import json
import pytest

from trawl.commands.shapes import fingerprint, deep_walk, cmd_shapes


class TestFingerprint:
    def test_deterministic(self):
        rec = {"type": "user", "timestamp": "2026-01-01T00:00:00Z", "message": {"content": "hi"}}
        assert fingerprint(rec) == fingerprint(rec)

    def test_different_types_differ(self):
        r1 = {"type": "user", "timestamp": "t"}
        r2 = {"type": "assistant", "timestamp": "t"}
        assert fingerprint(r1) != fingerprint(r2)

    def test_different_keys_differ(self):
        r1 = {"type": "user", "a": 1}
        r2 = {"type": "user", "b": 1}
        assert fingerprint(r1) != fingerprint(r2)

    def test_content_blocks_influence(self):
        r1 = {"type": "assistant", "message": {"content": [
            {"type": "text", "text": "hi"},
        ]}}
        r2 = {"type": "assistant", "message": {"content": [
            {"type": "text", "text": "hi"},
            {"type": "tool_use", "name": "Read"},
        ]}}
        assert fingerprint(r1) != fingerprint(r2)

    def test_length(self):
        fp = fingerprint({"type": "user"})
        assert len(fp) == 12


class TestDeepWalk:
    def test_simple_dict(self):
        paths = deep_walk({"a": 1, "b": "hello"})
        path_dict = dict(paths)
        assert path_dict["$"] == "dict"
        assert path_dict["$.a"] == "int"
        assert path_dict["$.b"] == "str"

    def test_nested_dict(self):
        paths = deep_walk({"outer": {"inner": 42}})
        path_dict = dict(paths)
        assert path_dict["$.outer"] == "dict"
        assert path_dict["$.outer.inner"] == "int"

    def test_list_collapse(self):
        paths = deep_walk({"items": [{"x": 1}, {"x": 2}, {"x": 3}]})
        path_dict = dict(paths)
        assert path_dict["$.items"] == "list"
        assert path_dict["$.items[*]"] == "dict"
        assert path_dict["$.items[*].x"] == "int"

    def test_max_depth(self):
        deep = {"a": {"b": {"c": {"d": {"e": {"f": 1}}}}}}
        paths = deep_walk(deep, max_depth=3)
        path_strs = [p for p, _ in paths]
        assert "$.a.b.c" in path_strs
        # Should stop before going deeper
        assert "$.a.b.c.d.e.f" not in path_strs

    def test_empty_list(self):
        paths = deep_walk({"items": []})
        path_dict = dict(paths)
        assert path_dict["$.items"] == "list"
        assert "$.items[*]" not in path_dict

    def test_scalar(self):
        paths = deep_walk(42)
        assert paths == [("$", "int")]


class TestCmdShapes:
    def test_inventory(self, sample_session):
        data = cmd_shapes(sample_session)
        assert data["session"] == "test-session-id-001"
        assert len(data["shapes"]) > 0
        for shape in data["shapes"]:
            assert "fingerprint" in shape
            assert "type" in shape
            assert "count" in shape
            assert "keys" in shape
            assert shape["count"] > 0

    def test_deep_mode(self, sample_session):
        data = cmd_shapes(sample_session, deep=True)
        has_paths = any("paths" in s for s in data["shapes"])
        assert has_paths
        for shape in data["shapes"]:
            if "paths" in shape:
                for p in shape["paths"]:
                    assert "path" in p
                    assert "type" in p

    def test_verify_mode(self, sample_session, tmp_path):
        # First get shapes, then verify against them
        shapes_data = cmd_shapes(sample_session)
        verify_file = tmp_path / "shapes.json"
        with open(verify_file, "w") as f:
            json.dump(shapes_data, f)

        result = cmd_shapes(sample_session, verify_file=str(verify_file))
        assert "coverage" in result
        cov = result["coverage"]
        assert cov["coverage_ratio"] == 1.0
        assert cov["matched"] == cov["session_shapes"]
        assert len(cov["missing_from_file"]) == 0
