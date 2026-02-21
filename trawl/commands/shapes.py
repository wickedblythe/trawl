"""Shape fingerprinting — structural variant analysis for JSONL transcripts."""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from trawl.session import Session, Record, iter_jsonl


def fingerprint(record: dict) -> str:
    """Create a structural fingerprint from a raw JSONL dict."""
    parts: list[str] = []

    # Sorted top-level keys
    parts.append(",".join(sorted(record.keys())))

    # Discriminators
    if "type" in record:
        parts.append(f"type:{record['type']}")
    if "role" in record:
        parts.append(f"role:{record['role']}")
    if "subtype" in record:
        parts.append(f"subtype:{record['subtype']}")

    # Content block types if content is a list
    content = record.get("content")
    if content is None:
        # Check nested message.content
        msg = record.get("message")
        if isinstance(msg, dict):
            content = msg.get("content")

    if isinstance(content, list):
        block_types: list[str] = []
        for block in content:
            if isinstance(block, dict):
                bt = block.get("type")
                if bt and bt not in block_types:
                    block_types.append(bt)
        if block_types:
            parts.append("blocks:" + "+".join(sorted(block_types)))

    raw = "|".join(parts)
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def deep_walk(
    obj: object,
    path: str = "$",
    depth: int = 0,
    max_depth: int = 5,
) -> list[tuple[str, str]]:
    """Recursively walk a nested structure, producing (json_path, type_description) tuples."""
    results: list[tuple[str, str]] = []

    if depth > max_depth:
        return results

    if isinstance(obj, dict):
        results.append((path, "dict"))
        for key, value in obj.items():
            child_path = f"{path}.{key}"
            results.extend(deep_walk(value, child_path, depth + 1, max_depth))
    elif isinstance(obj, list):
        results.append((path, "list"))
        if obj:
            # Collapse all items — recurse into first item only
            child_path = f"{path}[*]"
            results.extend(deep_walk(obj[0], child_path, depth + 1, max_depth))
    else:
        results.append((path, type(obj).__name__))

    return results


def cmd_shapes(
    session: Session,
    deep: bool = False,
    verify_file: str | None = None,
) -> dict:
    """Shape inventory or verification for a session."""

    # Collect fingerprint data from session records
    # Map fingerprint -> (type, keys_str, first_index, first_raw)
    fp_info: dict[str, tuple[str, str, int, dict]] = {}
    fp_counts: Counter[str] = Counter()

    for idx, rec in enumerate(session.records()):
        fp = fingerprint(rec.raw)
        fp_counts[fp] += 1
        if fp not in fp_info:
            keys_str = ",".join(sorted(rec.raw.keys()))
            fp_info[fp] = (rec.type, keys_str, idx, rec.raw)

    if verify_file is not None:
        # Load the JSON file and extract fingerprints from it
        file_path = Path(verify_file)
        file_fps: set[str] = set()

        with open(file_path) as f:
            data = json.load(f)

        # Support both a list of records and a shapes result dict
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    file_fps.add(fingerprint(item))
        elif isinstance(data, dict):
            # Assume it's a previous cmd_shapes result
            shapes_list = data.get("shapes", [])
            for shape in shapes_list:
                if isinstance(shape, dict) and "fingerprint" in shape:
                    file_fps.add(shape["fingerprint"])

        session_fps = set(fp_info.keys())
        matched = session_fps & file_fps
        missing_from_file = sorted(session_fps - file_fps)
        extra_in_file = sorted(file_fps - session_fps)
        coverage_ratio = len(matched) / len(session_fps) if session_fps else 0.0

        return {
            "session": session.id,
            "coverage": {
                "session_shapes": len(session_fps),
                "file_shapes": len(file_fps),
                "matched": len(matched),
                "missing_from_file": missing_from_file,
                "extra_in_file": extra_in_file,
                "coverage_ratio": round(coverage_ratio, 4),
            },
        }

    # Default shape inventory mode
    shapes = []
    for fp, count in fp_counts.most_common():
        rec_type, keys_str, first_idx, first_raw = fp_info[fp]
        entry: dict = {
            "fingerprint": fp,
            "type": rec_type,
            "keys": keys_str,
            "count": count,
            "example_index": first_idx,
        }
        if deep:
            paths = deep_walk(first_raw)
            entry["paths"] = [{"path": p, "type": t} for p, t in paths]
        shapes.append(entry)

    return {
        "session": session.id,
        "shapes": shapes,
    }
