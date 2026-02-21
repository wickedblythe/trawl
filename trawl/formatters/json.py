"""JSON formatter — structured output to stdout."""
from __future__ import annotations

import json
import sys
from datetime import datetime


def _default_serializer(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    return str(obj)


def format_json(data) -> None:
    """Write canonical data as JSON to stdout."""
    json.dump(data, sys.stdout, indent=2, default=_default_serializer)
    sys.stdout.write("\n")
