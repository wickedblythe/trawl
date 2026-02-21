"""Toon formatter — pipes JSON through toon-cli if available."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys


def format_toon(data) -> None:
    json_str = json.dumps(data, indent=2, default=str)
    toon = shutil.which("toon-cli")
    if toon:
        proc = subprocess.run([toon], input=json_str, text=True, capture_output=True)
        sys.stdout.write(proc.stdout)
    else:
        sys.stdout.write(json_str + "\n")
