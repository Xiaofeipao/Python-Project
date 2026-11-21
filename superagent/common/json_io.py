"""Small atomic JSON helpers and append-only event records."""

import json
import os
from datetime import datetime, timezone
from pathlib import Path


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def append_json(path, value):
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def event(output, kind, **values):
    append_json(Path(output) / "events.jsonl", {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(), "event": kind, **values,
    })
