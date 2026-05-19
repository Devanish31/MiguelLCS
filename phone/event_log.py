"""Per-call JSONL event bridge from phone server to desktop GUI."""
from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path

EVENT_DIR = Path("output_logs") / "phone_events"


def event_path(call_sid: str) -> Path:
    EVENT_DIR.mkdir(parents=True, exist_ok=True)
    return EVENT_DIR / f"{call_sid}.jsonl"


def emit(call_sid: str, kind: str, text: str = "", **extra) -> None:
    payload = {
        "timestamp": datetime.now().isoformat(),
        "kind": kind,
        "text": text,
        **extra,
    }
    with event_path(call_sid).open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
