"""Tiny append-only debug logger for phone-call bring-up."""
from __future__ import annotations
from datetime import datetime
from pathlib import Path

LOG_PATH = Path("output_logs") / "phone_debug.log"


def log(message: str) -> None:
    LOG_PATH.parent.mkdir(exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(f"{datetime.now().isoformat()} {message}\n")
