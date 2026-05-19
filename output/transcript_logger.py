"""Logs the conversation with timestamps."""
from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path


class TranscriptLogger:
    """Records and saves the full conversation transcript."""

    def __init__(self, patient_id: str, output_dir: str = "output_logs"):
        self.patient_id = patient_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.start_time = datetime.now()
        self.entries: list[dict] = []

    def log(self, role: str, text: str, metadata: dict | None = None) -> None:
        """Log a conversation entry."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": round(
                (datetime.now() - self.start_time).total_seconds(), 2
            ),
            "role": role,
            "text": text,
        }
        if metadata:
            entry["metadata"] = metadata
        self.entries.append(entry)

    def save(self, filename: str | None = None) -> str:
        """Save transcript to JSON file. Returns filepath."""
        if filename is None:
            ts = self.start_time.strftime("%Y%m%d_%H%M%S")
            filename = f"transcript_{self.patient_id}_{ts}.json"
        filepath = self.output_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "patient_id": self.patient_id,
                    "start_time": self.start_time.isoformat(),
                    "duration_seconds": round(
                        (datetime.now() - self.start_time).total_seconds(), 2
                    ),
                    "entries": self.entries,
                },
                f,
                indent=2,
                default=str,
            )
        return str(filepath)
