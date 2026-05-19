"""
Agentic Voice AI — Desktop Application
=======================================
GUI entry point. Launches the CustomTkinter desktop app.

Usage:
    python app.py
"""
from __future__ import annotations
import sys
import os
import warnings
import logging

# ── Suppress noisy warnings BEFORE any imports ──────────────────
# HuggingFace transformers warnings (forced_decoder_ids, logits processors)
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", message=".*forced_decoder_ids.*")
warnings.filterwarnings("ignore", message=".*custom logits processor.*")
warnings.filterwarnings("ignore", message=".*SuppressTokens.*")
warnings.filterwarnings("ignore", message=".*has been passed to.*")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
logging.getLogger("transformers").setLevel(logging.ERROR)

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gui.main_window import MainWindow


def main() -> None:
    app = MainWindow()
    app.mainloop()


if __name__ == "__main__":
    main()
