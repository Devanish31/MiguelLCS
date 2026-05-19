"""Status bar showing phase, gaps resolved, and confidence."""
from __future__ import annotations
import customtkinter as ctk
from gui.themes import (
    DARK_BG_SECONDARY, ACCENT_GREEN, ACCENT_AMBER, ACCENT_BLUE,
    TEXT_PRIMARY, TEXT_SECONDARY,
    FONT_FAMILY, FONT_SIZE_SMALL, FONT_SIZE_TINY, PADDING,
)


class StatusBar(ctk.CTkFrame):
    """Bottom status bar: Phase | Gaps | Confidence | Speaking indicator."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, fg_color=DARK_BG_SECONDARY, height=36, **kwargs)
        self.pack_propagate(False)

        # Phase label
        self._phase_label = ctk.CTkLabel(
            self,
            text="Phase: IDLE",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            text_color=TEXT_SECONDARY,
        )
        self._phase_label.pack(side="left", padx=(PADDING, 20))

        # Gaps resolved
        self._gaps_label = ctk.CTkLabel(
            self,
            text="Gaps: 0/0",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            text_color=TEXT_SECONDARY,
        )
        self._gaps_label.pack(side="left", padx=(0, 20))

        # Confidence bar
        conf_frame = ctk.CTkFrame(self, fg_color="transparent")
        conf_frame.pack(side="left", padx=(0, 20))
        ctk.CTkLabel(
            conf_frame,
            text="Confidence:",
            font=(FONT_FAMILY, FONT_SIZE_TINY),
            text_color=TEXT_SECONDARY,
        ).pack(side="left", padx=(0, 4))
        self._conf_bar = ctk.CTkProgressBar(
            conf_frame, width=100, height=12,
            progress_color=ACCENT_GREEN,
        )
        self._conf_bar.pack(side="left")
        self._conf_bar.set(0)

        # Speaking indicator
        self._speaking_label = ctk.CTkLabel(
            self,
            text="",
            font=(FONT_FAMILY, FONT_SIZE_TINY),
            text_color=ACCENT_AMBER,
        )
        self._speaking_label.pack(side="right", padx=PADDING)

        # Turn counter
        self._turn_label = ctk.CTkLabel(
            self,
            text="Turn: 0",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            text_color=TEXT_SECONDARY,
        )
        self._turn_label.pack(side="right", padx=(0, 20))

    def update_phase(self, phase: str) -> None:
        self._phase_label.configure(text=f"Phase: {phase}")

    def update_gaps(self, resolved: int, total: int) -> None:
        self._gaps_label.configure(text=f"Gaps: {resolved}/{total}")

    def update_confidence(self, value: float) -> None:
        self._conf_bar.set(min(1.0, max(0.0, value)))
        color = ACCENT_GREEN if value >= 0.7 else ACCENT_AMBER if value >= 0.4 else "#EF5350"
        self._conf_bar.configure(progress_color=color)

    def update_turn(self, turn: int) -> None:
        self._turn_label.configure(text=f"Turn: {turn}")

    def set_speaking(self, speaking: bool) -> None:
        self._speaking_label.configure(text="Speaking..." if speaking else "")

    def set_listening(self, listening: bool) -> None:
        self._speaking_label.configure(text="Listening..." if listening else "")
