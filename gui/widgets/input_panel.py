"""Text entry panel with Send button and Record toggle."""
from __future__ import annotations
from typing import Callable
import customtkinter as ctk
from gui.themes import (
    DARK_BG_SECONDARY, DARK_CARD, ACCENT_BLUE, ACCENT_RED,
    TEXT_PRIMARY, TEXT_SECONDARY,
    FONT_FAMILY, FONT_SIZE_BODY, PADDING, CORNER_RADIUS,
)


class InputPanel(ctk.CTkFrame):
    """Bottom panel: text entry + Send + Record button."""

    def __init__(
        self,
        parent,
        on_send: Callable[[str], None],
        on_record_toggle: Callable[[bool], None] | None = None,
        **kwargs,
    ):
        super().__init__(parent, fg_color=DARK_BG_SECONDARY, height=60, **kwargs)
        self._on_send = on_send
        self._on_record_toggle = on_record_toggle
        self._recording = False

        self.pack_propagate(False)

        # Text entry
        self._entry = ctk.CTkEntry(
            self,
            placeholder_text="Type patient response...",
            font=(FONT_FAMILY, FONT_SIZE_BODY),
            fg_color=DARK_CARD,
            text_color=TEXT_PRIMARY,
            placeholder_text_color=TEXT_SECONDARY,
            corner_radius=CORNER_RADIUS,
            height=38,
        )
        self._entry.pack(side="left", fill="x", expand=True, padx=(PADDING, 6), pady=PADDING)
        self._entry.bind("<Return>", self._handle_send)

        # Send button
        self._send_btn = ctk.CTkButton(
            self,
            text="Send",
            font=(FONT_FAMILY, FONT_SIZE_BODY, "bold"),
            fg_color=ACCENT_BLUE,
            hover_color="#4090C8",
            width=70,
            height=38,
            corner_radius=CORNER_RADIUS,
            command=lambda: self._handle_send(None),
        )
        self._send_btn.pack(side="left", padx=(0, 6), pady=PADDING)

        # Record button
        self._record_btn = ctk.CTkButton(
            self,
            text="Mic",
            font=(FONT_FAMILY, FONT_SIZE_BODY),
            fg_color=DARK_CARD,
            hover_color="#404854",
            width=50,
            height=38,
            corner_radius=CORNER_RADIUS,
            command=self._toggle_record,
        )
        self._record_btn.pack(side="left", padx=(0, PADDING), pady=PADDING)

    def _handle_send(self, _event) -> None:
        text = self._entry.get().strip()
        if text:
            self._entry.delete(0, "end")
            self._on_send(text)

    def _toggle_record(self) -> None:
        self._recording = not self._recording
        if self._recording:
            self._record_btn.configure(fg_color=ACCENT_RED, text="Stop")
        else:
            self._record_btn.configure(fg_color=DARK_CARD, text="Mic")
        if self._on_record_toggle:
            self._on_record_toggle(self._recording)

    def set_recording_state(self, recording: bool) -> None:
        """Externally update the recording button state."""
        self._recording = recording
        if recording:
            self._record_btn.configure(fg_color=ACCENT_RED, text="Stop")
        else:
            self._record_btn.configure(fg_color=DARK_CARD, text="Mic")

    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable the input controls."""
        state = "normal" if enabled else "disabled"
        self._entry.configure(state=state)
        self._send_btn.configure(state=state)
        self._record_btn.configure(state=state)

    def focus_entry(self) -> None:
        """Focus the text entry field."""
        self._entry.focus_set()
