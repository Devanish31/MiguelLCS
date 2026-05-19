"""TTS selector: engine dropdown plus explicit voice dropdown."""
from __future__ import annotations
from typing import Callable
import customtkinter as ctk
from gui.themes import (
    DARK_CARD, DARK_SURFACE, TEXT_PRIMARY, TEXT_SECONDARY,
    BORDER, BTN_NEUTRAL_BG, BTN_NEUTRAL_HOVER,
    FONT_FAMILY, FONT_SIZE_BODY, FONT_SIZE_SMALL,
    FONT_SIZE_HEADING, PADDING, CORNER_RADIUS,
)
from gui.widgets.dropdown import CTkDropdown


ENGINE_INFO = {
    "edge_tts":   {"label": "Edge TTS (Free, Neural)"},
    "chatterbox": {"label": "Chatterbox (Local, Emotional)"},
    "qwen3_tts":  {"label": "Qwen3-TTS (Local, Expressive)"},
}

# Concrete voice names each engine recognises.
ENGINE_VOICES: dict[str, list[str]] = {
    "edge_tts": [
        "Jenny (English) (Female)",
        "Andrew (English) (Male)",
        "Paloma (Spanish) (Female)",
        "Alonso (Spanish) (Male)",
    ],
    "chatterbox": [
        "Warm & Empathetic",
        "Calm & Steady",
    ],
    "qwen3_tts": [
        "Ryan (Male, English)",
        "Vivian (Female, Chinese)",
    ],
}


class TTSSelector(ctk.CTkFrame):
    """Engine dropdown followed by explicit voice dropdown."""

    def __init__(
        self,
        parent,
        on_config_changed: Callable[[str, str, str], None] | None = None,
        **kwargs,
    ):
        super().__init__(parent, fg_color="transparent", **kwargs)
        self._on_config_changed = on_config_changed
        self._current_engine = "edge_tts"
        self._current_voice = ENGINE_VOICES["edge_tts"][0]

        ctk.CTkLabel(
            self,
            text="Voice agent",
            font=(FONT_FAMILY, FONT_SIZE_HEADING, "bold"),
            text_color=TEXT_PRIMARY, anchor="w",
        ).pack(fill="x", padx=PADDING, pady=(PADDING, 4))

        engine_labels = [ENGINE_INFO[k]["label"] for k in ENGINE_INFO]
        self._engine_dropdown = CTkDropdown(
            self, values=engine_labels,
            font=(FONT_FAMILY, FONT_SIZE_BODY),
            fg_color=DARK_CARD,
            button_color=BTN_NEUTRAL_BG, button_hover_color=BTN_NEUTRAL_HOVER,
            dropdown_fg_color=DARK_CARD, dropdown_hover_color=BTN_NEUTRAL_BG,
            corner_radius=CORNER_RADIUS,
            command=self._on_engine_change,
        )
        self._engine_dropdown.pack(fill="x", padx=PADDING, pady=(0, 6))
        self._engine_dropdown.set(ENGINE_INFO["edge_tts"]["label"])

        self._voice_dropdown = CTkDropdown(
            self,
            values=ENGINE_VOICES["edge_tts"],
            font=(FONT_FAMILY, FONT_SIZE_BODY),
            fg_color=DARK_CARD,
            button_color=BTN_NEUTRAL_BG,
            button_hover_color=BTN_NEUTRAL_HOVER,
            dropdown_fg_color=DARK_CARD,
            dropdown_hover_color=BTN_NEUTRAL_BG,
            corner_radius=CORNER_RADIUS,
            command=self._on_voice_change,
        )
        self._voice_dropdown.pack(fill="x", padx=PADDING, pady=(0, 6))
        self._voice_dropdown.set(self._current_voice)

    def _on_engine_change(self, label: str) -> None:
        for name, info in ENGINE_INFO.items():
            if info["label"] == label:
                self._current_engine = name
                break
        voices = ENGINE_VOICES[self._current_engine]
        self._current_voice = voices[0]
        self._voice_dropdown.configure(values=voices)
        self._voice_dropdown.set(self._current_voice)
        self._notify_change()

    def _on_voice_change(self, voice: str) -> None:
        self._current_voice = voice
        self._notify_change()

    def _notify_change(self) -> None:
        if self._on_config_changed:
            self._on_config_changed(
                self._current_engine, self.voice, self.api_key,
            )

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------
    @property
    def engine_name(self) -> str:
        return self._current_engine

    @property
    def voice(self) -> str:
        return self._current_voice

    @property
    def api_key(self) -> str:
        return ""

    def set_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        self._engine_dropdown.configure(state=state)
        self._voice_dropdown.configure(state=state)
