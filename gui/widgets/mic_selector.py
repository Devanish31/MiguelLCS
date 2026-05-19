"""Microphone input device selector — sidebar widget."""
from __future__ import annotations
from typing import Callable
import customtkinter as ctk
from gui.themes import (
    DARK_CARD, DARK_SURFACE, TEXT_PRIMARY, TEXT_SECONDARY,
    BTN_NEUTRAL_BG, BTN_NEUTRAL_HOVER,
    FONT_FAMILY, FONT_SIZE_BODY, FONT_SIZE_SMALL,
    FONT_SIZE_HEADING, PADDING, CORNER_RADIUS,
)
from gui.widgets.dropdown import CTkDropdown
from voice.audio_recorder import list_input_devices


_AUTO_LABEL = "Auto-detect"
_REFRESH_LABEL = "↻  Refresh devices"
_NAME_MAX = 28  # truncate long device names so the dropdown stays narrow


def _truncate(text: str, n: int = _NAME_MAX) -> str:
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def _device_label(d: dict) -> str:
    """Compact, fixed-width-friendly label for a device."""
    tags = []
    if d["is_remote"]:
        tags.append("RDP")
    elif d["is_bluetooth"]:
        tags.append("BT")
    if d["is_default"]:
        tags.append("default")
    suffix = f"  [{'/'.join(tags)}]" if tags else ""
    return f"{_truncate(d['name'])}{suffix}"


class MicSelector(ctk.CTkFrame):
    """Dropdown to pick the input device. 'Refresh' is the last menu entry."""

    def __init__(
        self,
        parent,
        on_device_changed: Callable[[int | None], None] | None = None,
        content_padx: int = PADDING,
        **kwargs,
    ):
        super().__init__(parent, fg_color="transparent", **kwargs)
        self._on_device_changed = on_device_changed
        self._content_padx = content_padx
        self._devices: list[dict] = []
        self._labels: list[str] = []

        ctk.CTkLabel(
            self,
            text="Microphone",
            font=(FONT_FAMILY, FONT_SIZE_HEADING, "bold"),
            text_color=TEXT_PRIMARY,
            anchor="w",
        ).pack(fill="x", padx=self._content_padx, pady=(8, 4))

        self._dropdown = CTkDropdown(
            self,
            values=[_AUTO_LABEL],
            font=(FONT_FAMILY, FONT_SIZE_BODY),
            fg_color=DARK_CARD,
            button_color=BTN_NEUTRAL_BG,
            button_hover_color=BTN_NEUTRAL_HOVER,
            dropdown_fg_color=DARK_CARD,
            dropdown_hover_color=BTN_NEUTRAL_BG,
            corner_radius=CORNER_RADIUS,
            command=self._on_pick,
        )
        self._dropdown.pack(fill="x", padx=self._content_padx, pady=(0, 8))

        self.refresh()

    def refresh(self) -> None:
        """Re-enumerate input devices and update the dropdown."""
        self._devices = list_input_devices()
        self._labels = (
            [_AUTO_LABEL]
            + [_device_label(d) for d in self._devices]
            + [_REFRESH_LABEL]
        )
        current = self._dropdown.get()
        self._dropdown.configure(values=self._labels)
        # Don't keep "Refresh devices" as the visible value
        if current == _REFRESH_LABEL or current not in self._labels:
            self._dropdown.set(_AUTO_LABEL)
        else:
            self._dropdown.set(current)

    def _on_pick(self, label: str) -> None:
        if label == _REFRESH_LABEL:
            # Pop the menu back to whatever was active before & re-enumerate.
            self.refresh()
            return
        if not self._on_device_changed:
            return
        if label == _AUTO_LABEL:
            self._on_device_changed(None)
            return
        for d in self._devices:
            if _device_label(d) == label:
                self._on_device_changed(d["id"])
                return

    def set_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        self._dropdown.configure(state=state)
