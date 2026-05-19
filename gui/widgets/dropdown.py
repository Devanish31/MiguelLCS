"""CTkOptionMenu replacement with a custom-styled popup.

CTkOptionMenu wraps tkinter.Menu, which on Windows draws a native popup with
its own border and auto-sized width. This widget rebuilds the popup as a
borderless Toplevel + CTkFrame so the dropdown's border matches the rest of
the app and its width is locked to the trigger.
"""
from __future__ import annotations
from typing import Callable, Sequence
import customtkinter as ctk
import tkinter as tk

from gui.themes import (
    DARK_CARD, BORDER, BTN_NEUTRAL_BG, BTN_NEUTRAL_HOVER,
    TEXT_PRIMARY,
    FONT_FAMILY, FONT_SIZE_BODY, CORNER_RADIUS,
)


class CTkDropdown(ctk.CTkFrame):
    """Drop-in replacement for CTkOptionMenu with controllable popup chrome."""

    def __init__(
        self,
        parent,
        values: Sequence[str] = (),
        command: Callable[[str], None] | None = None,
        font=None,
        fg_color=DARK_CARD,
        button_color=BTN_NEUTRAL_BG,
        button_hover_color=BTN_NEUTRAL_HOVER,
        dropdown_fg_color=DARK_CARD,
        dropdown_hover_color=BTN_NEUTRAL_BG,
        text_color=TEXT_PRIMARY,
        corner_radius=CORNER_RADIUS,
        height=28,
        **kwargs,
    ):
        super().__init__(parent, fg_color="transparent", **kwargs)
        self._values: list[str] = list(values)
        self._command = command
        self._current: str = self._values[0] if self._values else ""
        self._popup: tk.Toplevel | None = None
        self._click_binding = None
        self._enabled = True
        self._font = font or (FONT_FAMILY, FONT_SIZE_BODY)
        self._fg_color = fg_color
        self._button_color = button_color
        self._button_hover_color = button_hover_color
        self._dropdown_fg_color = dropdown_fg_color
        self._dropdown_hover_color = dropdown_hover_color
        self._text_color = text_color
        self._corner_radius = corner_radius

        # Trigger row: text label (left) + chevron button (right), sharing
        # a single rounded background.
        self._trigger = ctk.CTkFrame(
            self, fg_color=fg_color, corner_radius=corner_radius,
            height=height,
        )
        self._trigger.pack(fill="x")
        self._trigger.pack_propagate(False)

        self._label = ctk.CTkLabel(
            self._trigger,
            text=self._current,
            font=self._font,
            text_color=text_color,
            anchor="w", cursor="hand2",
        )
        self._label.pack(side="left", fill="both", expand=True, padx=(10, 0))
        self._label.bind("<Button-1>", lambda _e: self._toggle())
        self._trigger.bind("<Button-1>", lambda _e: self._toggle())

        self._chevron = ctk.CTkButton(
            self._trigger,
            text="▾", width=28,
            font=(FONT_FAMILY, FONT_SIZE_BODY, "bold"),
            fg_color=button_color, hover_color=button_hover_color,
            text_color=text_color,
            corner_radius=corner_radius,
            command=self._toggle,
        )
        self._chevron.pack(side="right", fill="y")

    # ------------------------------------------------------------------
    # Public CTkOptionMenu-compatible API
    # ------------------------------------------------------------------
    def set(self, value: str) -> None:
        self._current = value
        self._label.configure(text=value)

    def get(self) -> str:
        return self._current

    def configure(self, **kwargs):
        if "values" in kwargs:
            self._values = list(kwargs.pop("values"))
        if "command" in kwargs:
            self._command = kwargs.pop("command")
        if "state" in kwargs:
            state = kwargs.pop("state")
            self._enabled = (state == "normal")
            self._chevron.configure(state=state)
        if kwargs:
            super().configure(**kwargs)

    # ------------------------------------------------------------------
    # Popup lifecycle
    # ------------------------------------------------------------------
    def _toggle(self) -> None:
        if not self._enabled:
            return
        if self._popup is not None:
            self._close()
        else:
            self._open()

    def _open(self) -> None:
        if not self._values:
            return
        self.update_idletasks()
        x = self._trigger.winfo_rootx()
        y = self._trigger.winfo_rooty() + self._trigger.winfo_height() + 2
        w = self._trigger.winfo_width()

        self._popup = tk.Toplevel(self)
        self._popup.overrideredirect(True)
        self._popup.configure(bg=BORDER)
        try:
            self._popup.attributes("-topmost", True)
        except Exception:
            pass

        inner = ctk.CTkFrame(
            self._popup,
            fg_color=self._dropdown_fg_color,
            corner_radius=self._corner_radius,
            border_width=1, border_color=BORDER,
        )
        inner.pack(fill="both", expand=True)

        for v in self._values:
            row = ctk.CTkButton(
                inner, text=v,
                font=self._font,
                fg_color="transparent",
                hover_color=self._dropdown_hover_color,
                text_color=self._text_color,
                corner_radius=4,
                anchor="w", height=28,
                command=lambda val=v: self._select(val),
            )
            row.pack(fill="x", padx=4, pady=1)

        self._popup.update_idletasks()
        h = self._popup.winfo_reqheight()
        self._popup.geometry(f"{w}x{h}+{x}+{y}")

        # Defer outside-click arming so the opening click isn't caught here.
        self.after(50, self._arm_outside_click)
        self._popup.bind("<Escape>", lambda _e: self._close())

    def _arm_outside_click(self) -> None:
        if self._popup is None:
            return
        top = self.winfo_toplevel()
        self._click_binding = top.bind("<Button-1>", self._on_root_click, add="+")

    def _on_root_click(self, event) -> None:
        if self._popup is None:
            return
        # Ignore clicks on our own trigger so the chevron/label can toggle.
        w = event.widget
        while w is not None:
            if w is self._trigger:
                return
            try:
                w = w.master
            except Exception:
                break
        # Outside the popup geometry → close
        px, py = self._popup.winfo_rootx(), self._popup.winfo_rooty()
        pw, ph = self._popup.winfo_width(), self._popup.winfo_height()
        if not (px <= event.x_root <= px + pw and py <= event.y_root <= py + ph):
            self._close()

    def _close(self) -> None:
        if self._popup is None:
            return
        try:
            if self._click_binding:
                self.winfo_toplevel().unbind("<Button-1>", self._click_binding)
        except Exception:
            pass
        try:
            self._popup.destroy()
        except Exception:
            pass
        self._popup = None
        self._click_binding = None

    def _select(self, value: str) -> None:
        self._current = value
        self._label.configure(text=value)
        self._close()
        if self._command:
            self._command(value)
