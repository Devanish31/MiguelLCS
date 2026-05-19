"""Scrollable chat display with message bubbles and thinking indicator."""
from __future__ import annotations
import customtkinter as ctk
from gui.themes import (
    DARK_BG, DARK_CARD, USER_BUBBLE, ASSISTANT_BUBBLE, SYSTEM_BUBBLE,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED,
    FONT_FAMILY, FONT_SIZE_BODY, FONT_SIZE_SMALL, FONT_SIZE_TINY,
    PADDING, CORNER_RADIUS,
)


class ChatBubble(ctk.CTkFrame):
    """A single chat message bubble."""

    def __init__(self, parent, role: str, text: str, **kwargs):
        bg_color = {
            "assistant": ASSISTANT_BUBBLE,
            "user": USER_BUBBLE,
            "system": SYSTEM_BUBBLE,
        }.get(role, DARK_CARD)

        super().__init__(parent, fg_color=bg_color, corner_radius=CORNER_RADIUS, **kwargs)

        # Role label
        role_text = {"assistant": "AI Agent", "user": "Patient", "system": "System"}.get(role, role)
        role_label = ctk.CTkLabel(
            self,
            text=role_text,
            font=(FONT_FAMILY, FONT_SIZE_TINY, "bold"),
            text_color=TEXT_SECONDARY,
            anchor="w",
        )
        role_label.pack(fill="x", padx=PADDING, pady=(6, 0))

        # Message text
        msg_label = ctk.CTkLabel(
            self,
            text=text,
            font=(FONT_FAMILY, FONT_SIZE_BODY),
            text_color=TEXT_PRIMARY,
            anchor="w",
            justify="left",
            wraplength=500,
        )
        msg_label.pack(fill="x", padx=PADDING, pady=(2, 8))


class ChatDisplay(ctk.CTkFrame):
    """Scrollable chat area with message bubbles."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, fg_color=DARK_BG, **kwargs)

        # Scrollable frame for messages
        self._scroll = ctk.CTkScrollableFrame(
            self,
            fg_color=DARK_BG,
            corner_radius=0,
        )
        self._scroll.pack(fill="both", expand=True, padx=2, pady=2)

        # Thinking indicator (hidden by default)
        self._thinking_label = None

    def add_message(self, role: str, text: str) -> None:
        """Add a chat bubble to the display."""
        self._remove_thinking()

        anchor = "e" if role == "user" else "w"
        bubble = ChatBubble(self._scroll, role, text)
        bubble.pack(
            fill="x",
            padx=(80 if role == "user" else PADDING, PADDING if role == "user" else 80),
            pady=4,
            anchor=anchor,
        )
        # Auto-scroll to bottom
        self._scroll.after(50, lambda: self._scroll._parent_canvas.yview_moveto(1.0))

    def show_thinking(self) -> None:
        """Show an animated 'Thinking...' indicator."""
        if self._thinking_label is not None:
            return
        self._thinking_label = ctk.CTkLabel(
            self._scroll,
            text="  Thinking...",
            font=(FONT_FAMILY, FONT_SIZE_SMALL, "italic"),
            text_color=TEXT_MUTED,
            anchor="w",
        )
        self._thinking_label.pack(fill="x", padx=PADDING, pady=4)
        self._scroll.after(50, lambda: self._scroll._parent_canvas.yview_moveto(1.0))

    def hide_thinking(self) -> None:
        """Hide the thinking indicator."""
        self._remove_thinking()

    def _remove_thinking(self) -> None:
        if self._thinking_label is not None:
            self._thinking_label.destroy()
            self._thinking_label = None

    def clear(self) -> None:
        """Clear all messages."""
        for widget in self._scroll.winfo_children():
            widget.destroy()
        self._thinking_label = None
