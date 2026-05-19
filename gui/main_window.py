"""Main application window composing all GUI panels."""
from __future__ import annotations
import sys
import threading
import json
from pathlib import Path
import customtkinter as ctk
from gui.themes import (
    PAGE_BG, CARD_BG, INPUT_BG, BORDER, BORDER_SOFT,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED,
    BTN_NEUTRAL_BG, BTN_NEUTRAL_HOVER,
    STATE_OK, STATE_BAD,
    FONT_FAMILY, FONT_SIZE_TITLE, FONT_SIZE_HEADING, FONT_SIZE_BODY,
    FONT_SIZE_SMALL, FONT_SIZE_TINY, FONT_SIZE_LABEL,
    SIDEBAR_WIDTH, PADDING, PADDING_LG, CORNER_RADIUS,
    # backward-compat aliases for downstream widgets
    DARK_BG, DARK_BG_SECONDARY, DARK_CARD, CARDINAL_RED, CARDINAL_RED_DARK,
    ACCENT_GREEN, ACCENT_RED, ACCENT_BLUE,
)
from gui.widgets.chat_display import ChatDisplay
from gui.widgets.status_bar import StatusBar
from gui.widgets.patient_selector import PatientSelector
from gui.widgets.tts_selector import TTSSelector
from gui.widgets.mic_selector import MicSelector
from gui.widgets.extraction_panel import ExtractionPanel
from orchestrator import Orchestrator
from phone.controller import PhoneCallController
from phone.event_log import event_path
from patient_data.models import PatientProfile


class MainWindow(ctk.CTk):
    """Root window for the Agentic Voice AI desktop application."""

    def __init__(self):
        super().__init__()

        # Window setup
        self.title("Agentic Voice AI — Lung Cancer Screening")
        self.geometry("1100x720")
        self.minsize(900, 600)

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("dark-blue")
        self.configure(fg_color=DARK_BG)

        # Orchestrator
        self._orchestrator = Orchestrator(self)
        self._bind_orchestrator_events()

        # State
        self._selected_patient: PatientProfile | None = None
        self._call_active = False
        self._real_phone_active = False
        self._phone_controller = PhoneCallController()
        self._phone_event_path: Path | None = None
        self._phone_event_offset = 0

        # Build layout
        self._build_ui()

        # Tint the Windows 11 title bar + set custom app icon (both deferred
        # so the HWND exists / window is realised when the calls run).
        self.after(50, self._apply_titlebar_theme)
        self.after(50, self._apply_app_icon)

        # Preload Whisper + AudioRecorder at startup (background thread)
        self._orchestrator.preload_models()
        self.after(500, self._poll_phone_events)

    # ------------------------------------------------------------------
    # Native window chrome
    # ------------------------------------------------------------------
    def _apply_app_icon(self) -> None:
        """Generate a circular phone-themed icon at runtime and apply it via
        both iconphoto and a real Windows .ico file (Windows uses the latter
        for the title-bar/taskbar). Silently no-ops if Pillow is missing."""
        try:
            from PIL import Image, ImageDraw, ImageFont, ImageTk
            import os, tempfile, sys

            def _render(side: int) -> "Image.Image":
                """Render the 📞 handset silhouette in a single light-brown
                colour on a transparent background.

                Step 1: rasterise the colour emoji so PIL draws its glyph
                shape. Step 2: keep only the alpha channel as a mask and
                paint a flat colour through it. We get the modern receiver
                outline without the emoji's red/grey colours.
                """
                emoji_font = None
                for path in (
                    r"C:\Windows\Fonts\seguiemj.ttf",
                    "/System/Library/Fonts/Apple Color Emoji.ttc",
                ):
                    if os.path.exists(path):
                        try:
                            emoji_font = ImageFont.truetype(
                                path, int(side * 0.92),
                            )
                            break
                        except Exception:
                            continue
                if emoji_font is None:
                    emoji_font = ImageFont.load_default()

                em = Image.new("RGBA", (side, side), (0, 0, 0, 0))
                d = ImageDraw.Draw(em)
                try:
                    d.text((side / 2, side / 2), "📞",
                           font=emoji_font, anchor="mm", embedded_color=True)
                except TypeError:
                    d.text((side / 2, side / 2), "📞",
                           font=emoji_font, anchor="mm", fill="white")

                alpha = em.split()[3]  # silhouette
                out = Image.new("RGBA", (side, side), (0, 0, 0, 0))
                tint = Image.new("RGBA", (side, side), "#B5B1AD")
                out.paste(tint, mask=alpha)
                return out

            # Tk icon (works cross-platform but Windows often ignores for the
            # title bar — kept as a fallback)
            self._icon_img = ImageTk.PhotoImage(_render(64))
            try:
                self.iconphoto(True, self._icon_img)
            except Exception:
                pass

            # Windows path: write a multi-resolution .ico and apply via
            # the native WM_SETICON message — Tk's iconbitmap on Win11
            # doesn't reliably reach the title bar.
            if sys.platform == "win32":
                ico_path = os.path.join(
                    tempfile.gettempdir(), "gemma_phone_icon.ico",
                )
                base = _render(256)
                base.save(
                    ico_path, format="ICO",
                    sizes=[(16, 16), (24, 24), (32, 32),
                           (48, 48), (64, 64), (128, 128), (256, 256)],
                )
                try:
                    self.iconbitmap(default=ico_path)
                except Exception:
                    pass

                # Native: load ICO as HICON, push to WM_SETICON for both
                # the title-bar icon (small) and Alt-Tab icon (big).
                import ctypes
                from ctypes import wintypes
                LR_LOADFROMFILE = 0x00000010
                LR_DEFAULTSIZE = 0x00000040
                IMAGE_ICON = 1
                WM_SETICON = 0x0080
                ICON_SMALL, ICON_BIG = 0, 1

                user32 = ctypes.windll.user32
                user32.LoadImageW.restype = wintypes.HANDLE
                user32.SendMessageW.restype = wintypes.LPARAM

                hwnd = user32.GetParent(self.winfo_id())
                for size, kind in ((16, ICON_SMALL), (32, ICON_BIG)):
                    hicon = user32.LoadImageW(
                        None, ico_path, IMAGE_ICON,
                        size, size, LR_LOADFROMFILE,
                    )
                    if hicon:
                        user32.SendMessageW(hwnd, WM_SETICON, kind, hicon)
        except Exception as e:
            print(f"  [icon set skipped: {e}]")

    def _apply_titlebar_theme(self) -> None:
        """Tint the Win11 title-bar caption/text/border via DWM.

        Silently no-ops on non-Windows and on Win10 (attributes unsupported).
        """
        if sys.platform != "win32":
            return
        try:
            import ctypes
            from ctypes import wintypes

            def colorref(hex_str: str) -> int:
                h = hex_str.lstrip("#")
                r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
                return (b << 16) | (g << 8) | r  # 0x00BBGGRR

            DWMWA_BORDER_COLOR = 34
            DWMWA_CAPTION_COLOR = 35
            DWMWA_TEXT_COLOR = 36

            hwnd = ctypes.windll.user32.GetParent(self.winfo_id())
            dwm = ctypes.windll.dwmapi
            for attr, hex_color in (
                (DWMWA_CAPTION_COLOR, CARD_BG),
                (DWMWA_TEXT_COLOR, TEXT_PRIMARY),
                (DWMWA_BORDER_COLOR, BORDER),
            ):
                value = wintypes.DWORD(colorref(hex_color))
                dwm.DwmSetWindowAttribute(
                    hwnd, attr, ctypes.byref(value), ctypes.sizeof(value),
                )
        except Exception:
            pass  # older Windows builds will ignore the call

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        # ── Custom header: just the underlined tabs, centered ─────────
        header = ctk.CTkFrame(self, fg_color=PAGE_BG, corner_radius=0, height=56)
        header.pack(fill="x", side="top", padx=PADDING_LG * 2, pady=(PADDING_LG, 0))
        header.pack_propagate(False)

        tabs = ctk.CTkFrame(header, fg_color="transparent")
        tabs.pack(expand=True, fill="y")
        self._tab_buttons: dict[str, dict] = {}
        for name in ("Clinical Note Extraction", "Voice Call Agent"):
            cell = ctk.CTkFrame(tabs, fg_color="transparent", width=180)
            cell.pack(side="left", fill="y", padx=24)
            cell.pack_propagate(False)
            label = ctk.CTkLabel(
                cell, text=name.upper(),
                font=(FONT_FAMILY, FONT_SIZE_SMALL, "bold"),
                text_color=TEXT_MUTED, cursor="hand2",
            )
            label.pack(side="top", expand=True)
            underline = ctk.CTkFrame(cell, fg_color=PAGE_BG, height=2, corner_radius=0)
            underline.pack(side="bottom", fill="x")
            label.bind("<Button-1>", lambda _e, n=name: self._switch_tab(n))
            self._tab_buttons[name] = {"label": label, "underline": underline}

        # Thin divider under header
        ctk.CTkFrame(self, fg_color=BORDER_SOFT, height=1, corner_radius=0).pack(
            fill="x", padx=PADDING_LG * 2, pady=(8, 0),
        )

        # ── Page container (stacked frames, only one visible) ─────────
        pages = ctk.CTkFrame(self, fg_color=PAGE_BG, corner_radius=0)
        pages.pack(fill="both", expand=True)

        # History Extraction page
        self._page_extract = ExtractionPanel(pages)
        # Voice Call page
        self._page_call = ctk.CTkFrame(pages, fg_color=PAGE_BG, corner_radius=0)

        # Default tab
        self._active_tab: str = "Clinical Note Extraction"
        self._switch_tab("Clinical Note Extraction")

        # Below: original voice-call layout lives on _page_call
        content = ctk.CTkFrame(self._page_call, fg_color=PAGE_BG, corner_radius=0)
        content.pack(fill="both", expand=True)

        # --- Sidebar (scrollable so overflow content stays reachable) ---
        sidebar = ctk.CTkScrollableFrame(
            content, fg_color=DARK_BG_SECONDARY,
            width=SIDEBAR_WIDTH, corner_radius=0,
            scrollbar_button_color=DARK_CARD,
            scrollbar_button_hover_color=CARDINAL_RED,
        )
        sidebar.pack(side="left", fill="y")

        # TTS selector
        self._tts_selector = TTSSelector(
            sidebar,
            on_config_changed=self._on_tts_changed,
        )
        self._tts_selector.pack(fill="x")

        # Patient selector
        self._patient_selector = PatientSelector(
            sidebar,
            on_patient_selected=self._on_patient_selected,
        )
        self._patient_selector.pack(fill="x", pady=(PADDING, 0))

        # PC Testing: existing local-microphone workflow.
        pc_testing = ctk.CTkFrame(
            sidebar,
            fg_color="transparent",
            corner_radius=CORNER_RADIUS,
            border_width=1,
            border_color=DARK_CARD,
        )
        pc_testing.pack(fill="x", padx=PADDING, pady=(PADDING * 2, 0))

        pc_inner = ctk.CTkFrame(pc_testing, fg_color="transparent")
        pc_inner.pack(fill="x", padx=12, pady=12)

        ctk.CTkLabel(
            pc_inner,
            text="PC Testing",
            font=(FONT_FAMILY, FONT_SIZE_HEADING, "bold"),
            text_color=TEXT_PRIMARY,
            anchor="w",
        ).pack(fill="x", pady=(0, 10))

        self._mic_selector = MicSelector(
            pc_inner,
            on_device_changed=self._on_mic_device_changed,
            content_padx=0,
        )
        self._mic_selector.pack(fill="x")

        pc_controls = ctk.CTkFrame(pc_inner, fg_color="transparent")
        pc_controls.pack(fill="x", pady=(0, 0))

        self._pc_start_btn = ctk.CTkButton(
            pc_controls, text="\U0001F4DE  Call",
            font=(FONT_FAMILY, FONT_SIZE_BODY, "bold"),
            fg_color=ACCENT_GREEN,
            hover_color="#388E3C",
            text_color="white",
            height=36,
            corner_radius=CORNER_RADIUS,
            command=self._start_call,
            state="disabled",
        )
        self._pc_start_btn.pack(side="left", fill="x", expand=True, padx=(0, 6))

        self._pc_end_btn = ctk.CTkButton(
            pc_controls, text="\u2715",
            font=(FONT_FAMILY, FONT_SIZE_BODY, "bold"),
            fg_color=ACCENT_RED,
            hover_color="#C62828",
            text_color="white",
            height=36,
            corner_radius=CORNER_RADIUS,
            command=self._end_call,
            state="disabled",
        )
        self._pc_end_btn.pack(side="left")
        self.after(50, lambda: self._apply_split_widths(
            pc_controls, self._pc_start_btn, self._pc_end_btn,
        ))

        # Real Phone Testing: UI shell for the upcoming Twilio workflow.
        phone_testing = ctk.CTkFrame(
            sidebar,
            fg_color="transparent",
            corner_radius=CORNER_RADIUS,
            border_width=1,
            border_color=DARK_CARD,
        )
        phone_testing.pack(fill="x", padx=PADDING, pady=(PADDING, 0))

        phone_inner = ctk.CTkFrame(phone_testing, fg_color="transparent")
        phone_inner.pack(fill="x", padx=12, pady=12)

        ctk.CTkLabel(
            phone_inner,
            text="Real Phone Testing",
            font=(FONT_FAMILY, FONT_SIZE_HEADING, "bold"),
            text_color=TEXT_PRIMARY,
            anchor="w",
        ).pack(fill="x", pady=(0, 10))

        self._phone_number_var = ctk.StringVar()
        self._phone_number_entry = ctk.CTkEntry(
            phone_inner,
            textvariable=self._phone_number_var,
            placeholder_text="+1 555 123 4567",
            fg_color=INPUT_BG,
            border_color=BORDER,
            text_color=TEXT_PRIMARY,
            placeholder_text_color=TEXT_MUTED,
            font=(FONT_FAMILY, FONT_SIZE_BODY),
            height=32,
        )
        self._phone_number_entry.pack(fill="x", pady=(0, 8))
        self._phone_number_var.trace_add("write", self._on_phone_number_changed)

        phone_controls = ctk.CTkFrame(phone_inner, fg_color="transparent")
        phone_controls.pack(fill="x")

        self._phone_start_btn = ctk.CTkButton(
            phone_controls, text="\U0001F4DE  Call",
            font=(FONT_FAMILY, FONT_SIZE_BODY, "bold"),
            fg_color=ACCENT_GREEN,
            hover_color="#388E3C",
            text_color="white",
            height=36,
            corner_radius=CORNER_RADIUS,
            command=self._start_real_phone_call,
            state="disabled",
        )
        self._phone_start_btn.pack(side="left", fill="x", expand=True, padx=(0, 6))

        self._phone_end_btn = ctk.CTkButton(
            phone_controls, text="\u2715",
            font=(FONT_FAMILY, FONT_SIZE_BODY, "bold"),
            fg_color=ACCENT_RED,
            hover_color="#C62828",
            text_color="white",
            height=36,
            corner_radius=CORNER_RADIUS,
            command=self._end_real_phone_call,
            state="disabled",
        )
        self._phone_end_btn.pack(side="left")
        self.after(50, lambda: self._apply_split_widths(
            phone_controls, self._phone_start_btn, self._phone_end_btn,
        ))

        # This is a phone agent — patient input is always live mic.
        self._orchestrator.set_input_mode("voice")

        # --- Center panel ---
        center = ctk.CTkFrame(content, fg_color=DARK_BG, corner_radius=0)
        center.pack(side="left", fill="both", expand=True)

        # Chat display (full height — no input panel; this is a phone agent,
        # patient responses arrive via the mic during a call).
        self._chat = ChatDisplay(center)
        self._chat.pack(fill="both", expand=True, padx=2, pady=2)

        # Status bar
        self._status_bar = StatusBar(self)
        # Voice-call telemetry; only shown on the Voice Call tab.
        # Default tab is "Clinical Note Extraction", so start hidden.
        # _on_tab_changed pack()s/pack_forget()s on tab switches.

    # ------------------------------------------------------------------
    # Event handlers — UI interactions
    # ------------------------------------------------------------------

    def _on_patient_selected(self, patient: PatientProfile) -> None:
        self._selected_patient = patient
        if not self._call_active:
            self._pc_start_btn.configure(state="normal")
        self._refresh_real_phone_controls()

    def _on_tts_changed(self, engine: str, voice: str, api_key: str) -> None:
        self._orchestrator.set_tts_config(engine, voice, api_key)

    def _on_mic_device_changed(self, device_id: int | None) -> None:
        self._orchestrator.set_mic_device(device_id)

    def _on_phone_number_changed(self, *_args) -> None:
        self._refresh_real_phone_controls()

    def _refresh_real_phone_controls(self) -> None:
        can_start = (
            self._selected_patient is not None
            and bool(self._phone_number_var.get().strip())
            and not self._real_phone_active
        )
        self._phone_start_btn.configure(state="normal" if can_start else "disabled")
        self._phone_end_btn.configure(state="normal" if self._real_phone_active else "disabled")

    def _switch_tab(self, name: str) -> None:
        """Show the page for `name` and update tab underline/text styling."""
        self._active_tab = name
        # Toggle pages
        for page in (self._page_extract, self._page_call):
            page.pack_forget()
        if name == "Clinical Note Extraction":
            self._page_extract.pack(fill="both", expand=True,
                                    padx=PADDING_LG * 2, pady=PADDING_LG)
        else:
            self._page_call.pack(fill="both", expand=True)
        # Tab styling
        for tab_name, parts in self._tab_buttons.items():
            active = (tab_name == name)
            parts["label"].configure(
                text_color=TEXT_PRIMARY if active else TEXT_MUTED,
            )
            parts["underline"].configure(
                fg_color=TEXT_PRIMARY if active else PAGE_BG,
            )
        # Status bar only on Voice Call tab (may not exist yet during init)
        bar = getattr(self, "_status_bar", None)
        if bar is not None:
            if name == "Voice Call Agent":
                bar.pack(fill="x", side="bottom")
            else:
                bar.pack_forget()

    def _start_call(self) -> None:
        if not self._selected_patient:
            return
        self._call_active = True
        self._chat.clear()
        self._pc_start_btn.configure(state="disabled")
        self._pc_end_btn.configure(state="normal")
        self._patient_selector.set_enabled(False)
        self._tts_selector.set_enabled(False)
        self._mic_selector.set_enabled(False)
        self._phone_number_entry.configure(state="disabled")
        self._phone_start_btn.configure(state="disabled")

        self._orchestrator.start_call(self._selected_patient)

    def _end_call(self) -> None:
        self._orchestrator.stop_call()

    def _start_real_phone_call(self) -> None:
        """Start an outbound Twilio call on a worker thread."""
        phone = self._phone_number_var.get().strip()
        if not self._selected_patient or not phone:
            return
        self._real_phone_active = True
        self._chat.clear()
        self._phone_start_btn.configure(state="disabled")
        self._phone_end_btn.configure(state="normal")
        self._patient_selector.set_enabled(False)
        self._tts_selector.set_enabled(False)
        self._phone_number_entry.configure(state="disabled")
        self._pc_start_btn.configure(state="disabled")
        self._chat.add_message(
            "system",
            f"Placing real phone call to {self._mask_phone_number(phone)}...",
        )
        threading.Thread(
            target=self._place_real_phone_call,
            args=(phone,),
            daemon=True,
        ).start()

    def _end_real_phone_call(self) -> None:
        try:
            self._phone_controller.end_call()
            self._chat.add_message("system", "Real phone call ended.")
        except Exception as e:
            self._chat.add_message("system", f"Phone hang-up error: {e}")
        finally:
            self._real_phone_active = False
            self._patient_selector.set_enabled(True)
            self._tts_selector.set_enabled(True)
            self._phone_number_entry.configure(state="normal")
            self._pc_start_btn.configure(
                state="normal" if self._selected_patient else "disabled"
            )
            self._refresh_real_phone_controls()

    def _place_real_phone_call(self, phone: str) -> None:
        try:
            sid = self._phone_controller.start_call(
                to_number=phone,
                patient_id=self._selected_patient.patient_id,
                voice_name=self._tts_selector.voice,
            )
            def _on_started():
                self._phone_event_path = event_path(sid)
                self._phone_event_offset = 0
                self._chat.add_message("system", "Real phone call started.")
            self.after(0, _on_started)
        except Exception as e:
            def _recover():
                self._chat.add_message("system", f"Phone call error: {e}")
                self._real_phone_active = False
                self._patient_selector.set_enabled(True)
                self._tts_selector.set_enabled(True)
                self._phone_number_entry.configure(state="normal")
                self._pc_start_btn.configure(
                    state="normal" if self._selected_patient else "disabled"
                )
                self._refresh_real_phone_controls()
            self.after(0, _recover)

    def _poll_phone_events(self) -> None:
        """Tail phone-server events into the desktop chat pane."""
        try:
            if self._phone_event_path is not None and self._phone_event_path.exists():
                with self._phone_event_path.open("r", encoding="utf-8") as f:
                    f.seek(self._phone_event_offset)
                    for line in f:
                        event = json.loads(line)
                        kind = event.get("kind")
                        text = event.get("text", "")
                        if kind == "assistant" and text:
                            self._chat.add_message("assistant", text)
                        elif kind == "user" and text:
                            self._chat.add_message("user", f"[Phone] {text}")
                        elif kind == "call_ended":
                            self._real_phone_active = False
                            self._phone_controller.call_sid = None
                            self._patient_selector.set_enabled(True)
                            self._tts_selector.set_enabled(True)
                            self._phone_number_entry.configure(state="normal")
                            self._pc_start_btn.configure(
                                state="normal" if self._selected_patient else "disabled"
                            )
                            self._refresh_real_phone_controls()
                            self._chat.add_message("system", "Real phone call ended.")
                            self._phone_event_path = None
                            self._phone_event_offset = 0
                    self._phone_event_offset = f.tell()
        except Exception as e:
            print(f"  [phone event poll error: {e}]")
        finally:
            self.after(500, self._poll_phone_events)

    @staticmethod
    def _mask_phone_number(phone: str) -> str:
        """Keep only the first visible digits; hide the final seven for privacy."""
        raw = phone.strip()
        visible = max(len(raw) - 7, 0)
        return raw[:visible] + ("#" * (len(raw) - visible))

    @staticmethod
    def _apply_split_widths(container, primary_btn, secondary_btn) -> None:
        """Force a true 75/25 split after Tk has measured the container."""
        container.update_idletasks()
        total = max(container.winfo_width() - 6, 0)
        primary_btn.configure(width=int(total * 0.75))
        secondary_btn.configure(width=int(total * 0.25))

    # ------------------------------------------------------------------
    # Event handlers — Orchestrator events (called on GUI thread)
    # ------------------------------------------------------------------

    def _bind_orchestrator_events(self) -> None:
        o = self._orchestrator
        o.on("call_started", self._evt_call_started)
        o.on("assistant_message", self._evt_assistant_message)
        o.on("user_message", self._evt_user_message)
        o.on("phase_change", self._evt_phase_change)
        o.on("gap_update", self._evt_gap_update)
        o.on("confidence_update", self._evt_confidence_update)
        o.on("turn_update", self._evt_turn_update)
        o.on("thinking_start", self._evt_thinking_start)
        o.on("thinking_end", self._evt_thinking_end)
        o.on("tts_start", self._evt_tts_start)
        o.on("tts_end", self._evt_tts_end)
        o.on("listening_start", self._evt_listening_start)
        o.on("listening_end", self._evt_listening_end)
        o.on("transcription_result", self._evt_transcription_result)
        o.on("call_ended", self._evt_call_ended)
        o.on("error", self._evt_error)
        o.on("status_message", self._evt_status_message)

    def _evt_call_started(self) -> None:
        self._chat.add_message("system", "Call started. Initializing agent...")

    def _evt_assistant_message(self, text: str = "") -> None:
        self._chat.add_message("assistant", text)

    def _evt_user_message(self, text: str = "") -> None:
        # Handled by _evt_transcription_result (mic input).
        pass

    def _evt_phase_change(self, phase: str = "") -> None:
        self._status_bar.update_phase(phase)

    def _evt_gap_update(self, resolved: int = 0, total: int = 0) -> None:
        self._status_bar.update_gaps(resolved, total)

    def _evt_confidence_update(self, value: float = 0.0) -> None:
        self._status_bar.update_confidence(value)

    def _evt_turn_update(self, turn: int = 0) -> None:
        self._status_bar.update_turn(turn)

    def _evt_thinking_start(self) -> None:
        self._chat.show_thinking()

    def _evt_thinking_end(self) -> None:
        self._chat.hide_thinking()

    def _evt_tts_start(self) -> None:
        self._status_bar.set_speaking(True)

    def _evt_tts_end(self) -> None:
        self._status_bar.set_speaking(False)

    def _evt_listening_start(self) -> None:
        self._status_bar.set_listening(True)

    def _evt_listening_end(self) -> None:
        self._status_bar.set_listening(False)

    def _evt_transcription_result(self, text: str = "") -> None:
        if text:
            self._chat.add_message("user", f"[Mic] {text}")

    def _evt_call_ended(self, transcript_path: str = "", summary_path: str = "") -> None:
        self._call_active = False
        self._pc_start_btn.configure(state="normal" if self._selected_patient else "disabled")
        self._pc_end_btn.configure(state="disabled")
        self._patient_selector.set_enabled(True)
        self._tts_selector.set_enabled(True)
        self._mic_selector.set_enabled(True)
        self._phone_number_entry.configure(state="normal")
        self._refresh_real_phone_controls()

        end_text = "Call ended."
        if transcript_path:
            end_text += f"\nTranscript: {transcript_path}"
        if summary_path:
            end_text += f"\nSummary: {summary_path}"
        self._chat.add_message("system", end_text)

    def _evt_error(self, message: str = "") -> None:
        self._chat.add_message("system", f"Error: {message}")

    def _evt_status_message(self, text: str = "") -> None:
        self._chat.add_message("system", text)
