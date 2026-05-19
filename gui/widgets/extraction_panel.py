"""Smoking-history extraction panel — vertical "result-forward" layout.

Single column, top-to-bottom:
  1. System prompt (collapsible, collapsed by default)
  2. Clinical note (textarea)
  3. Extract & Parse — single primary CTA
  4. Raw model output (collapsible, collapsed by default)
  5. Parsed fields — 4 horizontal stat tiles (HERO row)
"""
from __future__ import annotations
import re
import threading
import time
import customtkinter as ctk

from gui.themes import (
    PAGE_BG, CARD_BG, CARD_BG_ALT, INPUT_BG, BORDER, BORDER_SOFT,
    TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED, TEXT_INVERSE,
    BTN_NEUTRAL_BG, BTN_NEUTRAL_HOVER, BTN_BRIGHT_BG, BTN_BRIGHT_HOVER,
    STATE_OK, STATE_OK_TEXT, STATE_WARN, STATE_WARN_TEXT,
    STATE_BAD, STATE_BAD_TEXT,
    FONT_FAMILY, FONT_FAMILY_MONO,
    FONT_SIZE_TITLE, FONT_SIZE_HEADING, FONT_SIZE_BODY,
    FONT_SIZE_SMALL, FONT_SIZE_TINY, FONT_SIZE_LABEL,
    PADDING, PADDING_LG, CORNER_RADIUS,
)
from agent.extraction import (
    PATIENT_LEVEL_PROMPT, run_extraction, parse_extraction,
    extract_age_from_note,
)
from config import GeminiConfig


_SAMPLE_NOTES: list[str] = [
    "[2022-05-10] 60 y/o male, established patient. Active smoker — "
    "approximately 1 pack per day x 30 years. Discussed cessation options.",
    "[2023-08-22] 61 y/o male. Reports cutting back to half a pack per "
    "day. Plans to quit in the next few months.",
    "[2024-03-15] 62 y/o male presents for follow-up. Patient reports "
    "successfully quitting smoking 4 months ago. 30 pack-years lifetime.",
]


def _label(parent, text, **kw):
    return ctk.CTkLabel(
        parent, text=text.upper(),
        font=(FONT_FAMILY, FONT_SIZE_LABEL, "bold"),
        text_color=TEXT_SECONDARY, anchor="w", **kw,
    )


def _card(parent, **kw):
    return ctk.CTkFrame(
        parent, fg_color=CARD_BG, corner_radius=CORNER_RADIUS,
        border_width=1, border_color=BORDER, **kw,
    )


class _NoteEntry(ctk.CTkFrame):
    """Clinical-note entry rendered as a single bordered "pill":
    textbox fills the left, × sits flush at the right edge inside the same
    border. The date '[YYYY-MM-DD]' is part of the body text and is sent to
    the LLM verbatim.
    """

    def __init__(self, parent, note_value: str = "", on_remove=None):
        # Outer frame owns the border + corner radius. fg_color matches the
        # textbox interior so the button area blends into the same surface.
        super().__init__(
            parent,
            fg_color=INPUT_BG,
            border_width=1, border_color=BORDER,
            corner_radius=4,
        )
        self._on_remove = on_remove

        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=0)

        self.body = ctk.CTkTextbox(
            self, font=(FONT_FAMILY, FONT_SIZE_BODY),
            fg_color=INPUT_BG, text_color=TEXT_PRIMARY,
            border_width=0, corner_radius=0,
            wrap="word", height=40,
            undo=True, maxundo=-1,
        )
        self.body.grid(row=0, column=0, sticky="nsew", padx=(2, 0), pady=2)
        if note_value:
            self.body.insert("1.0", note_value)
            self.body._textbox.edit_reset()

        # × rendered as a borderless, transparent button sitting inside the
        # same bordered surface. Colour matches the scrollbar (BTN_NEUTRAL_BG).
        self.remove_btn = ctk.CTkButton(
            self, text="×", width=24, height=40,
            font=(FONT_FAMILY, FONT_SIZE_TITLE, "bold"),
            fg_color=INPUT_BG, hover_color=PAGE_BG,
            text_color=BTN_NEUTRAL_BG,
            corner_radius=0, border_width=0,
            command=self._handle_remove,
        )
        self.remove_btn.grid(row=0, column=1, sticky="ns", padx=(0, 2), pady=2)

    def _handle_remove(self) -> None:
        if self._on_remove:
            self._on_remove(self)

    def set_removable(self, allowed: bool) -> None:
        self.remove_btn.configure(state="normal" if allowed else "disabled")

    def set_processing(self, processing: bool) -> None:
        st = "disabled" if processing else "normal"
        self.body.configure(state=st,
                            text_color=TEXT_MUTED if processing else TEXT_PRIMARY)

    def formatted(self) -> str | None:
        """Return the note body verbatim (already includes the date), or None."""
        body = self.body.get("1.0", "end").strip()
        return body or None


class _Collapsible:
    """Click-to-expand strip drawn as a *single* bordered card.

    Collapsed: only the header row is visible.
    Expanded:  a thin internal divider + the body appear inside the same card.
    """

    def __init__(self, parent, summary_fn, build_body):
        self._summary_fn = summary_fn
        self._expanded = False

        # Single outer card — owns the border + rounded corners.
        self.frame = ctk.CTkFrame(
            parent, fg_color=CARD_BG, corner_radius=CORNER_RADIUS,
            border_width=1, border_color=BORDER,
        )

        # Header row — no border, no corner radius of its own.
        self._header_row = ctk.CTkFrame(
            self.frame, fg_color="transparent", height=40,
        )
        # Public alias so callers can drop widgets (e.g. action buttons) into
        # the right side of the header.
        self.header = self._header_row
        self._header_row.pack(fill="x")
        self._header_row.pack_propagate(False)
        self._chevron = ctk.CTkLabel(
            self._header_row, text="▸",
            font=(FONT_FAMILY, FONT_SIZE_BODY),
            text_color=TEXT_SECONDARY, width=24,
        )
        self._chevron.pack(side="left", padx=(PADDING, 0))
        self._summary = ctk.CTkLabel(
            self._header_row, text=self._summary_fn(),
            font=(FONT_FAMILY, FONT_SIZE_LABEL, "bold"),
            text_color=TEXT_SECONDARY, anchor="w", cursor="hand2",
        )
        self._summary.pack(side="left")
        for w in (self._header_row, self._chevron, self._summary):
            w.bind("<Button-1>", lambda _e: self.toggle())

        # Thin internal divider — shown only when expanded.
        self._divider = ctk.CTkFrame(
            self.frame, fg_color=BORDER_SOFT, height=1, corner_radius=0,
        )

        # Body — transparent so the outer card's bg shows through.
        self._body = ctk.CTkFrame(self.frame, fg_color="transparent")
        build_body(self._body)

    def update_summary(self) -> None:
        self._summary.configure(text=self._summary_fn())

    def toggle(self) -> None:
        self._expanded = not self._expanded
        self._chevron.configure(text="▾" if self._expanded else "▸")
        if self._expanded:
            self._divider.pack(fill="x", padx=PADDING)
            self._body.pack(fill="x")
        else:
            self._body.pack_forget()
            self._divider.pack_forget()

    def pack(self, **kw):
        self.frame.pack(**kw)


class ExtractionPanel(ctk.CTkFrame):
    """Vertical, result-forward extraction dashboard."""

    def __init__(self, parent, **kwargs):
        super().__init__(parent, fg_color=PAGE_BG, **kwargs)
        self._raw_output: str = ""
        self._last_elapsed: float | None = None
        self._parsed: dict | None = None
        self._pulse_job: str | None = None  # after()-id of the glow pulse
        self._pulse_phase = False
        self._build()

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------
    def _build(self) -> None:
        # Page-level scroll: the whole extraction view scrolls when its content
        # exceeds the window height (e.g. system-prompt or raw-output expanded,
        # or many notes added). Without this, the bottom rows of stat tiles +
        # screening card were unreachable.
        outer = ctk.CTkScrollableFrame(
            self,
            fg_color="transparent",
            scrollbar_button_color=BTN_NEUTRAL_BG,
            scrollbar_button_hover_color=BTN_NEUTRAL_HOVER,
        )
        outer.pack(fill="both", expand=True)
        outer.grid_columnconfigure(0, weight=1)

        col = ctk.CTkFrame(outer, fg_color="transparent")
        col.grid(row=0, column=0, sticky="nsew", padx=PADDING_LG, pady=PADDING_LG)
        col.grid_columnconfigure(0, weight=1)

        # 1. System prompt — collapsible
        self._prompt_collapsible = _Collapsible(
            col,
            summary_fn=lambda: f"SYSTEM PROMPT  ·  {len(PATIENT_LEVEL_PROMPT)} CHARS  ·  CLICK TO EDIT",
            build_body=self._build_prompt_body,
        )
        self._prompt_collapsible.pack(fill="x")
        # Reset button sits on the right side of the collapsible's header so
        # it doesn't eat a row below the textbox.
        ctk.CTkButton(
            self._prompt_collapsible.header, text="Reset to default",
            width=110, height=24,
            font=(FONT_FAMILY, FONT_SIZE_LABEL, "bold"),
            fg_color=BTN_NEUTRAL_BG, hover_color=BTN_NEUTRAL_HOVER,
            text_color=TEXT_PRIMARY, corner_radius=4,
            command=self._reset_prompt,
        ).pack(side="right", padx=(0, PADDING))

        # 2. Clinical notes — multi-entry, scrollable, capped to ~3 visible
        self._note_card = _card(col)
        self._note_card.pack(fill="x", pady=(PADDING, 0))
        note_inner = ctk.CTkFrame(self._note_card, fg_color="transparent")
        note_inner.pack(fill="both", expand=True, padx=PADDING_LG, pady=PADDING_LG)

        head_row = ctk.CTkFrame(note_inner, fg_color="transparent")
        head_row.pack(fill="x", pady=(0, PADDING // 2))
        _label(head_row, "Clinical notes").pack(side="left")
        ctk.CTkButton(
            head_row, text="+ Add note",
            font=(FONT_FAMILY, FONT_SIZE_LABEL, "bold"),
            fg_color=INPUT_BG, hover_color=BTN_NEUTRAL_BG,
            text_color=TEXT_SECONDARY, border_width=1, border_color=BORDER,
            corner_radius=4, width=110, height=26,
            command=self._add_note_entry,
        ).pack(
            side="right",
            # Right edge aligned with the note edit-box edge:
            # 16 px scrollbar + 10 px entry gutter.
            padx=(0, 26),
        )

        # Hard-cap the scroll area to ~2 entries. CTkScrollableFrame's own
        # `height=` is treated as a hint and grows to fit content, so we wrap
        # it in a fixed-height frame with pack_propagate(False).
        scroll_wrapper = ctk.CTkFrame(note_inner, fg_color="transparent",
                                      height=110)
        scroll_wrapper.pack(fill="x")
        scroll_wrapper.pack_propagate(False)
        self._notes_scroll = ctk.CTkScrollableFrame(
            scroll_wrapper,
            fg_color=CARD_BG, border_width=0,
            scrollbar_button_color=BTN_NEUTRAL_BG,
            scrollbar_button_hover_color=BTN_NEUTRAL_HOVER,
        )
        self._notes_scroll.pack(fill="both", expand=True)

        self._note_entries: list[_NoteEntry] = []
        for body in _SAMPLE_NOTES:
            self._add_note_entry(body)

        # 3. Single primary CTA
        cta_row = ctk.CTkFrame(col, fg_color="transparent")
        cta_row.pack(fill="x", pady=PADDING)
        cta_row.grid_columnconfigure(0, weight=1)
        cta_row.grid_columnconfigure(2, weight=1)
        self._analyze_btn = ctk.CTkButton(
            cta_row, text="EXTRACT & PARSE",
            font=(FONT_FAMILY, FONT_SIZE_LABEL, "bold"),
            fg_color=BTN_BRIGHT_BG, hover_color=BTN_BRIGHT_HOVER,
            text_color=TEXT_INVERSE, height=44, width=260,
            corner_radius=4,
            command=self._on_analyze,
        )
        self._analyze_btn.grid(row=0, column=1)

        # 4. Raw model output — collapsible (collapsed by default)
        self._raw_collapsible = _Collapsible(
            col,
            summary_fn=self._raw_summary,
            build_body=self._build_raw_body,
        )
        self._raw_collapsible.pack(fill="x")

        # 5. Parsed fields — 4 stat tiles (Age | Status | PY | Yrs since)
        self._tiles_row = ctk.CTkFrame(col, fg_color="transparent")
        self._tiles_row.pack(fill="x", pady=(PADDING, 0))
        self._tiles_row.grid_columnconfigure((0, 1, 2, 3), weight=1, uniform="tile")
        self._tiles: dict[str, dict] = {}
        for i, (key, label) in enumerate([
            ("age",                   "Age"),
            ("smoking_status",        "Status"),
            ("pack_years",            "Pack-years"),
            ("years_since_cessation", "Yrs since cessation"),
        ]):
            self._tiles[key] = self._build_tile(self._tiles_row, label, i)

        # 6. Screening result — full-width hero spanning all 4 tile columns.
        # Height fixed at ~75% of natural so the empty space below the
        # rationale doesn't dominate.
        self._screen_card = ctk.CTkFrame(
            col, fg_color=CARD_BG, corner_radius=CORNER_RADIUS,
            border_width=1, border_color=BORDER,
            height=95,
        )
        self._screen_card.pack(fill="x", pady=(PADDING, 0))
        self._screen_card.pack_propagate(False)
        self._screen_accent = ctk.CTkFrame(
            self._screen_card, fg_color=TEXT_MUTED, width=4, corner_radius=0,
        )
        self._screen_accent.pack(side="left", fill="y")
        screen_inner = ctk.CTkFrame(self._screen_card, fg_color="transparent")
        screen_inner.pack(side="left", fill="both", expand=True,
                          padx=PADDING_LG, pady=PADDING // 2)
        _label(screen_inner, "Lung cancer screening").pack(anchor="w")
        self._screen_value = ctk.CTkLabel(
            screen_inner, text="Awaiting analysis",
            font=(FONT_FAMILY, 22, "bold"),
            text_color=TEXT_MUTED, anchor="w",
        )
        self._screen_value.pack(anchor="w", pady=(2, 0))
        self._screen_rationale = ctk.CTkLabel(
            screen_inner,
            text="Run extraction to produce a USPSTF screening recommendation.",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            text_color=TEXT_SECONDARY, anchor="w",
        )
        self._screen_rationale.pack(anchor="w")

    # ------- collapsibles' body builders --------------------------------
    def _build_prompt_body(self, body) -> None:
        inner = ctk.CTkFrame(body, fg_color="transparent")
        inner.pack(fill="both", expand=True, padx=PADDING, pady=PADDING)
        self._prompt_box = ctk.CTkTextbox(
            inner, font=(FONT_FAMILY, FONT_SIZE_SMALL),
            fg_color=INPUT_BG, text_color=TEXT_SECONDARY,
            border_width=1, border_color=BORDER,
            corner_radius=4, wrap="word", height=180,
            undo=True, maxundo=-1,
        )
        self._prompt_box.pack(fill="both", expand=True)
        self._prompt_box.insert("1.0", PATIENT_LEVEL_PROMPT)
        self._prompt_box._textbox.edit_reset()

    def _build_raw_body(self, body) -> None:
        inner = ctk.CTkFrame(body, fg_color="transparent")
        inner.pack(fill="both", expand=True, padx=PADDING, pady=PADDING)
        self._raw_box = ctk.CTkTextbox(
            inner, font=(FONT_FAMILY_MONO, FONT_SIZE_SMALL),
            fg_color=INPUT_BG, text_color=TEXT_SECONDARY,
            border_width=1, border_color=BORDER,
            corner_radius=4, wrap="word", height=200,
        )
        self._raw_box.pack(fill="both", expand=True)
        self._raw_box.configure(state="disabled")

    def _raw_summary(self) -> str:
        if not self._raw_output:
            return "RAW MODEL OUTPUT  ·  RUN EXTRACTION TO POPULATE"
        elapsed = f"{self._last_elapsed:.1f}s" if self._last_elapsed else "—"
        return f"RAW MODEL OUTPUT  ·  GENERATED IN {elapsed}  ·  CLICK TO VIEW"

    # ------- stat tile widget ------------------------------------------
    def _build_tile(self, parent, label_text: str, col: int) -> dict:
        tile = _card(parent)
        tile.grid(row=0, column=col, sticky="nsew",
                  padx=(0 if col == 0 else 6, 6 if col < 3 else 0))
        inner = ctk.CTkFrame(tile, fg_color="transparent")
        inner.pack(fill="both", expand=True, padx=PADDING_LG, pady=PADDING_LG)

        _label(inner, label_text).pack(anchor="w")
        value = ctk.CTkLabel(
            inner, text="—",
            font=(FONT_FAMILY, 22, "bold"),
            text_color=TEXT_MUTED,
            anchor="w",
        )
        value.pack(anchor="w", pady=(8, 0))
        return {"tile": tile, "value": value}

    # ------------------------------------------------------------------
    # Behaviour
    # ------------------------------------------------------------------
    # ------- Notes entry management ----------------------------------
    def _add_note_entry(self, note_value: str = "") -> None:
        entry = _NoteEntry(
            self._notes_scroll,
            note_value=note_value,
            on_remove=self._remove_note_entry,
        )
        # Right-side padding keeps entries off the scrollbar gutter.
        side_pad = (0, 10)
        if self._note_entries:
            ctk.CTkFrame(self._notes_scroll, fg_color=BORDER_SOFT,
                         height=1, corner_radius=0).pack(
                fill="x", padx=side_pad, pady=PADDING // 2)
        entry.pack(fill="x", padx=side_pad, pady=(0, 4))
        self._note_entries.append(entry)
        self._refresh_remove_buttons()

    def _remove_note_entry(self, entry: _NoteEntry) -> None:
        if len(self._note_entries) <= 1:
            return  # always keep at least one
        survivors = [e.body.get("1.0", "end").rstrip("\n")
                     for e in self._note_entries if e is not entry]
        for child in list(self._notes_scroll.winfo_children()):
            child.destroy()
        self._note_entries = []
        for b in survivors:
            self._add_note_entry(b)

    def _refresh_remove_buttons(self) -> None:
        only_one = len(self._note_entries) <= 1
        for e in self._note_entries:
            e.set_removable(not only_one)

    def _combined_notes_text(self) -> tuple[str, int]:
        """Return (joined_for_llm, n_non_empty)."""
        parts = [t for t in (e.formatted() for e in self._note_entries) if t]
        return "\n\n---\n\n".join(parts), len(parts)

    def _all_notes_raw_text(self) -> str:
        """Concatenated raw text for fallback regex (age, year)."""
        return " ".join(e.body.get("1.0", "end") for e in self._note_entries)

    def _reset_prompt(self) -> None:
        self._prompt_box.delete("1.0", "end")
        self._prompt_box.insert("1.0", PATIENT_LEVEL_PROMPT)
        self._prompt_box._textbox.edit_reset()
        self._prompt_collapsible.update_summary()

    def _on_analyze(self) -> None:
        joined, n_notes = self._combined_notes_text()
        prompt = (self._prompt_box.get("1.0", "end").strip()
                  if hasattr(self, "_prompt_box") else PATIENT_LEVEL_PROMPT)
        if n_notes == 0:
            self._set_tile_value("smoking_status", "Paste a note first", TEXT_MUTED)
            return
        self._analyze_btn.configure(state="disabled", text="ANALYZING…")
        for key in ("age", "smoking_status", "pack_years", "years_since_cessation"):
            self._set_tile_value(key, "—", TEXT_MUTED)
        self._set_screening("Awaiting analysis",
                            "Run extraction to produce a USPSTF screening recommendation.",
                            tone="neutral")
        self._start_note_pulse()
        threading.Thread(
            target=self._analyze_in_background,
            args=(joined, prompt, n_notes), daemon=True,
        ).start()

    def _analyze_in_background(self, note: str, prompt: str, n_notes: int) -> None:
        cfg = GeminiConfig()
        t0 = time.perf_counter()
        try:
            raw = run_extraction(
                note, prompt, model=cfg.model_name, host=cfg.ollama_host,
                n_notes=n_notes,
            )
            elapsed = time.perf_counter() - t0
            self.after(0, self._after_extraction, raw, elapsed, None)
        except Exception as e:
            self.after(0, self._after_extraction, "", time.perf_counter() - t0, str(e))

    def _after_extraction(self, raw: str, elapsed: float, error: str | None) -> None:
        self._stop_note_pulse()
        self._analyze_btn.configure(state="normal", text="EXTRACT & PARSE")
        if error:
            self._set_tile_value("smoking_status", "Error", STATE_BAD_TEXT)
            return
        self._raw_output = raw
        self._last_elapsed = elapsed
        if hasattr(self, "_raw_box"):
            self._raw_box.configure(state="normal")
            self._raw_box.delete("1.0", "end")
            self._raw_box.insert("1.0", raw)
            self._raw_box.configure(state="disabled")
        self._raw_collapsible.update_summary()

        note_text = self._all_notes_raw_text()
        years = [int(m) for m in re.findall(r"\b((?:19|20)\d{2})", note_text)]
        note_year = max(years) if years else None
        age = extract_age_from_note(note_text)
        parsed = parse_extraction(raw, note_year=note_year, age=age)
        self._parsed = parsed
        self._apply_parsed(parsed)

    def _apply_parsed(self, parsed: dict) -> None:
        age = parsed.get("age")
        self._set_tile_value(
            "age",
            "—" if age is None else f"{age}",
            TEXT_PRIMARY if age is not None else TEXT_MUTED,
        )
        status = parsed["smoking_status"]
        self._set_tile_value(
            "smoking_status",
            (status or "—").title(),
            TEXT_PRIMARY if status else TEXT_MUTED,
        )
        py = parsed["pack_years"]
        self._set_tile_value(
            "pack_years",
            "—" if py is None else f"{py:.1f}",
            TEXT_PRIMARY if py is not None else TEXT_MUTED,
        )
        ys = parsed["years_since_cessation"]
        self._set_tile_value(
            "years_since_cessation",
            "—" if ys is None else f"{ys}",
            TEXT_PRIMARY if ys is not None else TEXT_MUTED,
        )
        # Screening hero
        elig = parsed.get("screening_eligible", "Unknown")
        rationale = self._eligibility_rationale(parsed, elig)
        tone = ("ok" if elig == "Eligible"
                else "bad" if elig == "Not eligible"
                else "warn")
        self._set_screening(elig, rationale, tone=tone)

    def _set_tile_value(self, key: str, text: str, colour: str) -> None:
        self._tiles[key]["value"].configure(text=text, text_color=colour)

    @staticmethod
    def _eligibility_rationale(parsed: dict, elig: str) -> str:
        age = parsed.get("age")
        py = parsed.get("pack_years")
        status = parsed.get("smoking_status") or "unknown"
        ys = parsed.get("years_since_cessation")
        if elig == "Eligible":
            return (f"Age {age} (50–80), {py:.1f} pack-years (≥20), "
                    f"{status} smoker. Meets USPSTF criteria.")
        if elig == "Not eligible":
            reasons = []
            if status == "never":
                reasons.append("never-smoker")
            if py is not None and py < 20:
                reasons.append(f"only {py:.1f} pack-years (<20 required)")
            if age is not None and not (50 <= age <= 80):
                reasons.append(f"age {age} outside 50–80")
            if status == "former" and ys is not None and ys > 15:
                reasons.append(f"quit {ys}y ago (>15 required)")
            return ("Does not meet USPSTF criteria"
                    + (" — " + ", ".join(reasons) if reasons else "") + ".")
        # Unknown variants
        return elig + ". Capture missing data via the voice agent or chart review."

    # ------- Processing "glow" — pulses the notes-card border ------------
    def _start_note_pulse(self) -> None:
        # Freeze every entry while extraction runs.
        for e in self._note_entries:
            e.set_processing(True)
        self._pulse_phase = False
        if self._pulse_job is None:
            self._pulse_tick()

    def _pulse_tick(self) -> None:
        self._pulse_phase = not self._pulse_phase
        self._note_card.configure(
            border_color=(TEXT_PRIMARY if self._pulse_phase else BORDER),
            border_width=2,
        )
        self._pulse_job = self.after(450, self._pulse_tick)

    def _stop_note_pulse(self) -> None:
        if self._pulse_job is not None:
            try:
                self.after_cancel(self._pulse_job)
            except Exception:
                pass
            self._pulse_job = None
        self._note_card.configure(border_color=BORDER, border_width=1)
        for e in self._note_entries:
            e.set_processing(False)

    def _set_screening(self, value: str, rationale: str, tone: str) -> None:
        tone_map = {
            "ok":      (STATE_OK,    STATE_OK_TEXT),
            "bad":     (STATE_BAD,   STATE_BAD_TEXT),
            "warn":    (STATE_WARN,  STATE_WARN_TEXT),
            "neutral": (TEXT_MUTED,  TEXT_MUTED),
        }
        bar, text_col = tone_map.get(tone, tone_map["neutral"])
        self._screen_accent.configure(fg_color=bar)
        self._screen_value.configure(text=value, text_color=text_col)
        self._screen_rationale.configure(text=rationale)
