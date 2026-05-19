"""Gemma Intelligence — Offline Medical Protocol theme.

Warm dark earth-tone palette. Two surfaces: page background + card.
Brand red is intentionally de-emphasised; the visual identity is monochrome
warm neutrals with state-only accents.
"""
from __future__ import annotations


# ── Core palette ─────────────────────────────────────────────────────
PAGE_BG = "#1A1712"          # page background — deep warm black
INPUT_BG = "#11100D"         # input fields / terminals — near-black
CARD_BG = "#524841"          # card surface — warm taupe
CARD_BG_ALT = "#3A332D"      # subtle alt for divided rows
CARD_HOVER = "#5E5448"

BORDER = "#8E887A"           # warm sand border
BORDER_SOFT = "#6E6859"      # softer divider

# Text
TEXT_PRIMARY = "#D8D3CD"     # off-white warm
TEXT_SECONDARY = "#B5B1AD"   # muted warm gray
TEXT_MUTED = "#8E887A"       # very muted (== border colour)
TEXT_INVERSE = "#1A1712"     # for use on bright CTAs

# Action / state
BTN_NEUTRAL_BG = "#746E61"
BTN_NEUTRAL_HOVER = "#8E887A"
BTN_BRIGHT_BG = "#D8D3CD"    # the "Commit to EHR" hero CTA
BTN_BRIGHT_HOVER = "#FFFFFF"

# State accents (only used for eligibility pill & status dots)
STATE_OK = "#2D5A3D"         # eligible / secure-ready
STATE_OK_TEXT = "#7FB28E"
STATE_WARN = "#7A6A2F"       # unknown / amber
STATE_WARN_TEXT = "#C9B477"
STATE_BAD = "#7A2F2F"        # not eligible
STATE_BAD_TEXT = "#D29B9B"

# ── Backwards-compat aliases (older widgets still import these) ──────
DARK_BG = PAGE_BG
DARK_BG_SECONDARY = INPUT_BG
DARK_CARD = CARD_BG
DARK_SURFACE = INPUT_BG
DARK_CARD_HOVER = CARD_HOVER
CARDINAL_RED = BTN_NEUTRAL_BG      # brand red retired; use neutral CTA
CARDINAL_RED_DARK = "#5C5749"
ACCENT_BLUE = BTN_NEUTRAL_BG
ACCENT_GREEN = STATE_OK
ACCENT_AMBER = STATE_WARN
ACCENT_RED = STATE_BAD
USER_BUBBLE = CARD_BG
ASSISTANT_BUBBLE = CARD_BG_ALT
SYSTEM_BUBBLE = "#2A2520"

# ── Typography ───────────────────────────────────────────────────────
# Inter if installed, otherwise the OS falls back gracefully.
FONT_FAMILY = "Inter"
FONT_FAMILY_MONO = "Consolas"

FONT_SIZE_HERO = 56          # "Eligible." display
FONT_SIZE_TITLE = 16
FONT_SIZE_HEADING = 13
FONT_SIZE_BODY = 13
FONT_SIZE_SMALL = 11
FONT_SIZE_TINY = 9
FONT_SIZE_LABEL = 10         # ALLCAPS micro-labels

# ── Sizing ───────────────────────────────────────────────────────────
SIDEBAR_WIDTH = 300
STATUS_BAR_HEIGHT = 36
INPUT_HEIGHT = 50
PADDING = 12
PADDING_LG = 24
CORNER_RADIUS = 6
