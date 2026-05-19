"""Edge TTS backend — free, neural voices from Microsoft.

Supports two playback modes:
  1. speak_blocking()          — sentence-pipelined (for non-streaming use)
  2. speak_sentence_blocking() — play a single sentence (for Gemini streaming)
"""
from __future__ import annotations
from tts.base import TTSBackend
from tts.audio_player import (
    stream_and_play_edge, stop_playback,
    _synth_sentence, _play_decoded,
)


EDGE_VOICES = {
    "Jenny (English) (Female)": "en-US-JennyNeural",
    "Andrew (English) (Male)": "en-US-AndrewNeural",
    "Paloma (Spanish) (Female)": "es-MX-PalomaNeural",
    "Alonso (Spanish) (Male)": "es-MX-AlonsoNeural",
}


class EdgeTTSBackend(TTSBackend):
    """Free neural TTS using Microsoft Edge's speech service."""

    def __init__(self):
        self._voice_id: str = "en-US-JennyNeural"

    @property
    def name(self) -> str:
        return "edge_tts"

    def initialize(self, voice: str = "Jenny (English) (Female)", api_key: str = "") -> None:
        self._voice_id = EDGE_VOICES.get(voice, "en-US-JennyNeural")

    def speak_blocking(self, text: str) -> None:
        """Synthesize and play with sentence-level pipelining."""
        stream_and_play_edge(text, self._voice_id)

    def speak_sentence_blocking(self, sentence: str) -> None:
        """Synthesize + play a single sentence. Blocks until audio finishes.

        Used by the streaming pipeline: Gemini streams a sentence →
        this method immediately synthesizes and plays it.
        """
        audio_bytes = _synth_sentence(sentence, self._voice_id)
        _play_decoded(audio_bytes)

    def get_available_voices(self) -> list[str]:
        return list(EDGE_VOICES.keys())

    def stop(self) -> None:
        stop_playback()
