"""pyttsx3 TTS backend — offline, Windows SAPI5."""
from __future__ import annotations
from tts.base import TTSBackend


class Pyttsx3Backend(TTSBackend):
    """Wraps pyttsx3 for offline TTS. Robotic but always available."""

    def __init__(self):
        self._engine = None

    @property
    def name(self) -> str:
        return "pyttsx3"

    def initialize(self, voice: str = "Default", api_key: str = "") -> None:
        import pyttsx3
        self._engine = pyttsx3.init()
        self._engine.setProperty("rate", 160)
        self._engine.setProperty("volume", 0.9)
        if voice and voice != "Default":
            voices = self._engine.getProperty("voices")
            voice_lower = voice.split("(")[0].strip().lower()
            for v in voices:
                if voice_lower in v.name.lower():
                    self._engine.setProperty("voice", v.id)
                    break

    def speak_blocking(self, text: str) -> None:
        if self._engine is None:
            self.initialize()
        self._engine.say(text)
        self._engine.runAndWait()

    def get_available_voices(self) -> list[str]:
        if self._engine is None:
            self.initialize()
        voices = self._engine.getProperty("voices")
        results = []
        for v in voices:
            tag = "(Female)" if "zira" in v.name.lower() else "(Male)"
            results.append(f"{v.name} {tag}")
        return results

    def stop(self) -> None:
        if self._engine:
            self._engine.stop()
