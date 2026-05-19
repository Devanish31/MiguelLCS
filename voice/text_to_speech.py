"""Text-to-speech output using pyttsx3 (offline, Windows SAPI5)."""
from __future__ import annotations
import time
from config import TTSConfig


class TextToSpeech:
    """Convert text to speech and play through speakers."""

    def __init__(self, config: TTSConfig | None = None):
        self.config = config or TTSConfig()
        self._engine = None

    def _init_engine(self) -> None:
        """Initialize the TTS engine on first use."""
        if self._engine is None:
            import pyttsx3
            self._engine = pyttsx3.init()
            self._engine.setProperty("rate", self.config.rate)
            self._engine.setProperty("volume", self.config.volume)

            # Try to pick a pleasant voice
            voices = self._engine.getProperty("voices")
            for voice in voices:
                name = voice.name.lower()
                if "zira" in name or "female" in name:
                    self._engine.setProperty("voice", voice.id)
                    break

            print("  [TTS engine initialized]")

    def speak(self, text: str) -> None:
        """Speak the given text aloud through speakers."""
        self._init_engine()
        print(f"  [Speaking] \"{text}\"")
        self._engine.say(text)
        self._engine.runAndWait()

    def speak_with_pause(self, text: str, pause_ms: int = 500) -> None:
        """Speak text with a brief pause afterward for conversational rhythm."""
        self.speak(text)
        time.sleep(pause_ms / 1000.0)
