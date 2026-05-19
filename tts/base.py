"""Abstract TTS backend interface."""
from abc import ABC, abstractmethod


class TTSBackend(ABC):
    """All TTS backends implement this interface."""

    @abstractmethod
    def initialize(self, voice: str = "Default", api_key: str = "") -> None:
        """Set up the engine. Called once before first use."""
        ...

    @abstractmethod
    def speak_blocking(self, text: str) -> None:
        """Synthesize and play through speakers. Blocks until done."""
        ...

    @abstractmethod
    def get_available_voices(self) -> list[str]:
        """Return list of available voice names."""
        ...

    @abstractmethod
    def stop(self) -> None:
        """Interrupt any currently playing audio."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable engine name."""
        ...
