"""ElevenLabs TTS backend — premium voices, requires API key."""
from __future__ import annotations
from tts.base import TTSBackend
from tts.audio_player import play_audio_bytes, stop_playback


ELEVENLABS_VOICES = ["Rachel", "Drew", "Clyde", "Domi", "Bella", "Antoni", "Elli"]


class ElevenLabsBackend(TTSBackend):
    """ElevenLabs TTS with high-quality emotional voices."""

    def __init__(self):
        self._client = None
        self._voice: str = "Rachel"

    @property
    def name(self) -> str:
        return "elevenlabs"

    def initialize(self, voice: str = "Rachel", api_key: str = "") -> None:
        if not api_key:
            raise ValueError("ElevenLabs requires an API key.")
        from elevenlabs.client import ElevenLabs
        self._client = ElevenLabs(api_key=api_key)
        self._voice = voice

    def speak_blocking(self, text: str) -> None:
        if self._client is None:
            raise RuntimeError("ElevenLabs not initialized. Provide API key.")
        audio_gen = self._client.generate(
            text=text,
            voice=self._voice,
            model="eleven_monolingual_v1",
        )
        audio_bytes = b"".join(audio_gen)
        play_audio_bytes(audio_bytes)

    def get_available_voices(self) -> list[str]:
        return list(ELEVENLABS_VOICES)

    def stop(self) -> None:
        stop_playback()
