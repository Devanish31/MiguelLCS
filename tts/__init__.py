"""TTS engine factory."""
from __future__ import annotations
from tts.base import TTSBackend


def create_tts_engine(
    engine_name: str,
    voice: str = "Default",
    api_key: str = "",
) -> TTSBackend:
    """Create and initialize a TTS backend by name.

    Falls back to pyttsx3 if the requested engine can't be imported.
    """
    try:
        if engine_name == "pyttsx3":
            from tts.pyttsx3_backend import Pyttsx3Backend
            engine = Pyttsx3Backend()
        elif engine_name == "edge_tts":
            from tts.edge_tts_backend import EdgeTTSBackend
            engine = EdgeTTSBackend()
        elif engine_name == "elevenlabs":
            from tts.elevenlabs_backend import ElevenLabsBackend
            engine = ElevenLabsBackend()
        elif engine_name == "chatterbox":
            from tts.chatterbox_backend import ChatterboxBackend
            engine = ChatterboxBackend()
        elif engine_name == "qwen3_tts":
            from tts.qwen3_tts_backend import Qwen3TTSBackend
            engine = Qwen3TTSBackend()
        else:
            raise ValueError(f"Unknown TTS engine: {engine_name}")

        engine.initialize(voice=voice, api_key=api_key)
        return engine

    except (ImportError, ValueError) as e:
        print(f"  [TTS Warning] {engine_name} unavailable ({e}). "
              f"Falling back to pyttsx3.")
        from tts.pyttsx3_backend import Pyttsx3Backend
        fallback = Pyttsx3Backend()
        fallback.initialize()
        return fallback
