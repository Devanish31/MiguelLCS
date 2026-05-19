"""Chatterbox TTS backend — local, neural, emotion-controllable (Resemble AI).

Uses the Chatterbox model for high-quality speech with emotion exaggeration control.
Supports zero-shot voice cloning from a short reference clip.

Requires: pip install chatterbox-tts
GPU: ~4-8 GB VRAM (Turbo variant uses less)
"""
from __future__ import annotations
import io
import threading
import numpy as np
import sounddevice as sd
from tts.base import TTSBackend


# Voice presets — maps friendly name → emotion exaggeration level
CHATTERBOX_VOICES = {
    "Default (Balanced)": {"exaggeration": 0.5, "cfg_weight": 0.5},
    "Warm & Empathetic": {"exaggeration": 0.6, "cfg_weight": 0.3},
    "Calm & Steady": {"exaggeration": 0.3, "cfg_weight": 0.5},
    "Expressive": {"exaggeration": 0.8, "cfg_weight": 0.3},
}

# Global stop flag shared with audio_player
_stop_flag = threading.Event()


class ChatterboxBackend(TTSBackend):
    """Local neural TTS using Chatterbox (Resemble AI). MIT license."""

    def __init__(self):
        self._model = None
        self._sample_rate: int = 24000
        self._exaggeration: float = 0.5
        self._cfg_weight: float = 0.5
        self._device: str = "cpu"
        self._voice_name: str = "Default (Balanced)"

    @property
    def name(self) -> str:
        return "chatterbox"

    def initialize(self, voice: str = "Default (Balanced)", api_key: str = "") -> None:
        """Load the Chatterbox model onto GPU/CPU."""
        import torch

        self._voice_name = voice
        preset = CHATTERBOX_VOICES.get(voice, CHATTERBOX_VOICES["Default (Balanced)"])
        self._exaggeration = preset["exaggeration"]
        self._cfg_weight = preset["cfg_weight"]

        if torch.cuda.is_available():
            self._device = "cuda"
        else:
            self._device = "cpu"

        print(f"  [Chatterbox] Loading model on {self._device}...")
        from chatterbox.tts import ChatterboxTTS
        self._model = ChatterboxTTS.from_pretrained(device=self._device)
        self._sample_rate = self._model.sr
        print(f"  [Chatterbox] Ready (sr={self._sample_rate}, "
              f"exag={self._exaggeration}, cfg={self._cfg_weight})")

    def speak_blocking(self, text: str) -> None:
        """Synthesize text and play through speakers. Blocks until done."""
        _stop_flag.clear()
        if not self._model or not text.strip():
            return

        try:
            wav = self._model.generate(
                text,
                exaggeration=self._exaggeration,
                cfg_weight=self._cfg_weight,
            )
            # wav is a torch Tensor [1, samples] — convert to numpy
            audio_np = wav.squeeze(0).cpu().numpy()
            audio_float = audio_np.astype(np.float32)
            # Normalize if needed
            peak = np.abs(audio_float).max()
            if peak > 0:
                audio_float = audio_float / peak

            if _stop_flag.is_set():
                return
            sd.play(audio_float, samplerate=self._sample_rate)
            sd.wait()
        except Exception as e:
            print(f"  [Chatterbox speak error: {e}]")

    def speak_sentence_blocking(self, sentence: str) -> None:
        """Synthesize + play a single sentence for streaming pipeline."""
        self.speak_blocking(sentence)

    def get_available_voices(self) -> list[str]:
        return list(CHATTERBOX_VOICES.keys())

    def stop(self) -> None:
        _stop_flag.set()
        sd.stop()
