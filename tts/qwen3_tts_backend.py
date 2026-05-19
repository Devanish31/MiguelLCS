"""Qwen3-TTS backend — local, neural, natural-language emotion control (Alibaba).

Uses natural language instructions to control emotion, tone, and speaking style.
Supports 9 built-in speakers and voice cloning from 3s of reference audio.

Requires: pip install qwen-tts
          pip install flash-attn --no-build-isolation  (optional but recommended)
GPU: ~4-8 GB VRAM (1.7B model with bfloat16)
License: Apache 2.0
"""
from __future__ import annotations
import threading
import numpy as np
import sounddevice as sd
from tts.base import TTSBackend


# Built-in speakers from the CustomVoice model
# Maps friendly display name → (speaker_id, language, default_instruct)
QWEN3_VOICES = {
    "Ryan (Male, English)": {"speaker": "Ryan", "language": "English", "instruct": "Speak in a warm, friendly tone."},
    "Aiden (Male, English)": {"speaker": "Aiden", "language": "English", "instruct": "Speak clearly and naturally."},
    "Ryan - Empathetic": {"speaker": "Ryan", "language": "English", "instruct": "Speak in a warm, empathetic, caring tone with gentle pauses."},
    "Aiden - Calm": {"speaker": "Aiden", "language": "English", "instruct": "Speak in a calm, steady, reassuring tone."},
    "Vivian (Female, Chinese)": {"speaker": "Vivian", "language": "Chinese", "instruct": ""},
    "Serena (Female, Chinese)": {"speaker": "Serena", "language": "Chinese", "instruct": ""},
}

# Global stop flag
_stop_flag = threading.Event()


class Qwen3TTSBackend(TTSBackend):
    """Local neural TTS using Qwen3-TTS (Alibaba). Apache 2.0 license."""

    def __init__(self):
        self._model = None
        self._sample_rate: int = 24000
        self._speaker: str = "Ryan"
        self._language: str = "English"
        self._instruct: str = "Speak in a warm, friendly tone."
        self._voice_name: str = "Ryan (Male, English)"
        self._device: str = "cpu"

    @property
    def name(self) -> str:
        return "qwen3_tts"

    def initialize(self, voice: str = "Ryan (Male, English)", api_key: str = "") -> None:
        """Load the Qwen3-TTS CustomVoice model."""
        import torch

        self._voice_name = voice
        preset = QWEN3_VOICES.get(voice, QWEN3_VOICES["Ryan (Male, English)"])
        self._speaker = preset["speaker"]
        self._language = preset["language"]
        self._instruct = preset["instruct"]

        if torch.cuda.is_available():
            self._device = "cuda:0"
        else:
            self._device = "cpu"

        print(f"  [Qwen3-TTS] Loading model on {self._device}...")

        # Try with flash_attention_2 first, fall back to default
        from qwen_tts import Qwen3TTSModel
        try:
            self._model = Qwen3TTSModel.from_pretrained(
                "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                device_map=self._device,
                dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
            )
            print("  [Qwen3-TTS] Loaded with FlashAttention 2")
        except Exception:
            self._model = Qwen3TTSModel.from_pretrained(
                "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                device_map=self._device,
                dtype=torch.bfloat16,
            )
            print("  [Qwen3-TTS] Loaded without FlashAttention")

        print(f"  [Qwen3-TTS] Ready (speaker={self._speaker}, lang={self._language})")

    def speak_blocking(self, text: str) -> None:
        """Synthesize text and play through speakers. Blocks until done."""
        _stop_flag.clear()
        if not self._model or not text.strip():
            return

        try:
            wavs, sr = self._model.generate_custom_voice(
                text=text,
                language=self._language,
                speaker=self._speaker,
                instruct=self._instruct,
            )
            self._sample_rate = sr
            audio_np = wavs[0]  # numpy array
            if isinstance(audio_np, np.ndarray):
                audio_float = audio_np.astype(np.float32)
            else:
                # Might be a torch tensor
                audio_float = audio_np.cpu().numpy().astype(np.float32)

            # Normalize
            peak = np.abs(audio_float).max()
            if peak > 0:
                audio_float = audio_float / peak

            if _stop_flag.is_set():
                return
            sd.play(audio_float, samplerate=sr)
            sd.wait()
        except Exception as e:
            print(f"  [Qwen3-TTS speak error: {e}]")

    def speak_sentence_blocking(self, sentence: str) -> None:
        """Synthesize + play a single sentence for streaming pipeline."""
        self.speak_blocking(sentence)

    def get_available_voices(self) -> list[str]:
        return list(QWEN3_VOICES.keys())

    def stop(self) -> None:
        _stop_flag.set()
        sd.stop()
