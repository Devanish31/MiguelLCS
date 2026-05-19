"""Speech-to-text using Whisper via HuggingFace Transformers (GPU-accelerated)."""
from __future__ import annotations
import warnings
import numpy as np
import torch
from transformers import pipeline as hf_pipeline
from config import WhisperConfig


class SpeechToText:
    """Transcribe audio using Whisper on GPU (RTX 5090)."""

    def __init__(self, config: WhisperConfig | None = None):
        self.config = config or WhisperConfig()
        self._pipe = None  # Lazy load

    def load_model(self) -> None:
        """Eagerly load the Whisper model (call at startup to avoid first-call delay)."""
        if self._pipe is None:
            device = self.config.device
            use_gpu = device == "cuda" and torch.cuda.is_available()

            # chunk_length_s=30 forces the chunked code path. Without it, newer
            # transformers releases raise KeyError: 'num_frames' on unchunked
            # input. Whisper's native receptive field is 30s, so this is a
            # no-op for typical utterances (<30s).
            if use_gpu:
                gpu_name = torch.cuda.get_device_name(0)
                print(f"  [Loading Whisper model: {self.config.model_id} on GPU ({gpu_name})...]")
                self._pipe = hf_pipeline(
                    "automatic-speech-recognition",
                    model=self.config.model_id,
                    dtype=torch.float16,   # fp16 for GPU speed
                    device="cuda",
                    chunk_length_s=30,
                )
            else:
                print(f"  [Loading Whisper model: {self.config.model_id} on CPU...]")
                self._pipe = hf_pipeline(
                    "automatic-speech-recognition",
                    model=self.config.model_id,
                    dtype=torch.float32,   # CPU needs float32
                    device="cpu",
                    chunk_length_s=30,
                )
            print("  [Whisper model loaded]")

    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe a numpy int16 audio array to text.

        Args:
            audio: numpy array of int16 audio samples
            sample_rate: sample rate (Whisper expects 16000)

        Returns:
            Transcribed text string
        """
        self.load_model()

        if len(audio) == 0:
            return ""

        # Convert int16 -> float32 normalized to [-1, 1]
        audio_float = audio.astype(np.float32) / 32768.0

        # English-only models (.en) don't accept language/task kwargs
        is_english_only = ".en" in self.config.model_id
        gen_kwargs = {} if is_english_only else {"language": "en", "task": "transcribe"}

        # Suppress noisy deprecation warnings from transformers
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*forced_decoder_ids.*")
            warnings.filterwarnings("ignore", message=".*custom logits processor.*")
            warnings.filterwarnings("ignore", category=FutureWarning)
            warnings.filterwarnings("ignore", category=UserWarning)

            result = self._pipe(
                {"raw": audio_float, "sampling_rate": sample_rate},
                generate_kwargs=gen_kwargs,
            )

        text = result["text"].strip()
        print(f"  [Transcribed] \"{text}\"")
        return text
