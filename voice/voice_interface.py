"""Orchestrates the voice interaction loop: speak -> listen -> transcribe.

Supports two input modes:
  - Voice: mic recording + Whisper STT (default, for local use)
  - Keyboard: typed input fallback (for RDP / no-mic scenarios)

The mode is auto-detected at startup based on whether the mic captures
any signal, or can be forced via constructor parameter.
"""
from __future__ import annotations
import threading
import queue
import numpy as np
import sounddevice as sd
from voice.audio_recorder import AudioRecorder
from voice.speech_to_text import SpeechToText
from voice.text_to_speech import TextToSpeech
from config import AudioConfig, WhisperConfig, TTSConfig


def _test_mic_has_signal(recorder: AudioRecorder) -> bool:
    """Test if the mic is picking up any signal.

    For Bluetooth devices, waits longer (5s) to allow HFP activation.
    For wired mics, a quick 1.5s test suffices.
    """
    import time as _time

    test_duration = 5.0 if recorder.is_bluetooth else 1.5
    audio_q: queue.Queue[np.ndarray] = queue.Queue()

    def cb(indata, frames, time_info, status):
        audio_q.put(indata[:, 0].copy())

    try:
        with sd.InputStream(
            samplerate=recorder.native_sample_rate,
            channels=1,
            dtype="int16",
            blocksize=int(recorder.native_sample_rate * 0.5),
            device=recorder.input_device,
            callback=cb,
        ):
            _time.sleep(test_duration)
    except Exception:
        return False

    chunks = []
    while not audio_q.empty():
        chunks.append(audio_q.get())

    if not chunks:
        return False

    audio = np.concatenate(chunks)
    rms = np.sqrt(np.mean(audio.astype(np.float32) ** 2)) / 32768.0
    # If RMS is exactly 0 or extremely low, mic has no signal (RDP / disconnected)
    return rms > 0.0001


class VoiceInterface:
    """High-level interface for voice-based conversation.

    Auto-detects whether to use mic or keyboard input:
      - If mic has signal → voice mode (mic + Whisper)
      - If mic is silent  → keyboard fallback (type responses)

    TTS output works in both modes.
    """

    def __init__(
        self,
        audio_config: AudioConfig | None = None,
        whisper_config: WhisperConfig | None = None,
        tts_config: TTSConfig | None = None,
        force_mode: str | None = None,  # "voice", "keyboard", or None (auto)
    ):
        self.audio_config = audio_config or AudioConfig()
        self.recorder = AudioRecorder(self.audio_config)
        self.stt = SpeechToText(whisper_config)
        self.tts = TextToSpeech(tts_config)
        self.force_mode = force_mode
        self.use_voice: bool = True  # Set during initialize()

    def initialize(self) -> None:
        """Pre-load models and auto-detect input mode."""
        print("\n[Initializing voice interface...]")

        if self.force_mode == "keyboard":
            self.use_voice = False
            print("  [Mode: KEYBOARD (forced)]")
        elif self.force_mode == "voice":
            self.use_voice = True
            self.stt.load_model()
            print("  [Mode: VOICE (forced)]")
        else:
            # Auto-detect: test if mic has any signal
            print("  [Testing microphone signal...]")
            has_signal = _test_mic_has_signal(self.recorder)
            if has_signal:
                self.use_voice = True
                self.stt.load_model()
                print("  [Mode: VOICE (mic signal detected)]")
            else:
                self.use_voice = False
                print("  [Mode: KEYBOARD (no mic signal — likely RDP)]")
                print("  [You will type your responses instead of speaking]")

        print("[Voice interface ready]\n")

    def say(self, text: str) -> None:
        """Speak text to the patient (works in both modes)."""
        self.tts.speak_with_pause(text)

    def listen(self) -> str:
        """Get patient input — via mic or keyboard depending on mode."""
        if self.use_voice:
            return self._listen_voice()
        else:
            return self._listen_keyboard()

    def _listen_voice(self) -> str:
        """Listen via microphone and transcribe with Whisper."""
        audio = self.recorder.record_until_silence()
        if len(audio) == 0:
            return ""
        text = self.stt.transcribe(audio, sample_rate=16000)
        return text

    def _listen_keyboard(self) -> str:
        """Get input from keyboard (RDP/no-mic fallback)."""
        try:
            text = input("  You (type): ").strip()
            return text
        except (EOFError, KeyboardInterrupt):
            return ""

    def say_and_listen(self, text: str) -> str:
        """Speak, then listen for a response."""
        self.say(text)
        return self.listen()
