"""Record audio from microphone with silence-based endpoint detection.

Uses a persistent callback-based InputStream that stays open between
recordings. This eliminates Bluetooth HFP mic activation delay (~4s)
on subsequent recordings — the mic stream stays warm.

Resamples to 16kHz for Whisper if needed.
"""
from __future__ import annotations
import queue
import time
import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly
from math import gcd
from config import AudioConfig


def _is_bluetooth_device(device_info: dict) -> bool:
    """Check if a device is a Bluetooth audio device."""
    name = device_info["name"].lower()
    return any(kw in name for kw in ("bluetooth", "headset", "bthhfenum", "hands-free", "x15"))


# Probe duration: RDP-forwarded mics ("Remote Audio") can take ~1s to start
# delivering frames after the InputStream opens, so the probe needs to wait
# long enough not to incorrectly reject them.
_PROBE_DURATION = 1.5


def list_input_devices() -> list[dict]:
    """Return user-pickable input devices for the GUI selector.

    Filters out aggregator/loopback entries (Stereo Mix, Sound Mapper,
    Primary Sound Driver) that are never what the user wants.
    Each entry: {id, name, default_sr, is_default, is_remote, is_bluetooth}.
    """
    devices = sd.query_devices()
    default_input = sd.default.device[0]
    result = []
    for i, d in enumerate(devices):
        if d["max_input_channels"] <= 0:
            continue
        name_lower = d["name"].lower()
        if "stereo mix" in name_lower or "sound mapper" in name_lower or "primary sound" in name_lower:
            continue
        result.append({
            "id": i,
            "name": d["name"],
            "default_sr": int(d["default_samplerate"]),
            "is_default": i == default_input,
            "is_remote": "remote audio" in name_lower,
            "is_bluetooth": _is_bluetooth_device(d),
        })
    return result


def _probe_device(dev_id: int, probe_duration: float = _PROBE_DURATION) -> int | None:
    """Try a device at several sample rates. Returns the working SR or None."""
    test_rates = [16000, 48000, 44100, 32000, 22050]
    for sr in test_rates:
        try:
            q: queue.Queue = queue.Queue()
            def cb(indata, frames, time_info, status, _q=q):
                _q.put(True)
            with sd.InputStream(samplerate=sr, channels=1, dtype="int16",
                                blocksize=int(sr * 0.1), device=dev_id,
                                callback=cb):
                time.sleep(probe_duration)
            if not q.empty():
                return sr
        except Exception:
            continue
    return None


def _find_working_input_device() -> tuple[int | None, int, bool]:
    """Find a working input device and its supported sample rate.

    Priority:
      1. Windows default input device
      2. Bluetooth / headset devices
      3. Named microphones
      4. Any other input device

    Returns (device_id, sample_rate, is_bluetooth) or (None, 16000, False).
    """
    devices = sd.query_devices()
    default_input = sd.default.device[0]

    tier_default = []
    tier_bluetooth = []
    tier_microphone = []
    tier_other = []

    for i, d in enumerate(devices):
        if d["max_input_channels"] <= 0:
            continue
        name = d["name"].lower()

        if "stereo mix" in name or "sound mapper" in name or "primary sound" in name:
            continue
        if name.strip() == "microphone ()" or name.strip() == "microphone":
            tier_other.append(i)
            continue

        if i == default_input:
            tier_default.append(i)
        elif _is_bluetooth_device(d):
            tier_bluetooth.append(i)
        elif "microphone" in name or "mic" in name:
            tier_microphone.append(i)
        else:
            tier_other.append(i)

    candidates = tier_default + tier_bluetooth + tier_microphone + tier_other

    if not candidates and default_input is not None:
        candidates = [default_input]

    for dev_id in candidates:
        sr = _probe_device(dev_id)
        if sr is not None:
            is_bt = _is_bluetooth_device(sd.query_devices(dev_id))
            return dev_id, sr, is_bt

    return None, 16000, False


def force_input_device(dev_id: int) -> tuple[int, int, bool]:
    """User-pinned device: probe it for a working SR, no fallback.

    Returns (device_id, sample_rate, is_bluetooth). Falls back to the device's
    default_samplerate if no test rate works (the stream may still open even
    if no frames arrived during the probe — common with RDP at session start).
    """
    info = sd.query_devices(dev_id)
    sr = _probe_device(dev_id)
    if sr is None:
        sr = int(info["default_samplerate"])
    return dev_id, sr, _is_bluetooth_device(info)


class AudioRecorder:
    """Records audio from the microphone, stopping after silence is detected.

    Uses a PERSISTENT audio stream that stays open between recordings.
    This eliminates the Bluetooth HFP mic activation delay (~4s) on all
    recordings after the first warmup.

    The stream continuously captures audio into a drain queue. When
    record_until_silence() is called, it switches to capture mode and
    collects frames until silence is detected.
    """

    # First-time Bluetooth warm-up (only once)
    BT_FIRST_WARMUP = 3.0

    def __init__(self, config: AudioConfig | None = None,
                 device_override: int | None = None):
        self.config = config or AudioConfig()
        self.target_sample_rate = 16000

        if device_override is not None:
            self.input_device, self.native_sample_rate, self.is_bluetooth = (
                force_input_device(device_override)
            )
        else:
            self.input_device, self.native_sample_rate, self.is_bluetooth = (
                _find_working_input_device()
            )

        if self.input_device is not None:
            dev_info = sd.query_devices(self.input_device)
            bt_tag = " [Bluetooth]" if self.is_bluetooth else ""
            print(f"  [Audio input: {dev_info['name']} (id={self.input_device}) "
                  f"@ {self.native_sample_rate}Hz{bt_tag}]")
            if self.native_sample_rate != self.target_sample_rate:
                print(f"  [Will resample {self.native_sample_rate}Hz -> "
                      f"{self.target_sample_rate}Hz for Whisper]")
        else:
            print("  [Warning: No working input device found!]")
            self.native_sample_rate = self.target_sample_rate

        self.chunk_samples = int(self.config.chunk_duration * self.native_sample_rate)

        # Persistent stream state
        self._audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self._stream: sd.InputStream | None = None
        self._warmed_up = False

    def _ensure_stream(self) -> None:
        """Start the persistent audio stream if not already running."""
        if self._stream is not None and self._stream.active:
            return

        def _callback(indata, frames, time_info, status):
            self._audio_queue.put(indata[:, 0].copy())

        self._stream = sd.InputStream(
            samplerate=self.native_sample_rate,
            channels=1,
            dtype="int16",
            blocksize=self.chunk_samples,
            device=self.input_device,
            callback=_callback,
        )
        self._stream.start()

        # First-time Bluetooth warm-up
        if self.is_bluetooth and not self._warmed_up:
            print(f"  [Bluetooth first-time warm-up: {self.BT_FIRST_WARMUP}s...]")
            time.sleep(self.BT_FIRST_WARMUP)
            # Drain warm-up audio
            while not self._audio_queue.empty():
                self._audio_queue.get_nowait()
            self._warmed_up = True
            print("  [Bluetooth mic active — stream staying open]")

    def _drain_queue(self) -> None:
        """Drain any buffered audio from between recordings."""
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break

    def record_until_silence(self) -> np.ndarray:
        """Record audio, stopping after silence_duration seconds of silence.

        Returns numpy array of int16 audio samples (mono, 16kHz).
        """
        self._ensure_stream()
        self._drain_queue()

        frames: list[np.ndarray] = []
        silence_chunks = 0
        max_silence_chunks = int(
            self.config.silence_duration / self.config.chunk_duration
        )
        max_total_chunks = int(
            self.config.max_recording_duration / self.config.chunk_duration
        )
        speech_started = False

        print("  [Listening... speak now]")

        for _ in range(max_total_chunks):
            try:
                chunk = self._audio_queue.get(timeout=self.config.chunk_duration + 1.0)
            except queue.Empty:
                continue

            frames.append(chunk)

            rms = np.sqrt(np.mean(chunk.astype(np.float32) ** 2)) / 32768.0

            if rms > self.config.silence_threshold:
                speech_started = True
                silence_chunks = 0
            else:
                if speech_started:
                    silence_chunks += 1

            if speech_started and silence_chunks >= max_silence_chunks:
                break

        if not frames or not speech_started:
            print("  [No speech detected]")
            return np.array([], dtype=np.int16)

        audio = np.concatenate(frames)
        duration = len(audio) / self.native_sample_rate
        print(f"  [Recorded {duration:.1f}s of audio]")

        if self.native_sample_rate != self.target_sample_rate:
            audio = self._resample(audio, self.native_sample_rate, self.target_sample_rate)

        return audio

    def close(self) -> None:
        """Stop and close the persistent stream."""
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    def _resample(self, audio: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
        """Resample audio using polyphase filtering (high quality)."""
        divisor = gcd(from_sr, to_sr)
        up = to_sr // divisor
        down = from_sr // divisor
        audio_float = audio.astype(np.float32)
        resampled = resample_poly(audio_float, up, down)
        return resampled.astype(np.int16)
