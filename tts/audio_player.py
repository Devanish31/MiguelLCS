"""Unified audio playback for TTS backends.

Supports two modes:
  1. play_audio_bytes()     — decode-then-play (simple, for pyttsx3/elevenlabs)
  2. stream_and_play_edge() — starts playing while Edge TTS is still synthesizing

Decodes MP3 via miniaudio (no ffmpeg needed), WAV via stdlib.
Plays through sounddevice (routes to default output — Bluetooth, speakers, etc.).
"""
from __future__ import annotations
import io
import threading
import numpy as np
import sounddevice as sd


# ── Global stop flag ──────────────────────────────────────────────
_stop_flag = threading.Event()


def stop_playback() -> None:
    """Stop any currently playing audio."""
    _stop_flag.set()
    sd.stop()


# ── Simple (non-streaming) playback ──────────────────────────────

def play_audio_bytes(audio_bytes: bytes) -> None:
    """Play audio bytes (MP3 or WAV) through speakers. Blocks until done."""
    _stop_flag.clear()
    audio, sr = _decode_audio(audio_bytes)
    audio_float = audio.astype(np.float32) / 32768.0
    sd.play(audio_float, samplerate=sr)
    sd.wait()


# ── Pipelined Edge TTS playback ──────────────────────────────────

def stream_and_play_edge(text: str, voice_id: str) -> None:
    """Synthesize with Edge TTS using sentence-level pipelining.

    Splits text into sentences and overlaps synthesis with playback:
      1. Synthesize sentence 1
      2. While playing sentence 1, synthesize sentence 2 in background
      3. While playing sentence 2, synthesize sentence 3 in background
      ... and so on.

    This means the user hears the first sentence after ~0.5-1s of synthesis
    instead of waiting for the entire response to be synthesized (~2-4s).

    Blocks until all audio has finished playing.
    """
    import time as _time
    _stop_flag.clear()

    # Split into sentences for pipelined playback
    sentences = _split_sentences(text)
    if not sentences:
        return

    t0 = _time.perf_counter()
    print(f"  [TTS pipeline] {len(sentences)} sentence(s) to synthesize")

    # Synthesize first sentence (must wait for this one)
    if _stop_flag.is_set():
        return
    current_audio = _synth_sentence(sentences[0], voice_id)
    t_first = _time.perf_counter() - t0
    print(f"  [TTS pipeline] First sentence synth: {t_first:.2f}s (time-to-first-audio)")

    for i in range(len(sentences)):
        if _stop_flag.is_set():
            break

        # Start synthesizing NEXT sentence in background thread
        next_audio_result = [None]
        synth_thread = None
        if i + 1 < len(sentences):
            def _bg_synth(idx=i + 1):
                next_audio_result[0] = _synth_sentence(sentences[idx], voice_id)
            synth_thread = threading.Thread(target=_bg_synth, daemon=True)
            synth_thread.start()

        # Play current sentence (blocks until audio finishes)
        _play_decoded(current_audio)

        # Wait for background synthesis to finish (usually already done by now)
        if synth_thread is not None:
            synth_thread.join(timeout=30)
            current_audio = next_audio_result[0] or b""

    total = _time.perf_counter() - t0
    print(f"  [TTS pipeline] Total speak time: {total:.2f}s")


def _synth_sentence(sentence: str, voice_id: str) -> bytes:
    """Synthesize a single sentence to MP3 bytes via Edge TTS."""
    import asyncio
    import edge_tts

    async def _do():
        communicate = edge_tts.Communicate(sentence, voice_id)
        buf = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                buf.write(chunk["data"])
        return buf.getvalue()

    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(_do())
    finally:
        loop.close()


def _play_decoded(audio_bytes: bytes) -> None:
    """Decode MP3/WAV bytes and play through speakers. Blocks until done."""
    if _stop_flag.is_set() or not audio_bytes or len(audio_bytes) == 0:
        return
    try:
        audio, sr = _decode_audio(audio_bytes)
        audio_float = audio.astype(np.float32) / 32768.0
        sd.play(audio_float, samplerate=sr)
        sd.wait()
    except Exception as e:
        print(f"  [TTS play error: {e}]")


def _split_sentences(text: str) -> list[str]:
    """Split text into sentence chunks for pipelined TTS.

    Merges very short fragments to avoid too many synthesis calls.
    """
    import re
    # Split on sentence-ending punctuation followed by space
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    parts = [p.strip() for p in parts if p.strip()]

    if not parts:
        return [text] if text.strip() else []

    # Merge very short segments (< 40 chars) with the next one
    merged = []
    buf = ""
    for p in parts:
        if buf:
            buf += " " + p
            if len(buf) >= 40:
                merged.append(buf)
                buf = ""
        elif len(p) < 40 and p != parts[-1]:
            buf = p
        else:
            merged.append(p)
    if buf:
        if merged:
            merged[-1] += " " + buf
        else:
            merged.append(buf)

    return merged


# ── Decode helpers ───────────────────────────────────────────────

def _decode_audio(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    """Decode audio bytes to numpy array (int16 mono) + sample rate.

    Priority: miniaudio (MP3/WAV/FLAC/OGG) → stdlib wave → pydub fallback.
    """
    # --- Method 1: miniaudio (pure C, no ffmpeg) ---
    try:
        import miniaudio
        decoded = miniaudio.decode(audio_bytes, output_format=miniaudio.SampleFormat.SIGNED16)
        sr = decoded.sample_rate
        n_channels = decoded.nchannels
        audio = np.frombuffer(decoded.samples, dtype=np.int16)
        if n_channels > 1:
            audio = audio[::n_channels]
        return audio, sr
    except Exception:
        pass

    # --- Method 2: stdlib wave (WAV only) ---
    try:
        import wave
        buf = io.BytesIO(audio_bytes)
        with wave.open(buf) as wf:
            frames = wf.readframes(wf.getnframes())
            sr = wf.getframerate()
            n_channels = wf.getnchannels()
        audio = np.frombuffer(frames, dtype=np.int16)
        if n_channels > 1:
            audio = audio[::n_channels]
        return audio, sr
    except Exception:
        pass

    # --- Method 3: pydub (needs ffmpeg) ---
    try:
        from pydub import AudioSegment
        seg = AudioSegment.from_file(io.BytesIO(audio_bytes))
        seg = seg.set_channels(1)
        audio = np.array(seg.get_array_of_samples(), dtype=np.int16)
        return audio, seg.frame_rate
    except Exception as e:
        raise RuntimeError(f"Could not decode audio: {e}")
