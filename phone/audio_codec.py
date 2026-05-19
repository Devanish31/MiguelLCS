"""Telephony audio helpers for Twilio Media Streams.

Twilio expects standard G.711 mu-law payload bytes at 8 kHz.  A simple
log-companding formula is not byte-compatible with G.711 and sounds like
noise on the phone line, so this module implements the wire format directly.
"""
from __future__ import annotations
import numpy as np
from scipy.signal import resample_poly

_BIAS = 0x84
_CLIP = 32635


def _linear_sample_to_ulaw(sample: int) -> int:
    sign = 0x80 if sample < 0 else 0
    if sample < 0:
        sample = -sample
    sample = min(sample, _CLIP)
    sample += _BIAS

    exponent = 7
    exp_mask = 0x4000
    while exponent > 0 and not (sample & exp_mask):
        exponent -= 1
        exp_mask >>= 1

    mantissa = (sample >> (exponent + 3)) & 0x0F
    return (~(sign | (exponent << 4) | mantissa)) & 0xFF


def pcm16_to_mulaw(audio: np.ndarray) -> bytes:
    """Encode int16 PCM to standard 8-bit G.711 mu-law bytes."""
    samples = audio.astype(np.int16, copy=False)
    return bytes(_linear_sample_to_ulaw(int(s)) for s in samples)


def _ulaw_sample_to_linear(byte: int) -> int:
    u = (~byte) & 0xFF
    sign = u & 0x80
    exponent = (u >> 4) & 0x07
    mantissa = u & 0x0F
    sample = ((mantissa << 3) + _BIAS) << exponent
    sample -= _BIAS
    return -sample if sign else sample


def mulaw_to_pcm16(data: bytes) -> np.ndarray:
    """Decode standard 8-bit G.711 mu-law bytes to int16 PCM."""
    return np.array([_ulaw_sample_to_linear(b) for b in data], dtype=np.int16)


def resample(audio: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
    if from_sr == to_sr:
        return audio.astype(np.int16, copy=False)
    from math import gcd
    g = gcd(from_sr, to_sr)
    out = resample_poly(audio.astype(np.float32), to_sr // g, from_sr // g)
    return np.clip(out, -32768, 32767).astype(np.int16)
