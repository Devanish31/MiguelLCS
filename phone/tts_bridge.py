"""Generate phone-call audio from the existing Edge TTS stack."""
from __future__ import annotations
import numpy as np
from tts.audio_player import _synth_sentence, _decode_audio, _split_sentences
from phone.audio_codec import resample

EDGE_VOICE_IDS = {
    "Jenny (English) (Female)": "en-US-JennyNeural",
    "Andrew (English) (Male)": "en-US-AndrewNeural",
    "Paloma (Spanish) (Female)": "es-MX-PalomaNeural",
    "Alonso (Spanish) (Male)": "es-MX-AlonsoNeural",
}


def synthesize_phone_pcm(text: str, voice_name: str = "Jenny (English) (Female)") -> np.ndarray:
    """Synthesize response text to mono 8 kHz PCM for Twilio playback."""
    voice_id = EDGE_VOICE_IDS.get(voice_name, "en-US-JennyNeural")
    chunks: list[np.ndarray] = []
    for sentence in _split_sentences(text):
        audio_bytes = _synth_sentence(sentence, voice_id)
        pcm, sr = _decode_audio(audio_bytes)
        chunks.append(resample(pcm, sr, 8000))
    return np.concatenate(chunks) if chunks else np.array([], dtype=np.int16)
