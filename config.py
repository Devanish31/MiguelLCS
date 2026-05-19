"""Central configuration for the Agentic Voice AI prototype."""
from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()


@dataclass
class GeminiConfig:
    """LLM config. Now points at local Gemma 3 via Ollama (name kept for back-compat)."""
    api_key: str = ""  # unused for local Ollama
    model_name: str = "gemma4:e4b"
    temperature: float = 0.3
    max_output_tokens: int = 1024
    ollama_host: str = "http://127.0.0.1:11434"


@dataclass
class WhisperConfig:
    model_id: str = "openai/whisper-small.en"   # small.en: ~3x faster than medium.en, accuracy still strong for short utterances
    device: str = "cuda"                        # RTX 5090


@dataclass
class AudioConfig:
    sample_rate: int = 16000        # Whisper expects 16kHz
    channels: int = 1               # Mono
    silence_threshold: float = 0.01 # RMS threshold for silence detection
    silence_duration: float = 1.0   # Seconds of silence to stop recording
    max_recording_duration: float = 30.0  # Safety cap per utterance
    chunk_duration: float = 0.3     # Read audio in 300ms chunks (was 0.5, faster endpoint detection)


@dataclass
class TTSConfig:
    rate: int = 160       # Words per minute (default ~200, slower is clearer)
    volume: float = 0.9


@dataclass
class AgentConfig:
    max_turns: int = 20
    confidence_threshold: float = 0.7
    enable_shared_decision_making: bool = True


@dataclass
class TwilioConfig:
    """Environment-backed settings for the real-phone transport."""
    account_sid: str = os.getenv("TWILIO_ACCOUNT_SID", "")
    auth_token: str = os.getenv("TWILIO_AUTH_TOKEN", "")
    phone_number: str = os.getenv("TWILIO_PHONE_NUMBER", "")
    public_base_url: str = os.getenv("TWILIO_PUBLIC_BASE_URL", "")

    @property
    def is_configured(self) -> bool:
        return all((
            self.account_sid,
            self.auth_token,
            self.phone_number,
            self.public_base_url,
        ))
