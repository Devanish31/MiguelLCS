"""Conversation session for a single Twilio Media Streams call."""
from __future__ import annotations
import asyncio
import base64
import json
import numpy as np
from fastapi import WebSocket
from twilio.rest import Client

from agent.agent_core import AgentCore
from config import AgentConfig, GeminiConfig, WhisperConfig, TwilioConfig
from output.summary_generator import SummaryGenerator
from output.transcript_logger import TranscriptLogger
from patient_data.mock_patients import get_patient_by_id
from phone.audio_codec import mulaw_to_pcm16, pcm16_to_mulaw, resample
from phone.tts_bridge import synthesize_phone_pcm
from voice.speech_to_text import SpeechToText
from phone.debug_log import log
from phone.event_log import emit


class TwilioPhoneSession:
    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
        self.patient = None
        self.voice_name = "Jenny (Female, US)"
        self.agent = None
        self.stt = SpeechToText(WhisperConfig())
        self.logger = None
        self.summary = SummaryGenerator()
        self.stream_sid: str | None = None
        self.call_sid: str | None = None
        self.audio_chunks: list[np.ndarray] = []
        self.silence_chunks = 0
        self.speech_started = False
        self.processing = False
        self.closed = False
        self._prep_task: asyncio.Task | None = None
        self.threshold = 0.015
        self.max_silence_chunks = 50  # 1 second at Twilio's 20 ms packets

    async def run(self) -> None:
        log("session run entered")
        while True:
            raw = await self.websocket.receive_text()
            msg = json.loads(raw)
            event = msg.get("event")
            log(f"event={event}")
            if event == "start":
                self.stream_sid = msg["start"]["streamSid"]
                self.call_sid = msg["start"].get("callSid")
                params = msg["start"].get("customParameters", {})
                log(f"custom_parameters={params}")
                patient_id = params.get("patient_id", "P003")
                self.voice_name = params.get("voice_name", self.voice_name)
                self.patient = get_patient_by_id(patient_id)
                self.agent = AgentCore(self.patient, GeminiConfig(), AgentConfig())
                self.logger = TranscriptLogger(self.patient.patient_id)
                greeting = self.agent.generate_greeting()
                log("greeting generated")
                self._prep_task = asyncio.create_task(self._prepare_for_turns())
                self.logger.log("assistant", greeting)
                if self.call_sid:
                    emit(self.call_sid, "assistant", greeting)
                await self._send_text_audio(greeting)
            elif event == "media":
                if not self.processing:
                    await self._handle_media(msg["media"]["payload"])
            elif event == "stop":
                break
        await self._finish()

    async def _handle_media(self, payload: str) -> None:
        pcm8 = mulaw_to_pcm16(base64.b64decode(payload))
        rms = float(np.sqrt(np.mean(pcm8.astype(np.float32) ** 2)) / 32768.0)
        if rms > self.threshold:
            self.speech_started = True
            self.silence_chunks = 0
        elif self.speech_started:
            self.silence_chunks += 1
        if self.speech_started:
            self.audio_chunks.append(pcm8)
        if self.speech_started and self.silence_chunks >= self.max_silence_chunks:
            audio = np.concatenate(self.audio_chunks)
            self.audio_chunks.clear()
            self.silence_chunks = 0
            self.speech_started = False
            await self._process_turn(audio)

    async def _process_turn(self, pcm8: np.ndarray) -> None:
        self.processing = True
        try:
            if self.agent is None or self.logger is None:
                return
            if self._prep_task is not None:
                await self._prep_task
            pcm16 = resample(pcm8, 8000, 16000)
            text = await asyncio.to_thread(self.stt.transcribe, pcm16, 16000)
            if not text:
                await self._send_text_audio("I'm sorry, I didn't catch that. Could you say that again?")
                return
            self.logger.log("user", text)
            if self.call_sid:
                emit(self.call_sid, "user", text)
            observation, response = await asyncio.to_thread(
                self.agent.observe_and_respond, text
            )
            self.logger.log(
                "system",
                f"confidence={observation.confidence:.2f}, sentiment={observation.sentiment}",
                metadata=observation.interpreted_data,
            )
            self.logger.log("assistant", response)
            if self.call_sid:
                emit(self.call_sid, "assistant", response)
            await self._send_text_audio(response)
            if self.agent.is_conversation_over():
                await self._complete_call()
        finally:
            self.processing = False

    async def _prepare_for_turns(self) -> None:
        """Load slow dependencies after greeting starts, not before answer."""
        if self.agent is None:
            return
        log("background prep started")
        await asyncio.gather(
            asyncio.to_thread(self.stt.load_model),
            asyncio.to_thread(self.agent.plan),
        )
        log("background prep complete")

    async def _send_text_audio(self, text: str) -> None:
        log(f"sending text audio chars={len(text)}")
        pcm8 = await asyncio.to_thread(synthesize_phone_pcm, text, self.voice_name)
        log(f"synthesized phone samples={pcm8.size}")
        if pcm8.size == 0 or not self.stream_sid:
            return
        # Twilio buffers outbound media and plays it in order. Larger chunks
        # avoid Python scheduling jitter from drip-feeding 20 ms packets.
        chunk_size = 4000  # 0.5 s at 8 kHz
        for i in range(0, len(pcm8), chunk_size):
            chunk = pcm8[i:i + chunk_size]
            payload = base64.b64encode(pcm16_to_mulaw(chunk)).decode("ascii")
            await self.websocket.send_text(json.dumps({
                "event": "media",
                "streamSid": self.stream_sid,
                "media": {"payload": payload},
            }))
        await self.websocket.send_text(json.dumps({
            "event": "mark",
            "streamSid": self.stream_sid,
            "mark": {"name": "tts_complete"},
        }))

    async def _finish(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self.logger is not None:
            self.logger.save()
        if self.agent is not None:
            self.summary.generate_and_save(self.agent)
        if self.call_sid:
            emit(self.call_sid, "call_ended")

    async def _complete_call(self) -> None:
        """Hang up PSTN call once the agent reaches a terminal phase."""
        if not self.call_sid:
            return
        cfg = TwilioConfig()
        if not cfg.account_sid or not cfg.auth_token:
            return
        log("agent completed conversation; ending Twilio call")
        await asyncio.to_thread(
            Client(cfg.account_sid, cfg.auth_token).calls(self.call_sid).update,
            status="completed",
        )
