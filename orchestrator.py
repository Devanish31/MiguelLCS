"""Orchestrator: bridges the GUI and AgentCore via a background thread.

In voice mode, the loop is fully automatic:
  Agent speaks (TTS) → auto-record (mic) → transcribe (Whisper) → observe → repeat

Key latency optimizations:
  - Combined Gemini call (observe+respond in one API call)
  - Sentence-pipelined TTS (play first sentence while synthesizing next)
  - Mic is always-on (persistent Bluetooth stream, no activation delay)
  - Preloaded models (Whisper + AudioRecorder ready before call starts)

Every step is timed and logged so latency bottlenecks are visible.
"""
from __future__ import annotations
import queue
import threading
import time
from typing import Callable

from config import GeminiConfig, AgentConfig, AudioConfig, WhisperConfig
from agent.agent_core import AgentCore
from agent.conversation_state import ConversationPhase
from patient_data.models import PatientProfile
from output.transcript_logger import TranscriptLogger
from output.summary_generator import SummaryGenerator
from tts import create_tts_engine
from tts.base import TTSBackend
from voice.audio_recorder import AudioRecorder
from voice.speech_to_text import SpeechToText


class Orchestrator:
    """Runs the agentic conversation loop on a background thread."""

    def __init__(self, root):
        self._root = root
        self._text_queue: queue.Queue[str] = queue.Queue()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._event_handlers: dict[str, list[Callable]] = {}

        self._agent: AgentCore | None = None
        self._tts: TTSBackend | None = None
        self._recorder: AudioRecorder | None = None
        self._stt: SpeechToText | None = None
        self._logger: TranscriptLogger | None = None
        self._summary_gen: SummaryGenerator | None = None

        self._tts_engine_name = "edge_tts"
        self._tts_voice = "Jenny (Female, US)"
        self._tts_api_key = ""
        self._input_mode = "voice"
        self._mic_device_override: int | None = None  # None = auto-detect
        # Latency tracking.
        self._t_user_audio_end: float | None = None  # set when patient stops speaking
        self._t_call_start: float | None = None      # set when Start Call clicked
        self._last_response_latency: float | None = None  # filled in _on_sentence

        # Preload synchronization: conversation thread waits for models
        self._preload_done = threading.Event()

    # ------------------------------------------------------------------
    # Event system
    # ------------------------------------------------------------------

    def on(self, event_name: str, handler: Callable) -> None:
        self._event_handlers.setdefault(event_name, []).append(handler)

    def _post(self, event_name: str, **kwargs) -> None:
        handlers = self._event_handlers.get(event_name, [])
        for handler in handlers:
            self._root.after(0, lambda h=handler, kw=kwargs: h(**kw))

    def _log(self, msg: str) -> None:
        elapsed = time.perf_counter() - self._t0 if hasattr(self, '_t0') else 0
        print(f"  [{elapsed:7.2f}s] {msg}")

    def _timed(self, label: str):
        """Context manager to time a block and log it."""
        return _TimedBlock(label, self)

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------

    def set_tts_config(self, engine_name: str, voice: str, api_key: str) -> None:
        self._tts_engine_name = engine_name
        self._tts_voice = voice
        self._tts_api_key = api_key

    def set_input_mode(self, mode: str) -> None:
        self._input_mode = mode

    def set_mic_device(self, device_id: int | None) -> None:
        """Pin the mic to a specific device (None = re-auto-detect).
        Closes any existing recorder and rebuilds it on next use."""
        if device_id == self._mic_device_override and self._recorder is not None:
            return
        self._mic_device_override = device_id
        # Tear down existing recorder so the next access rebuilds with new device
        if self._recorder is not None:
            try:
                self._recorder.close()
            except Exception:
                pass
            self._recorder = None
        # Rebuild in background so the GUI stays responsive
        threading.Thread(target=self._rebuild_recorder, daemon=True).start()

    def _rebuild_recorder(self) -> None:
        try:
            self._recorder = AudioRecorder(
                AudioConfig(), device_override=self._mic_device_override
            )
            if self._recorder.input_device is not None:
                import sounddevice as sd
                name = sd.query_devices(self._recorder.input_device)["name"]
                self._post("status_message",
                           text=f"Mic: {name} (id={self._recorder.input_device})")
            else:
                self._post("status_message", text="Mic: no working device found")
        except Exception as e:
            self._post("error", message=f"Mic switch error: {e}")

    # ------------------------------------------------------------------
    # Preload (runs once at app startup)
    # ------------------------------------------------------------------

    def preload_models(self) -> None:
        threading.Thread(target=self._do_preload, daemon=True).start()

    def _do_preload(self) -> None:
        self._t0 = time.perf_counter()
        try:
            self._post("status_message", text="Loading audio + Whisper...")
            with self._timed("AudioRecorder init"):
                self._recorder = AudioRecorder(AudioConfig(),
                                               device_override=self._mic_device_override)
            with self._timed("Whisper model load"):
                self._stt = SpeechToText(WhisperConfig())
                self._stt.load_model()
            # Warm up the Bluetooth stream right away
            if self._recorder.is_bluetooth:
                with self._timed("Bluetooth stream warm-up"):
                    self._recorder._ensure_stream()
            self._post("status_message", text="Models ready.")
            self._log("Pre-load complete.")
        except Exception as e:
            self._log(f"Pre-load error: {e}")
            self._post("error", message=f"Model preload error: {e}")
        finally:
            # Signal that preload is done (even if it failed)
            self._preload_done.set()

    # ------------------------------------------------------------------
    # Start / stop
    # ------------------------------------------------------------------

    def start_call(self, patient: PatientProfile) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._t_call_start = time.perf_counter()
        self._stop_event.clear()
        while not self._text_queue.empty():
            self._text_queue.get_nowait()
        self._thread = threading.Thread(
            target=self._run_conversation, args=(patient,), daemon=True,
        )
        self._thread.start()

    def stop_call(self) -> None:
        self._stop_event.set()
        self._text_queue.put("")
        if self._tts:
            self._tts.stop()

    def submit_text(self, text: str) -> None:
        self._text_queue.put(text)

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------------
    # Manual recording (keyboard mode mic button)
    # ------------------------------------------------------------------

    def start_recording(self) -> None:
        threading.Thread(target=self._record_and_transcribe, daemon=True).start()

    def _record_and_transcribe(self) -> str:
        try:
            self._post("listening_start")
            if self._recorder is None:
                self._recorder = AudioRecorder(AudioConfig(),
                                               device_override=self._mic_device_override)
            if self._stt is None:
                self._stt = SpeechToText(WhisperConfig())
                self._stt.load_model()

            audio = self._recorder.record_until_silence()
            self._post("listening_end")
            if len(audio) == 0:
                self._post("transcription_result", text="")
                return ""
            text = self._stt.transcribe(audio, sample_rate=16000)
            if text:
                self._post("transcription_result", text=text)
                self._text_queue.put(text)
            return text
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._post("listening_end")
            self._post("error", message=f"Recording error: {e}")
            return ""

    def _auto_record_and_get_text(self) -> str:
        """Voice mode: record + transcribe, return text directly."""
        try:
            self._post("listening_start")

            if self._recorder is None:
                self._recorder = AudioRecorder(AudioConfig(),
                                               device_override=self._mic_device_override)
            if self._stt is None:
                self._stt = SpeechToText(WhisperConfig())
                self._stt.load_model()

            with self._timed("Mic recording"):
                audio = self._recorder.record_until_silence()
            # Subtract the silence-detection buffer so the timestamp reflects
            # when the patient actually stopped talking, not when the recorder
            # confirmed silence.
            self._t_user_audio_end = (
                time.perf_counter() - self._recorder.config.silence_duration
            )
            self._post("listening_end")

            if len(audio) == 0:
                self._log("No speech detected.")
                self._t_user_audio_end = None
                return ""

            with self._timed(f"Whisper transcribe ({len(audio)} samples)"):
                text = self._stt.transcribe(audio, sample_rate=16000)
            self._log(f"Transcribed: '{text}'")
            if text:
                self._post("transcription_result", text=text)
            return text

        except Exception as e:
            self._log(f"Record/transcribe error: {e}")
            import traceback
            traceback.print_exc()
            self._post("listening_end")
            self._post("error", message=f"Recording error: {e}")
            return ""

    # ------------------------------------------------------------------
    # Main conversation loop
    # ------------------------------------------------------------------

    def _run_conversation(self, patient: PatientProfile) -> None:
        try:
            self._t0 = time.perf_counter()
            self._post("call_started")
            self._log(f"=== CALL START: {patient.name} (mode={self._input_mode}) ===")

            # TTS
            with self._timed(f"TTS init ({self._tts_engine_name})"):
                try:
                    self._tts = create_tts_engine(
                        self._tts_engine_name,
                        voice=self._tts_voice,
                        api_key=self._tts_api_key,
                    )
                except Exception as e:
                    self._log(f"TTS fallback to pyttsx3: {e}")
                    self._tts = create_tts_engine("pyttsx3")

            # Wait for preloaded models (Whisper + AudioRecorder)
            # instead of creating duplicates
            if not self._preload_done.is_set():
                self._log("Waiting for model preload to finish...")
                self._preload_done.wait(timeout=30)

            # Only create if preload failed or wasn't called
            if self._recorder is None:
                with self._timed("AudioRecorder init (fallback)"):
                    self._recorder = AudioRecorder(AudioConfig(),
                                               device_override=self._mic_device_override)
            if self._stt is None:
                with self._timed("Whisper load (fallback)"):
                    self._stt = SpeechToText(WhisperConfig())
                    self._stt.load_model()

            self._log("Recorder + Whisper ready (preloaded)")

            # Agent
            with self._timed("AgentCore init"):
                gemini_cfg = GeminiConfig()
                agent_cfg = AgentConfig()
                self._agent = AgentCore(patient, gemini_cfg, agent_cfg)

            self._logger = TranscriptLogger(patient.patient_id)
            self._summary_gen = SummaryGenerator()

            # PLAN
            self._post("phase_change", phase="PLANNING")
            self._post("thinking_start")
            with self._timed("Agent PLAN (Gemini)"):
                self._agent.plan()
            self._post("thinking_end")
            total_gaps = len(self._agent.context.identified_gaps)
            self._post("gap_update", resolved=0, total=total_gaps)

            if self._stop_event.is_set():
                return

            # GREET
            self._post("phase_change", phase="GREETING")
            greeting = self._agent.generate_greeting()
            self._logger.log("assistant", greeting)
            if self._t_call_start is not None:
                startup_latency = time.perf_counter() - self._t_call_start
                self._log(f"Startup latency (Call button -> greeting): "
                          f"{startup_latency:.2f}s")
                greeting_display = f"{greeting}\n\n(Startup latency: {startup_latency:.2f}s)"
            else:
                greeting_display = greeting
            self._post("assistant_message", text=greeting_display)
            with self._timed("TTS speak greeting"):
                self._speak(greeting)
            self._post("phase_change", phase="CONSENT")

            # MAIN LOOP
            empty_count = 0
            max_empty = 3
            turn_num = 0

            while not self._agent.is_conversation_over() and not self._stop_event.is_set():
                turn_num += 1
                self._log(f"--- Turn {turn_num} ---")

                # GET INPUT
                patient_text = ""
                if self._input_mode == "voice":
                    self._post("waiting_for_input")
                    patient_text = self._auto_record_and_get_text()
                else:
                    self._post("waiting_for_input")
                    while not self._stop_event.is_set():
                        try:
                            patient_text = self._text_queue.get(timeout=0.5)
                            break
                        except queue.Empty:
                            continue

                if self._stop_event.is_set():
                    break

                if not patient_text:
                    empty_count += 1
                    self._log(f"Empty ({empty_count}/{max_empty})")
                    if empty_count >= max_empty:
                        farewell = "It seems like we're having trouble hearing each other. Thank you for your time!"
                        self._post("assistant_message", text=farewell)
                        self._logger.log("assistant", farewell)
                        self._speak(farewell)
                        break
                    if self._input_mode == "voice":
                        retry = "I'm sorry, I didn't catch that. Could you say that again?"
                        self._post("assistant_message", text=retry)
                        self._speak(retry)
                    continue

                empty_count = 0
                self._post("user_message", text=patient_text)
                self._logger.log("user", patient_text)
                self._log(f"Patient: '{patient_text[:80]}'")

                # ── STREAMING: Gemini → TTS pipeline ──
                # As Gemini streams its response, each sentence is sent
                # to TTS immediately. The user hears the first sentence
                # while Gemini is still generating the analysis JSON.
                self._post("thinking_start")
                t_start = time.perf_counter()
                first_sentence_time = [None]  # mutable for closure
                tts_sentences = []  # track what was spoken

                def _on_sentence(sentence: str) -> None:
                    """Called by agent as each sentence streams from Gemma."""
                    if self._stop_event.is_set() or not sentence.strip():
                        return
                    if first_sentence_time[0] is None:
                        first_sentence_time[0] = time.perf_counter() - t_start
                        self._post("thinking_end")
                        self._log(f"  First sentence in {first_sentence_time[0]:.2f}s: "
                                  f"'{sentence[:60]}'")
                        # Patient-perceived response latency: from when they
                        # stopped talking to first audible agent word.
                        if self._t_user_audio_end is not None:
                            latency = time.perf_counter() - self._t_user_audio_end
                            self._log(f"  Response latency (audio-end -> first word): "
                                      f"{latency:.2f}s")
                            self._last_response_latency = latency
                            self._t_user_audio_end = None  # consumed
                    self._post("tts_start")
                    tts_sentences.append(sentence)
                    # Synthesize + play this sentence immediately
                    if hasattr(self._tts, 'speak_sentence_blocking'):
                        self._tts.speak_sentence_blocking(sentence)
                    else:
                        self._tts.speak_blocking(sentence)
                    self._post("tts_end")

                observation, response = self._agent.observe_and_respond(
                    patient_text, on_sentence=_on_sentence
                )
                total_time = time.perf_counter() - t_start
                self._log(f"  Gemini+TTS total: {total_time:.2f}s "
                          f"(first audio at {first_sentence_time[0] or 0:.2f}s)")
                self._log(f"  conf={observation.confidence:.2f} sent={observation.sentiment}")

                self._logger.log(
                    "system",
                    f"confidence={observation.confidence:.2f}, sentiment={observation.sentiment}",
                    metadata=observation.interpreted_data,
                )
                self._post("confidence_update", value=observation.confidence)
                resolved = len(self._agent.context.resolved_gaps)
                self._post("gap_update", resolved=resolved, total=total_gaps)
                self._post("turn_update", turn=self._agent.context.turn_count)

                phase_name = self._agent.context.phase.name
                self._post("phase_change", phase=phase_name)
                self._log(f"  Phase: {phase_name}")
                self._log(f"  Response: '{response[:100]}'")

                self._logger.log("assistant", response)
                if self._last_response_latency is not None:
                    response_display = (
                        f"{response}\n\n(Response latency: "
                        f"{self._last_response_latency:.2f}s)"
                    )
                    self._last_response_latency = None
                else:
                    response_display = response
                self._post("assistant_message", text=response_display)

            # SAVE
            with self._timed("Save transcript + summary"):
                transcript_path = self._logger.save()
                summary_path = self._summary_gen.generate_and_save(self._agent)

            self._log(f"=== CALL END (total {time.perf_counter() - self._t0:.1f}s) ===")
            self._post("call_ended",
                       transcript_path=transcript_path,
                       summary_path=summary_path)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self._log(f"FATAL: {e}")
            self._post("error", message=f"Conversation error: {e}")
            self._post("call_ended", transcript_path="", summary_path="")
        finally:
            self._cleanup_call_resources()

    def _cleanup_call_resources(self) -> None:
        """Release per-call audio resources so the next call starts fresh.

        Leaving the TTS object and the recorder's persistent stream alive
        between calls causes PortAudio to segfault on Windows when the next
        call re-acquires the same devices. We close the stream (recorder
        object survives, so the preload benefit is kept) and drop the TTS
        reference entirely.
        """
        try:
            if self._tts is not None:
                try:
                    self._tts.stop()
                except Exception:
                    pass
                self._tts = None
        except Exception:
            pass
        try:
            if self._recorder is not None:
                self._recorder.close()  # next call's _ensure_stream re-opens it
        except Exception:
            pass

    def _speak(self, text: str) -> None:
        if self._tts and not self._stop_event.is_set():
            self._post("tts_start")
            try:
                self._tts.speak_blocking(text)
            except Exception as e:
                self._log(f"TTS Error: {e}")
            self._post("tts_end")


class _TimedBlock:
    """Context manager that logs elapsed time for a block."""
    def __init__(self, label: str, orchestrator: Orchestrator):
        self._label = label
        self._orch = orchestrator
    def __enter__(self):
        self._start = time.perf_counter()
        return self
    def __exit__(self, *args):
        elapsed = time.perf_counter() - self._start
        self._orch._log(f"{self._label}: {elapsed:.2f}s")
