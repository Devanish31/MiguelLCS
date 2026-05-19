"""
Agentic Voice AI for Lung Cancer Screening
===========================================
Main entry point. Runs a voice-based conversation with a simulated patient
to resolve conflicting smoking history data from EHR records.

Usage:
    python main.py
"""
from __future__ import annotations
import sys
import os

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import GeminiConfig, WhisperConfig, AudioConfig, TTSConfig, AgentConfig
from patient_data.mock_patients import get_all_patients
from patient_data.models import PatientProfile
from agent.agent_core import AgentCore
from voice.voice_interface import VoiceInterface
from output.transcript_logger import TranscriptLogger
from output.summary_generator import SummaryGenerator


def select_patient() -> PatientProfile:
    """Let the user select a mock patient."""
    patients = get_all_patients()
    print("\n" + "=" * 60)
    print("  SELECT A PATIENT")
    print("=" * 60)
    for i, p in enumerate(patients, 1):
        summary = p.get_conflicts_summary()
        print(f"  {i}. {p.name} (age {p.age}, {p.sex})")
        print(f"     Conflicts: {summary}")
        print()
    while True:
        try:
            choice = int(input("  Enter patient number (1-5): ")) - 1
            if 0 <= choice < len(patients):
                return patients[choice]
        except (ValueError, IndexError):
            pass
        print("  Invalid choice. Try again.")


def run_conversation(patient: PatientProfile) -> None:
    """Run the full agentic voice conversation."""
    # Initialize components
    gemini_cfg = GeminiConfig()
    agent_cfg = AgentConfig()
    audio_cfg = AudioConfig()
    whisper_cfg = WhisperConfig()
    tts_cfg = TTSConfig()

    agent = AgentCore(patient, gemini_cfg, agent_cfg)
    voice = VoiceInterface(audio_cfg, whisper_cfg, tts_cfg)
    logger = TranscriptLogger(patient.patient_id)
    summary_gen = SummaryGenerator()

    # Pre-load Whisper model
    voice.initialize()

    # ------------------------------------------------------------------
    # 1. PLAN: Analyze patient data
    # ------------------------------------------------------------------
    print("\n[Agent planning conversation strategy...]")
    agent.plan()

    # ------------------------------------------------------------------
    # 2. GREET: Introduce and ask for consent
    # ------------------------------------------------------------------
    greeting = agent.generate_greeting()
    logger.log("assistant", greeting)
    voice.say(greeting)

    # ------------------------------------------------------------------
    # 3. MAIN LOOP: Listen -> Observe -> Decide -> Act -> Speak
    # ------------------------------------------------------------------
    max_empty_responses = 3
    empty_count = 0

    while not agent.is_conversation_over():
        # LISTEN
        patient_text = voice.listen()

        if not patient_text:
            if voice.use_voice:
                # Mic mode: retry up to max_empty_responses times
                empty_count += 1
                if empty_count >= max_empty_responses:
                    print("  [Too many empty responses, ending call]")
                    farewell = "It seems like we're having trouble hearing each other. Thank you for your time, and take care!"
                    logger.log("assistant", farewell)
                    voice.say(farewell)
                    break
                retry_msg = "I'm sorry, I didn't catch that. Could you say that again?"
                logger.log("assistant", retry_msg)
                voice.say(retry_msg)
            # Keyboard mode: just skip empty lines
            continue

        empty_count = 0
        logger.log("user", patient_text)

        # OBSERVE
        observation = agent.observe(patient_text)
        logger.log(
            "system",
            f"Observation: confidence={observation.confidence:.2f}, "
            f"sentiment={observation.sentiment}",
            metadata=observation.interpreted_data,
        )

        # DECIDE + ACT
        response = agent.decide_and_act()
        logger.log("assistant", response)

        # SPEAK
        voice.say(response)

    # ------------------------------------------------------------------
    # 4. OUTPUT: Save results
    # ------------------------------------------------------------------
    transcript_path = logger.save()
    summary_path = summary_gen.generate_and_save(agent)

    print(f"\n  Transcript saved: {transcript_path}")
    print(f"  Summary saved:    {summary_path}")


def main() -> None:
    """Main entry point."""
    print("\n" + "=" * 60)
    print("  AGENTIC VOICE AI — LUNG CANCER SCREENING PROTOTYPE")
    print("  Stanford Medicine Research")
    print("=" * 60)

    patient = select_patient()
    print(f"\n  Selected: {patient.name}")
    print(f"  You will role-play as this patient.")
    print(f"  The AI agent will ask you questions about smoking history.")
    print(f"  (Input mode will be auto-detected: mic or keyboard)\n")

    input("  Press ENTER to start the call...")
    run_conversation(patient)

    print("\n  Done. Goodbye!\n")


if __name__ == "__main__":
    main()
