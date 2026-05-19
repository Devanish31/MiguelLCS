"""Conversation state machine for the agentic voice AI."""
from __future__ import annotations
from enum import Enum, auto
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


class ConversationPhase(Enum):
    """High-level phases of the conversation."""
    GREETING = auto()
    CONSENT = auto()
    DATA_GATHERING = auto()
    CLARIFICATION = auto()
    SHARED_DECISION = auto()
    SUMMARY = auto()
    GOODBYE = auto()
    TERMINATED = auto()


class DataGap(Enum):
    """Specific data gaps the agent needs to resolve."""
    PACK_YEARS_CONFLICT = auto()
    SMOKING_STATUS_CONFLICT = auto()
    QUIT_DATE_MISSING = auto()
    QUIT_DATE_CONFLICT = auto()
    PACKS_PER_DAY_UNKNOWN = auto()
    YEARS_SMOKED_UNKNOWN = auto()
    CURRENT_VS_FORMER_AMBIGUOUS = auto()
    QUANTITATIVE_DATA_MISSING = auto()


@dataclass
class AgentObservation:
    """What the agent observed from a patient response."""
    raw_transcript: str
    interpreted_data: dict = field(default_factory=dict)
    answered_gap: Optional[DataGap] = None
    confidence: float = 0.5
    needs_followup: bool = False
    followup_reason: str = ""
    sentiment: str = "neutral"


@dataclass
class ConversationContext:
    """Tracks the full state of the conversation."""
    phase: ConversationPhase = ConversationPhase.GREETING
    turn_count: int = 0
    patient_consented: bool = False

    # Data resolution tracking
    identified_gaps: list[DataGap] = field(default_factory=list)
    resolved_gaps: list[DataGap] = field(default_factory=list)
    current_gap_focus: Optional[DataGap] = None

    # Accumulated evidence from patient responses
    gathered_data: dict = field(default_factory=dict)

    # Full conversation history for Gemini context
    conversation_history: list[dict] = field(default_factory=list)

    # Agent decision audit trail
    decision_log: list[dict] = field(default_factory=list)

    # Observation accumulator
    observations: list[AgentObservation] = field(default_factory=list)

    def add_turn(self, role: str, text: str) -> None:
        """Add a conversation turn."""
        self.conversation_history.append({
            "role": role,
            "text": text,
            "timestamp": datetime.now().isoformat(),
        })

    def log_decision(self, observation: str, reasoning: str, action: str) -> None:
        """Log an agent decision for the audit trail."""
        self.decision_log.append({
            "turn": self.turn_count,
            "phase": self.phase.name,
            "observation": observation,
            "reasoning": reasoning,
            "action": action,
            "timestamp": datetime.now().isoformat(),
        })

    def get_unresolved_gaps(self) -> list[DataGap]:
        """Return gaps that have not yet been resolved."""
        return [g for g in self.identified_gaps if g not in self.resolved_gaps]

    def get_history_text(self, last_n: int = 0) -> str:
        """Format conversation history as text for prompts."""
        history = self.conversation_history
        if last_n > 0:
            history = history[-last_n:]
        lines = []
        for entry in history:
            role = "AI" if entry["role"] == "assistant" else "Patient"
            lines.append(f"{role}: {entry['text']}")
        return "\n".join(lines)
