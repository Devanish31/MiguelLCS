"""Decides the agent's next action based on observations and state."""
from __future__ import annotations
from typing import Optional
from agent.conversation_state import ConversationContext, ConversationPhase, DataGap


# Priority order for addressing data gaps (lower = higher priority)
GAP_PRIORITY = {
    DataGap.CURRENT_VS_FORMER_AMBIGUOUS: 1,
    DataGap.SMOKING_STATUS_CONFLICT: 1,
    DataGap.QUANTITATIVE_DATA_MISSING: 2,
    DataGap.PACK_YEARS_CONFLICT: 3,
    DataGap.PACKS_PER_DAY_UNKNOWN: 4,
    DataGap.YEARS_SMOKED_UNKNOWN: 4,
    DataGap.QUIT_DATE_MISSING: 5,
    DataGap.QUIT_DATE_CONFLICT: 5,
}


class DecisionMaker:
    """Determines what the agent should do next in the conversation."""

    def __init__(self, context: ConversationContext):
        self.ctx = context

    def decide_next_phase(self, enable_sdm: bool = True) -> ConversationPhase:
        """Determine if a phase transition is needed."""
        phase = self.ctx.phase

        if phase == ConversationPhase.GREETING:
            return ConversationPhase.CONSENT

        if phase == ConversationPhase.CONSENT:
            if self.ctx.patient_consented:
                return ConversationPhase.DATA_GATHERING
            return ConversationPhase.TERMINATED

        if phase == ConversationPhase.DATA_GATHERING:
            unresolved = self.ctx.get_unresolved_gaps()
            if len(unresolved) == 0:
                if enable_sdm:
                    return ConversationPhase.SHARED_DECISION
                return ConversationPhase.SUMMARY
            # Safety: don't go too long
            if self.ctx.turn_count > 15:
                return ConversationPhase.SUMMARY
            return ConversationPhase.DATA_GATHERING

        if phase == ConversationPhase.CLARIFICATION:
            return ConversationPhase.DATA_GATHERING

        if phase == ConversationPhase.SHARED_DECISION:
            return ConversationPhase.SUMMARY

        if phase == ConversationPhase.SUMMARY:
            return ConversationPhase.GOODBYE

        return phase

    def select_next_gap(self) -> Optional[DataGap]:
        """Pick the highest-priority unresolved gap to address next."""
        unresolved = self.ctx.get_unresolved_gaps()
        if not unresolved:
            return None
        return min(unresolved, key=lambda g: GAP_PRIORITY.get(g, 99))

    def should_clarify_last_response(self) -> bool:
        """Check if the last observation needs clarification."""
        if not self.ctx.observations:
            return False
        last = self.ctx.observations[-1]
        return last.needs_followup and last.confidence < 0.5

    def get_recent_sentiment(self) -> str:
        """Get the most recent patient sentiment."""
        if not self.ctx.observations:
            return "unknown"
        return self.ctx.observations[-1].sentiment
