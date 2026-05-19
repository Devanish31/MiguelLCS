"""Core agentic reasoning engine powered by Gemini."""
from __future__ import annotations
import json
import re
from typing import Callable
import ollama

from config import GeminiConfig, AgentConfig
from agent.conversation_state import (
    ConversationContext, ConversationPhase, AgentObservation, DataGap,
)
from agent.prompt_templates import (
    SYSTEM_INSTRUCTION, PLANNING_PROMPT, OBSERVATION_PROMPT,
    NEXT_ACTION_PROMPT, QUESTION_GENERATION_PROMPT,
    SDM_TRANSITION_PROMPT, SUMMARY_PROMPT, CONSENT_CHECK_PROMPT,
    OBSERVE_AND_RESPOND_PROMPT,
)
from agent.smoking_resolver import SmokingConflictAnalyzer, SmokingDataMerger
from agent.decision_maker import DecisionMaker
from patient_data.models import PatientProfile
from rag.sdm_retriever import SDMRetriever, RetrievedChunk


class AgentCore:
    """The agentic AI that drives the conversation.

    Implements the Plan / Decide / Act / Observe loop:
      - plan()           -> analyze patient, identify gaps, create strategy
      - observe(text)    -> interpret patient response, extract data
      - decide_and_act() -> choose next action, generate speech
    """

    def __init__(
        self,
        patient: PatientProfile,
        gemini_config: GeminiConfig | None = None,
        agent_config: AgentConfig | None = None,
    ):
        self.patient = patient
        gemini_config = gemini_config or GeminiConfig()
        self.agent_config = agent_config or AgentConfig()

        # Ollama / Gemma 3 client (local)
        self.client = ollama.Client(host=gemini_config.ollama_host)
        self.model_name = gemini_config.model_name
        self._system_instruction = SYSTEM_INSTRUCTION
        self._speech_options = {
            "temperature": gemini_config.temperature,
            "num_predict": gemini_config.max_output_tokens,
        }
        self._json_options = {
            "temperature": 0.2,
            "num_predict": gemini_config.max_output_tokens,
        }

        # Smoking analysis
        self.analyzer = SmokingConflictAnalyzer(patient)
        self.merger = SmokingDataMerger(patient)

        # Conversation state
        self.context = ConversationContext()
        self.decision_maker = DecisionMaker(self.context)
        self.sdm_retriever = SDMRetriever()

        # Plan output (stored after plan() call)
        self._plan: dict = {}

    def retrieve_sdm_context(self, question: str, k: int = 3) -> list[RetrievedChunk]:
        """Optional local SDM retrieval hook; not used in the live flow yet."""
        return self.sdm_retriever.search(question, k=k)

    # ------------------------------------------------------------------
    # PLAN
    # ------------------------------------------------------------------

    def plan(self) -> dict:
        """PLAN phase: Analyze patient data, identify conflicts, plan strategy."""
        gaps = self.analyzer.identify_data_gaps()
        self.context.identified_gaps = gaps

        prompt = PLANNING_PROMPT.format(
            patient_data=self.analyzer.format_for_prompt(),
            conflicts=self.analyzer.format_conflicts(),
        )
        response = self._call_gemini_json(prompt)
        self._plan = self._parse_json(response)

        self.context.log_decision(
            observation=f"Analyzed {len(self.patient.smoking_records)} clinical notes",
            reasoning=f"Found {len(gaps)} data gaps: {[g.name for g in gaps]}",
            action="Created conversation plan",
        )
        print(f"  [Agent Plan] {len(gaps)} gaps to resolve: "
              f"{[g.name for g in gaps]}")
        return self._plan

    # ------------------------------------------------------------------
    # GREETING
    # ------------------------------------------------------------------

    def generate_greeting(self) -> str:
        """Generate the initial greeting and transition to CONSENT phase."""
        self.context.phase = ConversationPhase.GREETING
        greeting = (
            f"Hi, is this {self.patient.name}? "
            f"This is a research assistant from Stanford Medicine. "
            f"I just have a few quick questions about your smoking history — should only take a couple minutes. Is that okay?"
        )
        self.context.add_turn("assistant", greeting)
        self.context.phase = ConversationPhase.CONSENT
        return greeting

    # ------------------------------------------------------------------
    # OBSERVE
    # ------------------------------------------------------------------

    def observe(self, patient_text: str) -> AgentObservation:
        """OBSERVE phase: Interpret a patient's spoken response."""
        self.context.add_turn("user", patient_text)
        self.context.turn_count += 1

        # --- Handle consent phase ---
        if self.context.phase == ConversationPhase.CONSENT:
            return self._observe_consent(patient_text)

        # --- Handle goodbye/summary confirmation ---
        if self.context.phase in (ConversationPhase.SUMMARY, ConversationPhase.GOODBYE):
            obs = AgentObservation(raw_transcript=patient_text, confidence=1.0)
            self.context.observations.append(obs)
            return obs

        # --- Main observation: use Gemini to interpret ---
        prompt = OBSERVATION_PROMPT.format(
            conversation_history=self.context.get_history_text(last_n=8),
            current_gap=self.context.current_gap_focus.name if self.context.current_gap_focus else "General",
            patient_response=patient_text,
        )
        response = self._call_gemini_json(prompt)
        parsed = self._parse_json(response)

        extracted = parsed.get("extracted_data", {})
        observation = AgentObservation(
            raw_transcript=patient_text,
            interpreted_data=extracted,
            answered_gap=self.context.current_gap_focus,
            confidence=float(parsed.get("confidence", 0.5)),
            needs_followup=bool(parsed.get("needs_clarification", False)),
            followup_reason=parsed.get("clarification_reason", ""),
            sentiment=parsed.get("patient_sentiment", "neutral"),
        )

        # Update merged data with extracted info
        self.merger.update_from_response(extracted)
        self.context.gathered_data.update(
            {k: v for k, v in extracted.items() if v is not None and k != "other_info"}
        )

        # Mark gap as resolved if confidence is high enough
        if (observation.answered_gap
                and observation.confidence >= self.agent_config.confidence_threshold
                and not observation.needs_followup):
            if observation.answered_gap not in self.context.resolved_gaps:
                self.context.resolved_gaps.append(observation.answered_gap)
                print(f"  [Gap Resolved] {observation.answered_gap.name} "
                      f"(confidence: {observation.confidence:.2f})")

        self.context.observations.append(observation)

        self.context.log_decision(
            observation=f"Patient said: '{patient_text[:80]}...' | "
                        f"Confidence: {observation.confidence:.2f} | "
                        f"Sentiment: {observation.sentiment}",
            reasoning=f"Extracted: {extracted} | "
                      f"Needs followup: {observation.needs_followup}",
            action="Observed and extracted data",
        )
        return observation

    # ------------------------------------------------------------------
    # DECIDE + ACT
    # ------------------------------------------------------------------

    def decide_and_act(self) -> str:
        """DECIDE + ACT: Choose and execute the next action. Returns speech text."""
        # Phase transition
        new_phase = self.decision_maker.decide_next_phase(
            enable_sdm=self.agent_config.enable_shared_decision_making
        )
        if new_phase != self.context.phase:
            print(f"  [Phase] {self.context.phase.name} -> {new_phase.name}")
            self.context.phase = new_phase

        # Generate response based on current phase
        response = self._act_for_phase()
        self.context.add_turn("assistant", response)
        return response

    def _act_for_phase(self) -> str:
        """Generate the appropriate response for the current phase."""
        phase = self.context.phase

        if phase == ConversationPhase.TERMINATED:
            return "I completely understand. Thank you for your time, and take care."

        if phase == ConversationPhase.DATA_GATHERING:
            # Should we clarify last response or ask about a new gap?
            if self.decision_maker.should_clarify_last_response():
                return self._generate_clarification()
            else:
                next_gap = self.decision_maker.select_next_gap()
                self.context.current_gap_focus = next_gap
                if next_gap:
                    return self._generate_question(next_gap)
                else:
                    # All gaps resolved, move on
                    self.context.phase = ConversationPhase.SUMMARY
                    return self._generate_summary()

        if phase == ConversationPhase.SHARED_DECISION:
            return self._generate_sdm_transition()

        if phase == ConversationPhase.SUMMARY:
            return self._generate_summary()

        if phase == ConversationPhase.GOODBYE:
            return self._generate_goodbye()

        return "Thank you for sharing that."

    # ------------------------------------------------------------------
    # COMBINED: Observe + Respond in ONE Gemini call
    # ------------------------------------------------------------------

    def observe_and_respond(self, patient_text: str,
                            on_sentence: Callable[[str], None] | None = None,
                            ) -> tuple[AgentObservation, str]:
        """OBSERVE + RESPOND with streaming: calls on_sentence() as soon as
        each sentence of the spoken response is available from Gemini.

        The prompt outputs SPEAK text first, then JSON analysis after '---'.
        We stream Gemini tokens, extract sentences from the SPEAK line, and
        call on_sentence(sentence) immediately so TTS can start while Gemini
        is still generating the analysis JSON.

        Args:
            patient_text: What the patient said.
            on_sentence: Callback invoked with each complete sentence for TTS.
                         If None, waits for full response (non-streaming fallback).

        Returns (observation, full_response_text).
        """
        self.context.add_turn("user", patient_text)
        self.context.turn_count += 1

        # --- Handle consent phase (lightweight, no combined prompt needed) ---
        if self.context.phase == ConversationPhase.CONSENT:
            obs = self._observe_consent(patient_text)
            new_phase = self.decision_maker.decide_next_phase(
                enable_sdm=self.agent_config.enable_shared_decision_making
            )
            if new_phase != self.context.phase:
                self.context.phase = new_phase
            response = self._act_for_phase()
            self.context.add_turn("assistant", response)
            if on_sentence:
                on_sentence(response)
            return obs, response

        # --- Handle goodbye/summary (no Gemini needed) ---
        if self.context.phase in (ConversationPhase.SUMMARY, ConversationPhase.GOODBYE):
            obs = AgentObservation(raw_transcript=patient_text, confidence=1.0)
            self.context.observations.append(obs)
            new_phase = self.decision_maker.decide_next_phase(
                enable_sdm=self.agent_config.enable_shared_decision_making
            )
            if new_phase != self.context.phase:
                self.context.phase = new_phase
            response = self._act_for_phase()
            self.context.add_turn("assistant", response)
            if on_sentence:
                on_sentence(response)
            return obs, response

        # --- Main: STREAMING observe + respond ---
        unresolved = self.context.get_unresolved_gaps()
        prompt = OBSERVE_AND_RESPOND_PROMPT.format(
            conversation_history=self.context.get_history_text(last_n=8),
            current_gap=self.context.current_gap_focus.name if self.context.current_gap_focus else "General",
            unresolved_gaps=[g.name for g in unresolved],
            gathered_data=json.dumps(self.context.gathered_data, default=str),
            ehr_context=self.analyzer.format_for_prompt(),
            patient_response=patient_text,
        )

        # Stream Gemini response
        speech_text, json_text = self._stream_gemini_speak_then_json(
            prompt, on_sentence=on_sentence
        )

        # Parse the JSON analysis part
        parsed = self._parse_json(json_text) if json_text else {}

        # Map compact field names to full names
        data = parsed.get("data", {})
        extracted = {
            "packs_per_day": data.get("ppd"),
            "years_smoked": data.get("yrs"),
            "quit_date": data.get("quit"),
            "current_status": data.get("status"),
            "pack_years": None,  # computed downstream
            "years_since_quit": None,
            "other_info": "",
        }

        observation = AgentObservation(
            raw_transcript=patient_text,
            interpreted_data=extracted,
            answered_gap=self.context.current_gap_focus,
            confidence=float(parsed.get("conf", 0.5)),
            needs_followup=bool(parsed.get("clar", False)),
            followup_reason="",
            sentiment=str(parsed.get("sent", "neutral")),
        )

        # Update merged data
        self.merger.update_from_response(extracted)
        self.context.gathered_data.update(
            {k: v for k, v in extracted.items() if v is not None and k != "other_info"}
        )

        # Mark gap resolved if confident
        if (observation.answered_gap
                and observation.confidence >= self.agent_config.confidence_threshold
                and not observation.needs_followup):
            if observation.answered_gap not in self.context.resolved_gaps:
                self.context.resolved_gaps.append(observation.answered_gap)
                print(f"  [Gap Resolved] {observation.answered_gap.name} "
                      f"(confidence: {observation.confidence:.2f})")

        self.context.observations.append(observation)

        # Phase transition (pure Python, instant)
        new_phase = self.decision_maker.decide_next_phase(
            enable_sdm=self.agent_config.enable_shared_decision_making
        )
        if new_phase != self.context.phase:
            print(f"  [Phase] {self.context.phase.name} -> {new_phase.name}")
            self.context.phase = new_phase

        # Use the streamed speech text
        next_response = speech_text

        # For phase transitions that need special responses (SDM, Summary, Goodbye),
        # use the phase-specific generators instead
        if self.context.phase in (ConversationPhase.SHARED_DECISION,
                                   ConversationPhase.SUMMARY,
                                   ConversationPhase.GOODBYE):
            next_response = self._act_for_phase()
            if on_sentence:
                on_sentence(next_response)
        elif not next_response:
            next_response = self._act_for_phase()
            if on_sentence:
                on_sentence(next_response)

        # Update gap focus for next turn
        if self.context.phase == ConversationPhase.DATA_GATHERING:
            next_gap = self.decision_maker.select_next_gap()
            self.context.current_gap_focus = next_gap

        self.context.add_turn("assistant", next_response)

        self.context.log_decision(
            observation=f"Patient: '{patient_text[:60]}' | conf={observation.confidence:.2f}",
            reasoning=f"Extracted: {extracted} | Phase: {self.context.phase.name}",
            action=f"Streamed observe+respond",
        )

        return observation, next_response

    # ------------------------------------------------------------------
    # Conversation state
    # ------------------------------------------------------------------

    def is_conversation_over(self) -> bool:
        """Check if the conversation has reached a terminal state."""
        return self.context.phase in (
            ConversationPhase.GOODBYE,
            ConversationPhase.TERMINATED,
        )

    # ------------------------------------------------------------------
    # Final output
    # ------------------------------------------------------------------

    def get_final_output(self) -> dict:
        """Produce the structured JSON output after the call."""
        resolved = self.merger.generate_resolved_history()
        return {
            "patient_id": self.patient.patient_id,
            "patient_name": self.patient.name,
            "resolved_smoking_history": {
                "current_status": resolved.current_status.value,
                "pack_years": resolved.pack_years,
                "packs_per_day": resolved.packs_per_day,
                "years_smoked": resolved.years_smoked,
                "quit_date": str(resolved.quit_date) if resolved.quit_date else None,
                "years_since_quit": resolved.years_since_quit,
                "meets_uspstf_criteria": resolved.meets_uspstf_criteria,
            },
            "confidence": resolved.confidence,
            "resolution_notes": resolved.resolution_notes,
            "conversation_transcript": self.context.conversation_history,
            "agent_decision_audit_trail": self.context.decision_log,
            "total_turns": self.context.turn_count,
        }

    # ------------------------------------------------------------------
    # Private: Gemini calls
    # ------------------------------------------------------------------

    def _call_gemini_speech(self, prompt: str) -> str:
        """Call Gemma (local Ollama) with system instruction for spoken text.

        ``think=False`` is critical: Gemma 4 defaults to a thinking-mode that
        emits hundreds of tokens to a separate ``thinking`` field (invisible
        in ``content``) before producing the answer. Leaving it on adds
        ~1.5-2s of wall-clock latency on every call for no benefit here.
        """
        response = self.client.chat(
            model=self.model_name,
            messages=[
                {"role": "system", "content": self._system_instruction},
                {"role": "user", "content": prompt},
            ],
            options=self._speech_options,
            think=False,
        )
        return response["message"]["content"].strip()

    def _call_gemini_json(self, prompt: str) -> str:
        """Call Gemma (local Ollama) without system instruction for JSON analysis."""
        response = self.client.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            options=self._json_options,
            think=False,
        )
        return response["message"]["content"].strip()

    def _stream_gemini_speak_then_json(
        self,
        prompt: str,
        on_sentence: Callable[[str], None] | None = None,
    ) -> tuple[str, str]:
        """Stream Gemini response, extracting SPEAK text and JSON separately.

        The prompt format is:
            SPEAK: <spoken text>
            ---
            {json analysis}

        As tokens stream in, we detect complete sentences in the SPEAK portion
        and call on_sentence() immediately. This lets TTS start playing while
        Gemini is still generating the JSON analysis.

        Returns (speech_text, json_text).
        """
        accumulated = ""
        speech_text = ""
        json_text = ""
        in_speech = True  # Start in speech mode (before ---)
        speech_sent = ""  # What we've already sent to on_sentence
        got_speak_prefix = False

        try:
            stream = self.client.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                options=self._json_options,
                stream=True,
                think=False,  # see _call_gemini_speech for rationale
            )
            for chunk in stream:
                text = chunk.get("message", {}).get("content", "") or ""
                accumulated += text

                if in_speech:
                    # Check if we've hit the --- separator
                    if "---" in accumulated:
                        parts = accumulated.split("---", 1)
                        speech_raw = parts[0]
                        json_text = parts[1] if len(parts) > 1 else ""
                        in_speech = False

                        # Clean up the speech text
                        speech_text = self._clean_speak_text(speech_raw)

                        # Send any remaining unsent speech
                        unsent = speech_text[len(speech_sent):].strip()
                        if unsent and on_sentence:
                            on_sentence(unsent)
                    else:
                        # Still in speech portion — release whatever we can.
                        speech_so_far = self._clean_speak_text(accumulated)
                        if on_sentence and speech_so_far:
                            unsent = speech_so_far[len(speech_sent):]
                            flush_end = self._find_flush_point(unsent)
                            if flush_end > 0:
                                to_send = unsent[:flush_end].strip()
                                if to_send:
                                    on_sentence(to_send)
                                    speech_sent += unsent[:flush_end]
                else:
                    # Accumulating JSON after ---
                    json_text += text

        except Exception as e:
            print(f"  [Gemma stream error: {e}]")
            # Fallback: try to parse whatever we got
            if "---" in accumulated:
                parts = accumulated.split("---", 1)
                speech_text = self._clean_speak_text(parts[0])
                json_text = parts[1] if len(parts) > 1 else ""
            else:
                speech_text = self._clean_speak_text(accumulated)

        # If we never hit ---, treat everything as speech
        if in_speech:
            speech_text = self._clean_speak_text(accumulated)
            # Send any unsent speech
            unsent = speech_text[len(speech_sent):].strip()
            if unsent and on_sentence:
                on_sentence(unsent)

        return speech_text, json_text.strip()

    # Threshold of word count in the unsent buffer before clause-level
    # (comma/semicolon) flushes are allowed. Sentence-level flushes are
    # always allowed regardless of word count.
    _CLAUSE_MIN_WORDS = 6

    @staticmethod
    def _find_flush_point(unsent: str) -> int:
        """Return the cut index into `unsent` we should flush to TTS, or -1.

        Picks the latest eligible boundary, where eligible means:
          * Sentence boundary: '.', '?', '!' followed by any whitespace
            (space, newline, tab). Always eligible.
          * Clause boundary: ',', ';' followed by any whitespace, BUT only
            when the text up to and including that boundary has at least
            ``_CLAUSE_MIN_WORDS`` words.

        Returning the latest acceptable boundary maximises the chunk handed
        to TTS, which keeps prosody natural while still cutting latency.
        """
        import re
        best = -1
        # Sentence boundaries.
        for m in re.finditer(r'[.!?]\s', unsent):
            end = m.start() + 1  # cut just after the punctuation
            if end > best:
                best = end
        # Clause boundaries — only when the prefix has enough words.
        for m in re.finditer(r'[,;]\s', unsent):
            end = m.start() + 1
            if end <= best:
                continue
            if len(unsent[:end].split()) >= AgentCore._CLAUSE_MIN_WORDS:
                best = end
        return best

    @staticmethod
    def _clean_speak_text(raw: str) -> str:
        """Remove 'SPEAK:' prefix and whitespace from speech text."""
        text = raw.strip()
        if text.upper().startswith("SPEAK:"):
            text = text[6:]
        elif text.upper().startswith("SPEAK :"):
            text = text[7:]
        return text.strip()

    def _parse_json(self, text: str) -> dict:
        """Safely parse JSON from Gemini, handling markdown fences and truncation."""
        text = text.strip()
        # Remove markdown fences
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            text = text.rsplit("```", 1)[0]
        text = text.strip()

        # First try: direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Second try: find the outermost { ... } and try to fix it
        start = text.find("{")
        if start == -1:
            print(f"  [Warning] No JSON object in Gemini response: {text[:100]}")
            return {"raw_text": text, "parse_error": True}

        # Try progressively larger substrings
        brace_count = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                brace_count += 1
            elif text[i] == "}":
                brace_count -= 1
                if brace_count == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break

        # Third try: repair truncated JSON by closing open structures
        json_text = text[start:]
        # Close open strings, arrays, objects
        repairs = ['}', '"}', ']}', '"]}', '"}]}'  , '"}}']
        for repair in repairs:
            try:
                return json.loads(json_text + repair)
            except json.JSONDecodeError:
                continue

        # Last resort: try with more aggressive repair
        try:
            # Close all open braces/brackets
            open_braces = json_text.count("{") - json_text.count("}")
            open_brackets = json_text.count("[") - json_text.count("]")
            fixed = json_text + '"' + "]" * open_brackets + "}" * open_braces
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass

        print(f"  [Warning] Could not parse JSON: {text[:150]}...")
        return {"raw_text": text, "parse_error": True}

    # ------------------------------------------------------------------
    # Private: Phase-specific actions
    # ------------------------------------------------------------------

    def _observe_consent(self, text: str) -> AgentObservation:
        """Interpret whether the patient consented."""
        prompt = CONSENT_CHECK_PROMPT.format(patient_response=text)
        response = self._call_gemini_json(prompt).strip().upper()
        consented = "YES" in response

        self.context.patient_consented = consented
        obs = AgentObservation(
            raw_transcript=text,
            confidence=0.9 if consented else 0.9,
            sentiment="cooperative" if consented else "resistant",
        )
        self.context.observations.append(obs)

        if consented:
            self.context.log_decision(
                observation=f"Patient response to consent: '{text}'",
                reasoning="Patient consented to participate",
                action="Proceeding to data gathering",
            )
        else:
            self.context.log_decision(
                observation=f"Patient response to consent: '{text}'",
                reasoning="Patient declined or was unclear",
                action="Terminating call",
            )
        return obs

    def _generate_question(self, gap: DataGap) -> str:
        """Use Gemini to generate a question targeting a specific gap."""
        prompt = QUESTION_GENERATION_PROMPT.format(
            target_gap=gap.name,
            ehr_context=self.analyzer.format_for_prompt(),
            gathered_data=json.dumps(self.context.gathered_data, default=str),
            recent_history=self.context.get_history_text(last_n=4),
        )
        response = self._call_gemini_speech(prompt)

        self.context.log_decision(
            observation=f"Need to address gap: {gap.name}",
            reasoning=f"Unresolved gaps remaining: "
                      f"{[g.name for g in self.context.get_unresolved_gaps()]}",
            action=f"Asking question about {gap.name}",
        )
        return response

    def _generate_clarification(self) -> str:
        """Generate a follow-up clarification question."""
        last_obs = self.context.observations[-1]
        prompt = (
            f"The patient just said something unclear or contradictory about their "
            f"smoking history. Their response was: \"{last_obs.raw_transcript}\"\n"
            f"The issue is: {last_obs.followup_reason}\n"
            f"Recent conversation:\n{self.context.get_history_text(last_n=4)}\n\n"
            f"Generate a gentle, clarifying follow-up question. "
            f"1-2 sentences maximum. Speak naturally."
        )
        response = self._call_gemini_speech(prompt)

        self.context.log_decision(
            observation=f"Last response was unclear (confidence: {last_obs.confidence:.2f})",
            reasoning=f"Reason: {last_obs.followup_reason}",
            action="Asking clarification",
        )
        return response

    def _generate_sdm_transition(self) -> str:
        """Generate the shared decision-making transition."""
        eligibility = self.merger.check_uspstf_eligibility()
        if eligibility is True:
            status = "MAY"
        elif eligibility is False:
            status = "likely do NOT"
        else:
            status = "might"

        prompt = SDM_TRANSITION_PROMPT.format(
            eligibility_status=status,
            resolved_data=json.dumps(self.context.gathered_data, default=str),
            recent_history=self.context.get_history_text(last_n=4),
        )
        response = self._call_gemini_speech(prompt)

        self.context.log_decision(
            observation=f"Data gathering complete. Eligibility: {eligibility}",
            reasoning="Transitioning to shared decision-making",
            action="SDM transition",
        )

        # After SDM, move to summary
        self.context.phase = ConversationPhase.SUMMARY
        return response

    def _generate_summary(self) -> str:
        """Generate a summary for patient confirmation."""
        prompt = SUMMARY_PROMPT.format(
            resolved_data=json.dumps(self.context.gathered_data, default=str),
        )
        response = self._call_gemini_speech(prompt)

        self.context.log_decision(
            observation="Generating summary for patient confirmation",
            reasoning=f"Gathered data: {self.context.gathered_data}",
            action="Summary confirmation",
        )

        # After summary, move to goodbye
        self.context.phase = ConversationPhase.GOODBYE
        return response

    def _generate_goodbye(self) -> str:
        """Generate a goodbye message."""
        return (
            "Thank you so much for taking the time to speak with me today. "
            "The information you've shared is really helpful for our study. "
            "If your doctor thinks screening would be a good idea, their office "
            "will be in touch. Have a wonderful day!"
        )
