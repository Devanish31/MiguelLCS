"""Generates the final structured output after the conversation."""
from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path
from agent.agent_core import AgentCore


class SummaryGenerator:
    """Produces the final structured JSON summary."""

    def __init__(self, output_dir: str = "output_logs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def generate_and_save(self, agent: AgentCore) -> str:
        """Generate the complete output and save to file."""
        output = agent.get_final_output()
        output["generated_at"] = datetime.now().isoformat()

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"summary_{agent.patient.patient_id}_{ts}.json"
        filepath = self.output_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, default=str)

        self._print_summary(output)
        return str(filepath)

    def _print_summary(self, output: dict) -> None:
        """Print a human-readable summary to the console."""
        h = output["resolved_smoking_history"]
        conf = output.get("confidence", 0)

        print("\n" + "=" * 60)
        print("  CALL SUMMARY")
        print("=" * 60)
        print(f"  Patient:            {output['patient_name']}")
        print(f"  Smoking Status:     {h['current_status']}")
        print(f"  Pack-years:         {h['pack_years']}")
        print(f"  Packs/day:          {h.get('packs_per_day', 'N/A')}")
        print(f"  Years smoked:       {h.get('years_smoked', 'N/A')}")
        print(f"  Quit date:          {h['quit_date'] or 'N/A'}")
        print(f"  Years since quit:   {h.get('years_since_quit', 'N/A')}")
        print(f"  USPSTF Eligible:    {h['meets_uspstf_criteria']}")
        print(f"  Confidence:         {conf:.0%}")
        print(f"  Total turns:        {output['total_turns']}")
        print(f"  Resolution:         {output.get('resolution_notes', '')}")
        print("=" * 60)
