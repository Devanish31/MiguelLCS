"""Logic for analyzing and resolving conflicts in smoking history data."""
from __future__ import annotations
from typing import Optional
from datetime import date
from patient_data.models import (
    PatientProfile, SmokingRecord, SmokingStatus, ResolvedSmokingHistory,
)
from agent.conversation_state import DataGap


class SmokingConflictAnalyzer:
    """Analyzes a patient's records to identify specific conflicts."""

    def __init__(self, patient: PatientProfile):
        self.patient = patient
        self.records = patient.smoking_records

    def identify_data_gaps(self) -> list[DataGap]:
        """Identify all data gaps from the patient's records."""
        gaps: list[DataGap] = []

        # Check if there's any quantitative data at all
        py_vals = [r.pack_years for r in self.records if r.pack_years is not None]
        ppd_vals = [r.packs_per_day for r in self.records if r.packs_per_day is not None]
        yrs_vals = [r.years_smoked for r in self.records if r.years_smoked is not None]

        if not py_vals and not ppd_vals and not yrs_vals:
            gaps.append(DataGap.QUANTITATIVE_DATA_MISSING)
        else:
            # Check pack-years conflict
            if len(py_vals) >= 2 and max(py_vals) - min(py_vals) > 3:
                gaps.append(DataGap.PACK_YEARS_CONFLICT)
            # Check if packs per day is unknown
            if not ppd_vals:
                gaps.append(DataGap.PACKS_PER_DAY_UNKNOWN)
            # Check if years smoked is unknown
            if not yrs_vals:
                gaps.append(DataGap.YEARS_SMOKED_UNKNOWN)

        # Check smoking status conflicts
        statuses = set(r.status for r in self.records if r.status is not None)
        if SmokingStatus.CURRENT in statuses and SmokingStatus.FORMER in statuses:
            gaps.append(DataGap.CURRENT_VS_FORMER_AMBIGUOUS)
        elif len(statuses) > 1:
            gaps.append(DataGap.SMOKING_STATUS_CONFLICT)

        # Check for "occasional" / ambiguous language suggesting current-vs-former
        # confusion even if all records say FORMER
        occasional_keywords = ["occasional", "sometimes", "social", "rarely", "once in a while"]
        for r in self.records:
            if any(kw in r.raw_text.lower() for kw in occasional_keywords):
                if DataGap.CURRENT_VS_FORMER_AMBIGUOUS not in gaps:
                    gaps.append(DataGap.CURRENT_VS_FORMER_AMBIGUOUS)
                break

        # Check quit date issues
        has_former = any(r.status == SmokingStatus.FORMER for r in self.records)
        quit_dates = [r.quit_date for r in self.records if r.quit_date is not None]
        quit_years = [r.years_since_quit for r in self.records if r.years_since_quit is not None]

        if has_former:
            if not quit_dates and not quit_years:
                gaps.append(DataGap.QUIT_DATE_MISSING)
            elif len(quit_dates) >= 2:
                # Check if quit dates conflict
                years = set(d.year for d in quit_dates)
                if len(years) > 1:
                    gaps.append(DataGap.QUIT_DATE_CONFLICT)
            # If we have years_since_quit but no actual quit date, that's a gap
            if not quit_dates and quit_years:
                gaps.append(DataGap.QUIT_DATE_MISSING)

        return gaps

    def format_for_prompt(self) -> str:
        """Format all patient data for inclusion in a Gemini prompt."""
        lines = [
            f"Patient: {self.patient.name}, Age {self.patient.age}, {self.patient.sex}",
            f"PCP: {self.patient.primary_care_provider}",
            f"EHR Structured Status: {self.patient.structured_ehr_status.value if self.patient.structured_ehr_status else 'None'}",
            f"EHR Structured Pack-Years: {self.patient.structured_ehr_pack_years or 'None'}",
            "",
            "Clinical Notes:",
        ]
        for r in self.records:
            lines.append(f"\n--- {r.source} ---")
            lines.append(f"  Status: {r.status.value if r.status else 'Not recorded'}")
            lines.append(f"  Pack-years: {r.pack_years or 'Not recorded'}")
            lines.append(f"  Packs/day: {r.packs_per_day or 'Not recorded'}")
            lines.append(f"  Years smoked: {r.years_smoked or 'Not recorded'}")
            lines.append(f"  Quit date: {r.quit_date or 'Not recorded'}")
            lines.append(f"  Years since quit: {r.years_since_quit or 'Not recorded'}")
            lines.append(f"  Note excerpt: \"{r.raw_text}\"")
        return "\n".join(lines)

    def format_conflicts(self) -> str:
        """Format identified conflicts as human-readable text."""
        gaps = self.identify_data_gaps()
        if not gaps:
            return "No significant conflicts identified."
        descriptions = {
            DataGap.PACK_YEARS_CONFLICT: "Pack-years values differ across clinical notes",
            DataGap.SMOKING_STATUS_CONFLICT: "Smoking status is inconsistent across records",
            DataGap.QUIT_DATE_MISSING: "Patient is listed as former smoker but quit date is missing",
            DataGap.QUIT_DATE_CONFLICT: "Quit dates differ across clinical notes",
            DataGap.PACKS_PER_DAY_UNKNOWN: "Cigarettes/packs per day not documented",
            DataGap.YEARS_SMOKED_UNKNOWN: "Duration of smoking not documented",
            DataGap.CURRENT_VS_FORMER_AMBIGUOUS: "Some notes say current smoker, others say former",
            DataGap.QUANTITATIVE_DATA_MISSING: "No quantitative smoking data (pack-years, packs/day, years) in any record",
        }
        lines = []
        for gap in gaps:
            lines.append(f"- {descriptions.get(gap, gap.name)}")
        return "\n".join(lines)


class SmokingDataMerger:
    """Merges EHR data with patient-reported data to produce resolved history."""

    def __init__(self, patient: PatientProfile):
        self.patient = patient
        self.patient_reported: dict = {}

    def update_from_response(self, extracted_data: dict) -> None:
        """Integrate newly extracted data from a patient response."""
        for key, value in extracted_data.items():
            if value is not None and key != "other_info":
                self.patient_reported[key] = value

    def compute_pack_years(self) -> Optional[float]:
        """Calculate pack-years from best available data."""
        # Patient-reported pack-years take priority
        if "pack_years" in self.patient_reported and self.patient_reported["pack_years"] is not None:
            return float(self.patient_reported["pack_years"])
        # Try to compute from components
        ppd = self.patient_reported.get("packs_per_day")
        years = self.patient_reported.get("years_smoked")
        if ppd is not None and years is not None:
            return float(ppd) * float(years)
        return None

    def check_uspstf_eligibility(self) -> Optional[bool]:
        """Check if resolved data meets USPSTF criteria."""
        pack_years = self.compute_pack_years()
        age = self.patient.age
        status = self.patient_reported.get("current_status")
        years_since_quit = self.patient_reported.get("years_since_quit")

        # Need minimum data
        if pack_years is None:
            return None
        if not (50 <= age <= 80):
            return False
        if pack_years < 20:
            return False
        if status == "current":
            return True
        if status == "former" and years_since_quit is not None:
            return float(years_since_quit) <= 15
        # If former but no quit info, can't determine
        return None

    def generate_resolved_history(self) -> ResolvedSmokingHistory:
        """Produce the final resolved smoking history."""
        status_str = self.patient_reported.get("current_status")
        status_map = {
            "current": SmokingStatus.CURRENT,
            "former": SmokingStatus.FORMER,
            "never": SmokingStatus.NEVER,
        }
        status = status_map.get(status_str, SmokingStatus.UNKNOWN)

        pack_years = self.compute_pack_years()
        eligibility = self.check_uspstf_eligibility()

        # Compute confidence based on how much data we have
        data_points = sum(1 for v in self.patient_reported.values() if v is not None)
        confidence = min(data_points / 5.0, 1.0)  # 5 key data points = full confidence

        quit_date_str = self.patient_reported.get("quit_date")
        quit_date = None
        if quit_date_str:
            try:
                parts = str(quit_date_str).split("-")
                if len(parts) >= 2:
                    quit_date = date(int(parts[0]), int(parts[1]), 1)
                else:
                    quit_date = date(int(parts[0]), 1, 1)
            except (ValueError, IndexError):
                pass

        yrs_quit = self.patient_reported.get("years_since_quit")

        notes = []
        if pack_years is not None:
            notes.append(f"Pack-years resolved to {pack_years:.1f}")
        if status != SmokingStatus.UNKNOWN:
            notes.append(f"Status: {status.value}")
        if eligibility is True:
            notes.append("Patient meets USPSTF screening criteria")
        elif eligibility is False:
            notes.append("Patient does NOT meet USPSTF screening criteria")
        else:
            notes.append("Eligibility could not be fully determined")

        return ResolvedSmokingHistory(
            current_status=status,
            pack_years=pack_years,
            packs_per_day=self.patient_reported.get("packs_per_day"),
            years_smoked=self.patient_reported.get("years_smoked"),
            quit_date=quit_date,
            years_since_quit=float(yrs_quit) if yrs_quit is not None else None,
            confidence=confidence,
            meets_uspstf_criteria=eligibility,
            resolution_notes=". ".join(notes),
        )
