"""Data models for patient records and smoking history."""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from typing import Optional


class SmokingStatus(Enum):
    CURRENT = "current_smoker"
    FORMER = "former_smoker"
    NEVER = "never_smoker"
    UNKNOWN = "unknown"


@dataclass
class SmokingRecord:
    """A single smoking history entry from one clinical note."""
    source: str                            # e.g., "Progress Note 2023-03-15"
    date_recorded: date
    status: Optional[SmokingStatus] = None
    pack_years: Optional[float] = None
    packs_per_day: Optional[float] = None
    years_smoked: Optional[float] = None
    quit_date: Optional[date] = None
    years_since_quit: Optional[float] = None
    raw_text: str = ""                     # The original note excerpt


@dataclass
class PatientProfile:
    """Complete patient profile with all conflicting smoking records."""
    patient_id: str
    name: str
    age: int
    sex: str
    primary_care_provider: str
    smoking_records: list[SmokingRecord] = field(default_factory=list)
    structured_ehr_status: Optional[SmokingStatus] = None
    structured_ehr_pack_years: Optional[float] = None

    def get_conflicts_summary(self) -> str:
        """Return a human-readable summary of conflicts for display."""
        conflicts = []
        # Check pack-years
        py_vals = [r.pack_years for r in self.smoking_records if r.pack_years is not None]
        if len(py_vals) >= 2 and max(py_vals) - min(py_vals) > 3:
            conflicts.append(f"Pack-years range: {min(py_vals)}-{max(py_vals)}")
        # Check status
        statuses = set(r.status for r in self.smoking_records if r.status is not None)
        if len(statuses) > 1:
            conflicts.append(f"Status conflict: {', '.join(s.value for s in statuses)}")
        # Check quit date
        quit_dates = [r.quit_date for r in self.smoking_records if r.quit_date is not None]
        quit_years_since = [r.years_since_quit for r in self.smoking_records if r.years_since_quit is not None]
        if not quit_dates and not quit_years_since:
            has_former = any(r.status == SmokingStatus.FORMER for r in self.smoking_records)
            if has_former:
                conflicts.append("Quit date: MISSING")
        # Check if quantitative data is missing
        if not py_vals:
            ppd_vals = [r.packs_per_day for r in self.smoking_records if r.packs_per_day is not None]
            yrs_vals = [r.years_smoked for r in self.smoking_records if r.years_smoked is not None]
            if not ppd_vals and not yrs_vals:
                conflicts.append("Quantitative data: MISSING")
        if not conflicts:
            conflicts.append("Minor inconsistencies")
        return "; ".join(conflicts)


@dataclass
class ResolvedSmokingHistory:
    """The output after the agent resolves conflicts via patient conversation."""
    current_status: SmokingStatus
    pack_years: Optional[float] = None
    packs_per_day: Optional[float] = None
    years_smoked: Optional[float] = None
    quit_date: Optional[date] = None
    years_since_quit: Optional[float] = None
    confidence: float = 0.0
    meets_uspstf_criteria: Optional[bool] = None
    resolution_notes: str = ""
