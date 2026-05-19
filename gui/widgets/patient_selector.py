"""Patient selector dropdown with conflict info display."""
from __future__ import annotations
from typing import Callable
import customtkinter as ctk
from patient_data.mock_patients import get_all_patients
from patient_data.models import PatientProfile
from gui.themes import (
    DARK_CARD, DARK_SURFACE, TEXT_PRIMARY, TEXT_SECONDARY, TEXT_MUTED,
    CARDINAL_RED, BTN_NEUTRAL_BG, BTN_NEUTRAL_HOVER,
    FONT_FAMILY, FONT_SIZE_BODY, FONT_SIZE_SMALL,
    FONT_SIZE_HEADING, PADDING, CORNER_RADIUS,
)
from gui.widgets.dropdown import CTkDropdown


class PatientSelector(ctk.CTkFrame):
    """Sidebar widget: patient dropdown + conflict info."""

    def __init__(
        self,
        parent,
        on_patient_selected: Callable[[PatientProfile], None] | None = None,
        **kwargs,
    ):
        super().__init__(parent, fg_color="transparent", **kwargs)
        self._on_patient_selected = on_patient_selected
        self._patients = get_all_patients()
        self._selected: PatientProfile | None = None

        # Section header
        ctk.CTkLabel(
            self,
            text="Patient",
            font=(FONT_FAMILY, FONT_SIZE_HEADING, "bold"),
            text_color=TEXT_PRIMARY,
            anchor="w",
        ).pack(fill="x", padx=PADDING, pady=(PADDING, 4))

        # Dropdown
        patient_names = [f"{p.name} ({p.age}{p.sex[0]})" for p in self._patients]
        self._dropdown = CTkDropdown(
            self,
            values=patient_names,
            font=(FONT_FAMILY, FONT_SIZE_BODY),
            fg_color=DARK_CARD,
            button_color=BTN_NEUTRAL_BG,
            button_hover_color=BTN_NEUTRAL_HOVER,
            dropdown_fg_color=DARK_CARD,
            dropdown_hover_color=BTN_NEUTRAL_BG,
            corner_radius=CORNER_RADIUS,
            command=self._on_dropdown_change,
        )
        self._dropdown.pack(fill="x", padx=PADDING, pady=(0, 6))
        self._dropdown.set("Select patient...")

        # Conflict info area — created but NOT packed until a patient is picked.
        self._info_frame = ctk.CTkFrame(self, fg_color=DARK_CARD, corner_radius=CORNER_RADIUS)
        self._info_label = ctk.CTkLabel(
            self._info_frame,
            text="",
            font=(FONT_FAMILY, FONT_SIZE_SMALL),
            text_color=TEXT_SECONDARY,
            justify="left",
            anchor="nw",
            wraplength=260,
        )
        self._info_label.pack(fill="x", padx=PADDING, pady=PADDING)

    def _on_dropdown_change(self, choice: str) -> None:
        for p in self._patients:
            label = f"{p.name} ({p.age}{p.sex[0]})"
            if label == choice:
                self._selected = p
                self._update_info(p)
                if self._on_patient_selected:
                    self._on_patient_selected(p)
                break

    def _update_info(self, patient: PatientProfile) -> None:
        summary = patient.get_conflicts_summary()
        records_count = len(patient.smoking_records)
        ehr_status = patient.structured_ehr_status.value if patient.structured_ehr_status else "N/A"
        ehr_py = patient.structured_ehr_pack_years if patient.structured_ehr_pack_years else "N/A"

        info_text = (
            f"ID: {patient.patient_id}  |  PCP: {patient.primary_care_provider}\n"
            f"EHR Status: {ehr_status}  |  Pack-years: {ehr_py}\n"
            f"Clinical Notes: {records_count}\n\n"
            f"Conflicts:\n{summary}"
        )
        self._info_label.configure(text=info_text, text_color=TEXT_SECONDARY)
        # Only show the info card after a patient has been selected.
        if not self._info_frame.winfo_ismapped():
            self._info_frame.pack(fill="x", padx=PADDING, pady=(0, PADDING))

    @property
    def selected_patient(self) -> PatientProfile | None:
        return self._selected

    def set_enabled(self, enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        self._dropdown.configure(state=state)
