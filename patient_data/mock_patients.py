"""5 synthetic patients with realistic conflicting smoking history data."""
from datetime import date
from patient_data.models import PatientProfile, SmokingRecord, SmokingStatus


def _build_patient_1() -> PatientProfile:
    """Maria Santos - Pack-years discordance (20 vs 30 across notes)."""
    return PatientProfile(
        patient_id="P001",
        name="Maria Santos",
        age=62,
        sex="Female",
        primary_care_provider="Dr. Thompson",
        structured_ehr_status=SmokingStatus.CURRENT,
        structured_ehr_pack_years=25.0,
        smoking_records=[
            SmokingRecord(
                source="Progress Note 2021-06-10",
                date_recorded=date(2021, 6, 10),
                status=SmokingStatus.CURRENT,
                pack_years=20.0,
                packs_per_day=1.0,
                years_smoked=20.0,
                raw_text="Patient reports smoking approximately 1 pack per day for 20 years. "
                         "Currently smoking. Counseled on cessation.",
            ),
            SmokingRecord(
                source="Annual Physical 2022-09-14",
                date_recorded=date(2022, 9, 14),
                status=SmokingStatus.CURRENT,
                pack_years=30.0,
                packs_per_day=1.5,
                years_smoked=20.0,
                raw_text="Smoking history: 1.5 PPD x 20 years = 30 pack-years. "
                         "Active smoker. Declined NRT.",
            ),
            SmokingRecord(
                source="Pulmonology Consult 2023-03-22",
                date_recorded=date(2023, 3, 22),
                status=SmokingStatus.CURRENT,
                pack_years=25.0,
                packs_per_day=None,
                years_smoked=None,
                raw_text="Approximately 25 pack-year smoking history. Continues to smoke. "
                         "Discussed lung cancer screening.",
            ),
            SmokingRecord(
                source="PCP Visit 2024-01-08",
                date_recorded=date(2024, 1, 8),
                status=SmokingStatus.CURRENT,
                pack_years=None,
                packs_per_day=1.0,
                years_smoked=25.0,
                raw_text="Long-time smoker, about a pack a day for 25 years. "
                         "Still smoking. Referral to cessation program.",
            ),
        ],
    )


def _build_patient_2() -> PatientProfile:
    """James Whitfield - Missing quit date + occasional cigarette ambiguity."""
    return PatientProfile(
        patient_id="P002",
        name="James Whitfield",
        age=58,
        sex="Male",
        primary_care_provider="Dr. Patel",
        structured_ehr_status=SmokingStatus.FORMER,
        structured_ehr_pack_years=35.0,
        smoking_records=[
            SmokingRecord(
                source="Progress Note 2020-04-15",
                date_recorded=date(2020, 4, 15),
                status=SmokingStatus.FORMER,
                pack_years=35.0,
                packs_per_day=1.5,
                years_smoked=23.0,
                years_since_quit=2.0,
                raw_text="Former smoker. 1.5 PPD for approximately 23 years. "
                         "Reports quitting about 2 years ago.",
            ),
            SmokingRecord(
                source="Cardiology Note 2022-08-20",
                date_recorded=date(2022, 8, 20),
                status=SmokingStatus.FORMER,
                pack_years=None,
                quit_date=None,
                raw_text="Former smoker, quit date unknown. Significant smoking history. "
                         "Cardiac risk factor noted.",
            ),
            SmokingRecord(
                source="PCP Visit 2024-02-10",
                date_recorded=date(2024, 2, 10),
                status=SmokingStatus.FORMER,
                pack_years=None,
                raw_text="Patient states he quit smoking but admits to having "
                         "an occasional cigarette with friends on weekends. "
                         "Maybe 2-3 per month. Counseled on complete cessation.",
            ),
        ],
    )


def _build_patient_3() -> PatientProfile:
    """Linda Chen - Status timeline inconsistency (never smoker doc error)."""
    return PatientProfile(
        patient_id="P003",
        name="Linda Chen",
        age=67,
        sex="Female",
        primary_care_provider="Dr. Rodriguez",
        structured_ehr_status=SmokingStatus.NEVER,
        structured_ehr_pack_years=None,
        smoking_records=[
            SmokingRecord(
                source="Progress Note 2019-05-20",
                date_recorded=date(2019, 5, 20),
                status=SmokingStatus.CURRENT,
                packs_per_day=1.0,
                years_smoked=15.0,
                pack_years=15.0,
                raw_text="Current smoker, approximately 1 pack per day for 15 years. "
                         "Started in her late 30s. Discussed risks.",
            ),
            SmokingRecord(
                source="ED Visit 2021-01-12",
                date_recorded=date(2021, 1, 12),
                status=SmokingStatus.NEVER,
                raw_text="Non-smoker. No tobacco use.",
            ),
            SmokingRecord(
                source="Progress Note 2023-07-05",
                date_recorded=date(2023, 7, 5),
                status=SmokingStatus.FORMER,
                pack_years=None,
                quit_date=date(2022, 1, 1),
                raw_text="Former smoker. Patient reports quitting in early 2022. "
                         "No current tobacco use. Great progress.",
            ),
            SmokingRecord(
                source="PCP Visit 2024-11-18",
                date_recorded=date(2024, 11, 18),
                status=SmokingStatus.FORMER,
                pack_years=None,
                years_since_quit=3.0,
                raw_text="Former smoker, quit approximately 3 years ago. "
                         "Doing well with cessation.",
            ),
        ],
    )


def _build_patient_4() -> PatientProfile:
    """Robert Johnson - Near-threshold uncertainty (18 vs 22 pack-years)."""
    return PatientProfile(
        patient_id="P004",
        name="Robert Johnson",
        age=55,
        sex="Male",
        primary_care_provider="Dr. Kim",
        structured_ehr_status=SmokingStatus.FORMER,
        structured_ehr_pack_years=20.0,
        smoking_records=[
            SmokingRecord(
                source="Progress Note 2020-03-10",
                date_recorded=date(2020, 3, 10),
                status=SmokingStatus.FORMER,
                pack_years=18.0,
                packs_per_day=0.75,
                years_smoked=24.0,
                quit_date=date(2012, 6, 1),
                raw_text="Former smoker. About 3/4 pack per day for 24 years. "
                         "Quit June 2012. 18 pack-year history.",
            ),
            SmokingRecord(
                source="Annual Physical 2022-05-22",
                date_recorded=date(2022, 5, 22),
                status=SmokingStatus.FORMER,
                pack_years=22.0,
                packs_per_day=1.0,
                years_smoked=22.0,
                quit_date=date(2011, 1, 1),
                raw_text="Former smoker, 1 PPD x 22 years = 22 pack-years. "
                         "Quit in 2011. Doing well.",
            ),
            SmokingRecord(
                source="PCP Visit 2024-09-05",
                date_recorded=date(2024, 9, 5),
                status=SmokingStatus.FORMER,
                pack_years=None,
                years_since_quit=None,
                raw_text="Former smoker with approximately 20 pack-year history. "
                         "Quit over a decade ago. Exact quit date unclear.",
            ),
        ],
    )


def _build_patient_5() -> PatientProfile:
    """Patricia Williams - Completely missing quantitative data."""
    return PatientProfile(
        patient_id="P005",
        name="Patricia Williams",
        age=71,
        sex="Female",
        primary_care_provider="Dr. Nguyen",
        structured_ehr_status=SmokingStatus.FORMER,
        structured_ehr_pack_years=None,
        smoking_records=[
            SmokingRecord(
                source="New Patient Intake 2019-02-14",
                date_recorded=date(2019, 2, 14),
                status=SmokingStatus.FORMER,
                raw_text="Former smoker. No further details documented.",
            ),
            SmokingRecord(
                source="Progress Note 2021-06-30",
                date_recorded=date(2021, 6, 30),
                status=SmokingStatus.FORMER,
                raw_text="History of heavy smoking for many years. "
                         "Quit some time ago. Patient did not recall specifics.",
            ),
            SmokingRecord(
                source="Pulmonology Referral 2023-10-10",
                date_recorded=date(2023, 10, 10),
                status=SmokingStatus.FORMER,
                raw_text="Significant smoking history per patient report. "
                         "Duration and quantity not quantified. "
                         "Recommend detailed smoking history assessment.",
            ),
        ],
    )


def get_all_patients() -> list[PatientProfile]:
    """Return the demo patient roster (Linda, Robert, Patricia)."""
    return [
        _build_patient_3(),  # Linda Chen (F, 67)
        _build_patient_4(),  # Robert Johnson (M, 55)
        _build_patient_5(),  # Patricia Williams (F, 71)
    ]


def get_patient_by_id(patient_id: str) -> PatientProfile:
    """Get a specific patient by ID."""
    for p in get_all_patients():
        if p.patient_id == patient_id:
            return p
    raise ValueError(f"Patient {patient_id} not found")
