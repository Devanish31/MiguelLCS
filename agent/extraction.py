"""Single-shot smoking-history extraction.

Distilled from the batch scripts 07/08 in SmokingHistory_AgenticAI:
keeps the patient-level system prompt and the parsing logic, drops the
batch/checkpoint/resume machinery (the GUI does one note at a time).
"""
from __future__ import annotations
import re
import ollama


PATIENT_LEVEL_PROMPT = """
You are a researcher interested in monitoring smoking status within lung cancer patients. You are given multiple clinical notes for the same patient in chronological order, each prepended with its date. Your task is to extract the patient's smoking history by reasoning across all notes temporally.

IMPORTANT — Using note dates for calculations:
Each note is prepended with its date in [YYYY-MM-DD] format. Use the date of the MOST RECENT note as the reference date ("today") for any time-based calculations (e.g., years since cessation, inferring year of cessation from "quit X years ago"). Always anchor calculations to this most recent note date, not any other date.

When notes conflict:
- Prefer the most recent explicit mention for each field.
- For smoking_status: if the patient was ever Current and a later note indicates Former (with clear quit evidence MORE than 4 weeks before the most recent note date), classify as Former. If the most recent note indicates Current, classify as Current.
- For numeric fields (pack_years, smoking_intensity, duration): prefer the most recent explicitly stated value.
- For quit dates: if multiple quit dates are mentioned across notes, use the LATEST (most recent) quit date.

Never smoker override rule:
A patient with ANY concrete smoking evidence (pack-years, quit date, smoking intensity, specific smoking duration, or explicit mention of tobacco use) in ANY note CANNOT be classified as Never smoker, even if other notes state "never smoker" or "denies tobacco." In such cases, decide between Current and Former based on the rules below.

Quit < 4 weeks rule:
If a patient reports quitting smoking but the quit date is less than 4 weeks before the most recent note date, classify as Current smoker (not Former). Only classify as Former smoker if the quit occurred more than 4 weeks before the most recent note date.

The model should classify the smoking status into one of four categories: Current smoker, Former smoker, Never smoker, and Unknown (if no smoking information is present). If the patient is a current or former smoker, the model should also extract information on smoking duration, cessation status, age at cessation, year of cessation, years since cessation, pack years, and smoking intensity.

Age:
Extract the patient's current age in years if it is explicitly stated in any note (e.g., "62 y/o", "62 yo", "62 year-old", "age 62"). If multiple ages appear across notes, use the value from the most recent note. Report N/A if age is not stated anywhere.

Output all relevant smoking-related evidence at the beginning (include the note date with each piece of evidence), separated by a slash (/) if there are multiple excerpts. Then, determine all smoking details using temporal reasoning across all notes. Finally, output each element on a single line separated by semicolons in this format: Age: [N/A | (NUMBER)]; Evidence: [All smoking-related evidence with dates]; Smoking Status Category: [Current smoker | Former smoker | Never smoker | Unknown]; Reasoning: [reason for the classification, including how conflicts across notes were resolved]; Years of Tobacco Use: [N/A | (NUMBER or if RANGE is provided consider the middle of that range) years]; Cessation Status: [Yes | N/A]; Age at Cessation: [N/A | (NUMBER)]; Year of Cessation: [N/A | (YEAR or if RANGE is provided consider the middle of that range)]; Years Since Cessation: [N/A | (NUMBER or if RANGE is provided consider the middle of that range) years]; Smoking Intensity: [N/A | (NUMBER or if RANGE is provided consider the middle of that range) (UNIT)]; Pack Years: [N/A | (NUMBER or if RANGE is provided consider the middle of that range)].
""".strip()


def wrap_notes(notes_text: str, n_notes: int = 1) -> str:
    """Wrap user-supplied note text in the strict header the model expects."""
    strict_header = (
        f"YOU ARE PROVIDED EXACTLY {n_notes} CLINICAL NOTE(S) ENCLOSED IN <clinical_notes> TAGS BELOW.\n"
        f"YOU MUST REASON OVER ONLY THESE {n_notes} NOTE(S).\n"
        f"DO NOT FABRICATE, INVENT, OR ASSUME ANY NOTES, DATES, OR CLINICAL INFORMATION NOT EXPLICITLY PRESENT WITHIN <clinical_notes>.\n"
        f"EVERY DATE AND PIECE OF EVIDENCE IN YOUR OUTPUT MUST COME DIRECTLY FROM THE TEXT INSIDE <clinical_notes>.\n\n"
    )
    body = f"<clinical_notes>\n{notes_text.strip()}\n</clinical_notes>"
    return strict_header + "Clinical Notes:\n" + body


def run_extraction(
    notes_text: str,
    system_prompt: str,
    model: str = "gemma4:e4b",
    host: str = "http://127.0.0.1:11434",
    num_ctx: int = 8192,
    temperature: float = 0.1,
    n_notes: int = 1,
) -> str:
    """Run the extraction prompt against Gemma. Returns raw text output."""
    user_text = wrap_notes(notes_text, n_notes=n_notes)
    client = ollama.Client(host=host)
    response = client.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
        options={"temperature": temperature, "num_ctx": num_ctx, "num_gpu": 999},
        think=False,
    )
    return response["message"]["content"].strip()


# ---------------------------------------------------------------------------
# Parsing — distilled from 08_parse_patient_level.py
# ---------------------------------------------------------------------------

def _extract_value(text: str, field: str) -> str | None:
    if not isinstance(text, str):
        return None
    matches = re.findall(r'(?:^|;|\n)\s*' + re.escape(field) + r':\s*([^;\n]+)', text)
    if not matches:
        return None
    v = matches[-1].strip()
    return None if v in ("N/A", "") else v


def _process_age(x: str | None) -> int | None:
    """Parse the 'Age: ...' field returned by the model."""
    if not x:
        return None
    s = str(x).lower().strip()
    if s.startswith("n/a"):
        return None
    nums = [int(n) for n in re.findall(r"\d+", s)]
    if not nums:
        return None
    age = nums[0]
    return age if 10 <= age <= 120 else None


def _parse_status(text: str) -> str | None:
    s = _extract_value(text, "Smoking Status Category")
    if s is None:
        return None
    s = s.lower()
    if "current" in s:
        return "current"
    if "former" in s:
        return "former"
    if "never" in s:
        return "never"
    return None


def _process_duration(x: str | None) -> float | None:
    if not x:
        return None
    x = str(x).lower().strip()
    if x.startswith("n/a"):
        return None
    for w, n in [("decade", "10"), ("twenty", "20"), ("thirty", "30"),
                 ("forty", "40"), ("fifty", "50"), ("sixty", "60")]:
        x = re.sub(rf"\b{w}\b", n, x)
    nums = [float(n) for n in re.findall(r"\d+\.?\d*", x)]
    if not nums:
        for k, v in {"few months": 0.5, "few years": 3, "couple": 2,
                     "several years": 5, "decades": 30, "lifelong": 50}.items():
            if k in x:
                return v
        return None
    if len(nums) >= 2:
        return min(nums)
    if "month" in x:
        return nums[0] / 12
    if "week" in x:
        return nums[0] / 52
    return nums[0]


def _process_age_cessation(x: str | None) -> int | None:
    if not x:
        return None
    x = str(x).lower().strip()
    if x.startswith("n/a"):
        return None
    nums = [int(n) for n in re.findall(r"\d+", x)]
    if not nums:
        for pat, val in [("teenager", 17), ("20s", 25), ("30s", 35), ("40s", 45)]:
            if pat in x:
                return val
        return None
    return min(nums)


def _process_year_cessation(x: str | None, note_year: int | None = None) -> int | None:
    if not x:
        return None
    orig = str(x)
    s = orig.lower().strip()
    if s.startswith("n/a"):
        return None
    if note_year and "years ago" in s:
        n = re.search(r"(\d+)\+?\s*years ago", s)
        if n:
            return int(note_year) - int(n.group(1))
    y4 = re.search(r"\b(19|20)\d{2}\b", orig)
    if y4:
        return int(y4.group(0))
    return None


def _process_years_since(x: str | None) -> float | None:
    if not x:
        return None
    s = str(x).lower().strip()
    if s.startswith("n/a"):
        return None
    if "week" in s:
        n = re.search(r"\d+\.?\d*", s)
        return round(float(n.group()) / 52, 2) if n else 0.02
    if re.search(r"months?|mos", s):
        n = re.search(r"\d+\.?\d*", s)
        return round(float(n.group()) / 12, 2) if n else 0.08
    nums = [float(n) for n in re.findall(r"\d+\.?\d*", s)]
    if not nums:
        return None
    return min(nums) if len(nums) >= 2 else nums[0]


def _process_pack_years(x: str | None) -> float | None:
    if not x:
        return None
    s = str(x).lower().strip()
    if s.startswith("n/a"):
        return None
    nums = [float(n) for n in re.findall(r"\d+\.?\d*", s)]
    if not nums:
        return None
    if len(nums) >= 2 and "pack years" in s:
        return nums[0]
    return min(nums) if len(nums) >= 2 else nums[0]


def _process_intensity(x: str | None) -> float | None:
    if not x:
        return None
    orig = str(x)
    s = orig.lower().strip()
    if s.startswith("n/a"):
        return None
    for w, v in [("half", "0.5"), ("a pack", "1 pack"), ("one pack", "1 pack")]:
        s = s.replace(w, v)
    nums = [float(n) for n in re.findall(r"\.?\d+\.?\d*", s)]
    if re.search(r"pack|ppd", s):
        frac = re.search(r"(\d)/(\d)", orig)
        if frac:
            return round(int(frac.group(1)) / int(frac.group(2)), 2)
        if not nums:
            return None
        if "per week" in s:
            return round(nums[0] / 7, 2)
        if len(nums) >= 2:
            return min(nums)
        return None if nums[0] > 10 else nums[0]
    if re.search(r"cigarette|cig|cpd", s):
        if not nums:
            return None
        base = min(nums) if len(nums) >= 2 else nums[0]
        return round(base / 20, 2)
    return None


def _screening_eligibility(status: str | None,
                           pack_years: float | None,
                           years_since_cessation: float | None,
                           age: int | None = None) -> str:
    """USPSTF 2021 lung-cancer screening criteria:
      - age 50-80
      - pack-years >= 20
      - currently smoking, OR quit within 15 years.
    """
    if status == "never":
        return "Not eligible"
    if pack_years is None:
        return "Unknown (need pack-years)"
    if pack_years < 20:
        return "Not eligible"
    if age is not None and not (50 <= age <= 80):
        return "Not eligible"
    if status == "current":
        return "Unknown (need age)" if age is None else "Eligible"
    if status == "former":
        if years_since_cessation is None:
            return "Unknown (need quit date)"
        if years_since_cessation > 15:
            return "Not eligible"
        return "Unknown (need age)" if age is None else "Eligible"
    return "Unknown"


def extract_age_from_note(text: str) -> int | None:
    """Best-effort regex pull for patient age from free-text notes."""
    if not text:
        return None
    patterns = [
        r"\b(\d{2,3})\s*y\s*/\s*o\b",                       # "62 y/o"
        r"\b(\d{2,3})\s*[-–]?\s*y(?:ear)?[-\s]?o(?:ld)?\b", # "62 yo", "62-year-old"
        r"\b(\d{2,3})\s*yo\b",
        r"\bage\s*:?\s*(\d{2,3})\b",
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            age = int(m.group(1))
            if 10 <= age <= 120:
                return age
    return None


def parse_extraction(text: str, note_year: int | None = None,
                     age: int | None = None) -> dict:
    """Parse the raw model output into a structured dict.

    `age` argument: optional regex-extracted fallback. The model's own
    `Age:` field is preferred when present.
    """
    llm_age = _process_age(_extract_value(text, "Age"))
    final_age = llm_age if llm_age is not None else age
    parsed = {
        "age":                   final_age,
        "smoking_status":        _parse_status(text),
        "duration_years":        _process_duration(_extract_value(text, "Years of Tobacco Use")),
        "age_at_cessation":      _process_age_cessation(_extract_value(text, "Age at Cessation")),
        "year_of_cessation":     _process_year_cessation(_extract_value(text, "Year of Cessation"), note_year),
        "years_since_cessation": _process_years_since(_extract_value(text, "Years Since Cessation")),
        "pack_years":            _process_pack_years(_extract_value(text, "Pack Years")),
        "smoking_intensity_ppd": _process_intensity(_extract_value(text, "Smoking Intensity")),
    }
    if parsed["years_since_cessation"] is None:
        if parsed["year_of_cessation"] and note_year:
            parsed["years_since_cessation"] = int(note_year) - parsed["year_of_cessation"]
    parsed["screening_eligible"] = _screening_eligibility(
        parsed["smoking_status"],
        parsed["pack_years"],
        parsed["years_since_cessation"],
        age=final_age,
    )
    return parsed
