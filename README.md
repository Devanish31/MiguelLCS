# MiguelLCS — Local-First Voice & Note Agent for Lung Cancer Screening

A Stanford-research hackathon prototype built for the **Gemma 4 Good Hackathon**.
Two workflows, one desktop app, **all inference local**.

1. **Clinical Note Extraction** — Gemma reads years of fragmented EHR notes and
   produces structured smoking-history fields (status, pack-years, years since
   cessation) plus USPSTF screening eligibility.
2. **Voice Agent (SDM)** — Gemma calls eligible patients in English or Spanish,
   confirms smoking history, conducts the CMS-required Shared Decision-Making
   conversation, and returns a clinician-ready record.

No PHI ever leaves the clinic workstation. Twilio (optional) carries audio
between the patient phone and the workstation; transcription, reasoning, RAG
retrieval, and eligibility logic all run locally.

---

## Architecture (high level)

```
patient phone → Twilio Media Streams ──► local workstation ◄──► clinician

local workstation:
    Whisper STT  →  Orchestrator (Py state machine)  →  Edge TTS
                          ↕
                     Gemma 4 / 3 (Ollama)
                          ↕
                     RAG (SDM knowledge)
                          ↕
                  Smoking Resolver (USPSTF, Python)
```

A more detailed diagram lives at `voice_agent_architecture.png` in the
hackathon writeup attachments.

---

## Model stack

| Role | Model | Where |
|---|---|---|
| Voice-agent reasoning | **Gemma 4 E4B** (`gemma4:e4b`, ~9.6 GB) | local via Ollama |
| Batch note extraction | **Gemma 3 27B q4** (`gemma3:27b-it-q4_K_M`, ~17 GB) | local via Ollama |
| Speech-to-text | **OpenAI Whisper `small.en`** | local via HuggingFace transformers, CUDA fp16 |
| Text-to-speech | **Microsoft Edge TTS** — Jenny / Andrew (EN), Paloma / Alonso (ES) | local; first synth call goes to Edge's free streaming endpoint |
| RAG embeddings | **`sentence-transformers/all-MiniLM-L6-v2`** | local |

Models are **not** committed to this repo — they're pulled at install time
(see "Setup" below). Total disk footprint: ~30–35 GB for the full stack.

---

## Hardware target

- Single clinical workstation
- One consumer GPU (development target: RTX 5090 / 20 GB VRAM)
- Whisper and Edge TTS will run on CPU if no GPU is present, with a latency
  hit but no functional loss
- Ollama auto-detects the GPU and offloads as many layers as VRAM allows

---

## Setup

```bash
# 1. Install Ollama and pull the models
# https://ollama.com/download
ollama pull gemma4:e4b              # voice agent
ollama pull gemma3:27b-it-q4_K_M    # batch extraction (optional)

# 2. Python deps
python -m pip install -r requirements.txt
python -m pip install customtkinter edge-tts miniaudio pillow

# 3. (Optional) Twilio bridge — only needed for real phone calls
cp .env.example .env
# fill in TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER,
#         TWILIO_PUBLIC_BASE_URL

# 4. Launch the desktop app
python app.py
```

The first launch downloads the Whisper weights (~470 MB) into the
HuggingFace cache.

---

## Repo layout

```
app.py                      desktop launcher (CustomTkinter)
config.py                   model + audio + Twilio config (env-backed)
orchestrator.py             call orchestrator + latency timing
agent/                      AgentCore, decision_maker, prompt_templates
gui/                        CustomTkinter widgets (two tabs)
voice/                      AudioRecorder, Whisper STT
tts/                        Edge TTS, Chatterbox, Qwen3-TTS backends
phone/                      Twilio Media Streams adapter (production)
rag/                        Local SDM retrieval (sentence-transformers)
knowledge/sdm/              SDM knowledge documents (markdown)
patient_data/               Synthetic mock patient profiles
```

---

## Stack (one-liner)

Python + CustomTkinter desktop app · Ollama-served Gemma 4 / Gemma 3 ·
Whisper small.en · Edge TTS · sentence-transformers RAG · optional Twilio
Programmable Voice (Media Streams) telephony bridge ·
`sounddevice` + `miniaudio` for local audio I/O.

---

## License & ethics

Research prototype. No real patient data ships with this repo — the included
mock patient profiles are synthetic. The `.env` file (Twilio credentials,
local-only) is git-ignored. The synthetic mock patient data and SDM knowledge
documents are MIT-licensed.

Built for the Gemma 4 Good Hackathon (Kaggle, May 2026).
