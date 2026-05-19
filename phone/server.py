"""FastAPI service Twilio calls for real-phone conversations."""
from __future__ import annotations
from fastapi import FastAPI, WebSocket
from fastapi.responses import Response

from phone.session import TwilioPhoneSession
from phone.debug_log import log

app = FastAPI()


def _ws_base(url: str) -> str:
    return url.replace("https://", "wss://").replace("http://", "ws://").rstrip("/")


@app.post("/twilio/voice")
async def twilio_voice(patient_id: str, voice_name: str = "Jenny (Female, US)"):
    from config import TwilioConfig
    cfg = TwilioConfig()
    stream_url = f"{_ws_base(cfg.public_base_url)}/twilio/media"
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        "<Response>\n"
        "  <Connect>\n"
        f'    <Stream url="{stream_url}">\n'
        f'      <Parameter name="patient_id" value="{patient_id}" />\n'
        f'      <Parameter name="voice_name" value="{voice_name}" />\n'
        "    </Stream>\n"
        "  </Connect>\n"
        "</Response>"
    )
    return Response(content=xml, media_type="application/xml")


@app.websocket("/twilio/media")
async def twilio_media(websocket: WebSocket):
    log("websocket accepted")
    await websocket.accept()
    session = TwilioPhoneSession(websocket)
    try:
        await session.run()
    except Exception as e:
        log(f"session exception: {type(e).__name__}: {e}")
        raise
