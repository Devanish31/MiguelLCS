"""Desktop-side control of the Twilio phone transport."""
from __future__ import annotations
import threading
import socket
import uvicorn
from twilio.rest import Client
from urllib.parse import urlencode



class PhoneCallController:
    def __init__(self):
        self.call_sid: str | None = None
        self._server_started = False

    def ensure_server(self) -> None:
        if self._server_started:
            return
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex(("127.0.0.1", 8765)) == 0:
                self._server_started = True
                return
        config = uvicorn.Config(
            "phone.server:app",
            host="127.0.0.1",
            port=8765,
            log_level="warning",
            log_config=None,
        )
        server = uvicorn.Server(config)
        threading.Thread(target=server.run, daemon=True).start()
        self._server_started = True

    def start_call(self, to_number: str, patient_id: str,
                   voice_name: str) -> str:
        from config import TwilioConfig
        self.cfg = TwilioConfig()
        if not self.cfg.is_configured:
            raise RuntimeError("Twilio is not fully configured in .env")
        self.ensure_server()
        client = Client(self.cfg.account_sid, self.cfg.auth_token)
        voice_url = (
            f"{self.cfg.public_base_url.rstrip('/')}/twilio/voice?"
            + urlencode({"patient_id": patient_id, "voice_name": voice_name})
        )
        call = client.calls.create(
            to=to_number,
            from_=self.cfg.phone_number,
            url=voice_url,
            method="POST",
        )
        self.call_sid = call.sid
        return call.sid

    def end_call(self) -> None:
        if not self.call_sid:
            return
        from config import TwilioConfig
        self.cfg = TwilioConfig()
        client = Client(self.cfg.account_sid, self.cfg.auth_token)
        client.calls(self.call_sid).update(status="completed")
        self.call_sid = None
