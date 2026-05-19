"""Quick test: Edge TTS synthesis + playback via miniaudio."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sounddevice as sd
default_out = sd.query_devices(sd.default.device[1])
print(f"Default output: {default_out['name']}")

print("Synthesizing speech with Edge TTS...")
from tts.edge_tts_backend import EdgeTTSBackend
backend = EdgeTTSBackend()
backend.initialize("Jenny (Female, US)")

print("Playing through speakers/headphones — you should hear this...")
backend.speak_blocking("Hello! This is a test of the Edge TTS neural voice. Can you hear me clearly through your Bluetooth headphones?")
print("Done! Did you hear it?")
