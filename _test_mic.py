"""Focused X15 Bluetooth mic test — 5 seconds, speak continuously."""
import sounddevice as sd
import numpy as np
import queue
import time

print("=== X15 BLUETOOTH MIC TEST ===")
print(f"Default input: [{sd.default.device[0]}] {sd.query_devices(sd.default.device[0])['name']}")
print()

# Use the Windows default (which is X15)
dev_id = sd.default.device[0]
sr = 48000

audio_q = queue.Queue()
def cb(indata, frames, time_info, status):
    audio_q.put(indata[:, 0].copy())

print(">>> SPEAK NOW into your X15 mic for 5 seconds... <<<")
print()

with sd.InputStream(samplerate=sr, channels=1, dtype="int16",
                    blocksize=int(sr * 0.5), device=dev_id, callback=cb):
    for i in range(10):  # 10 x 0.5s = 5 seconds
        time.sleep(0.5)
        chunks = []
        while not audio_q.empty():
            chunks.append(audio_q.get())
        if chunks:
            audio = np.concatenate(chunks)
            rms = np.sqrt(np.mean(audio.astype(np.float32) ** 2)) / 32768.0
            peak = np.max(np.abs(audio.astype(np.float32))) / 32768.0
            bar = "#" * int(min(rms * 5000, 50))
            print(f"  [{i+1}/10] RMS={rms:.6f}  Peak={peak:.6f}  {bar}")
        else:
            print(f"  [{i+1}/10] no data")

print()
print("If RMS stayed near 0.000020, your X15 mic is not transmitting audio to Windows.")
print("Check: Settings > Privacy > Microphone > Allow apps to access your microphone")
print("Also: Right-click speaker icon > Sound settings > Input > test volume bar")
