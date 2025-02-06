import asyncio
import websockets
import sounddevice as sd
import numpy as np

# Audio settings
SAMPLERATE = 16000
CHUNK_DURATION = 0.5  # Send audio in 0.5-second chunks (adjust if needed)
CHUNK_SIZE = int(SAMPLERATE * CHUNK_DURATION)

async def send_audio():
    uri = "ws://localhost:8765"
    loop = asyncio.get_running_loop()

    async with websockets.connect(uri) as websocket:
        print("Connected to server")

        def callback(indata, frames, time, status):
            if status:
                print(status)
            audio_chunk = indata.tobytes()

            # ✅ Send only if the audio chunk is not empty
            if np.any(indata):
                asyncio.run_coroutine_threadsafe(websocket.send(audio_chunk), loop)

        # ✅ Increase buffer size for more stable audio capture
        with sd.InputStream(samplerate=SAMPLERATE, channels=1, callback=callback, blocksize=CHUNK_SIZE):
            await asyncio.Future()  # Keep the connection open

if __name__ == "__main__":
    asyncio.run(send_audio())