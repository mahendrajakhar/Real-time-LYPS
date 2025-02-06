import asyncio
import websockets
import numpy as np
import soundfile as sf
from inference import SadTalker  # ✅ Import now works

# Load SadTalker Model
sadtalker_model = SadTalker()


# Function to process audio and generate lip-synced video
def process_audio(audio_data):
    audio_path = 'temp_audio.wav'
    audio_array = np.frombuffer(audio_data, dtype=np.float32)

    # ✅ Check if audio is empty
    if audio_array.size == 0:
        raise ValueError("Received empty audio data.")

    # ✅ Ensure minimum audio length (for stable processing)
    min_duration = 0.5  # 0.5 seconds
    min_samples = int(min_duration * 16000)  # Assuming 16 kHz sample rate
    if len(audio_array) < min_samples:
        raise ValueError("Audio data too short for processing.")

    sf.write(audio_path, audio_array, samplerate=16000)
    output_video = sadtalker_model.infer(
        source_image='./examples/source_image/full_body_1.png',  # Example image
        driven_audio=audio_path
    )
    return output_video

# WebSocket handler
async def handler(websocket, path):
    print("Client connected")
    try:
        async for audio_data in websocket:
            video_data = process_audio(audio_data)
            await websocket.send(video_data)
    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected")

# ✅ Wrap server startup inside an async function
async def main():
    async with websockets.serve(handler, "localhost", 8765):
        print("Server started on ws://localhost:8765")
        await asyncio.Future()  # Run forever

# Explicitly run the event loop
if __name__ == "__main__":
    asyncio.run(main())