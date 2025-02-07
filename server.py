import asyncio
import websockets
import numpy as np
import soundfile as sf
from inference import SadTalker
import logging
import socket

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Load SadTalker Model
sadtalker_model = SadTalker()

# Function to process audio and generate lip-synced video
def process_audio(audio_data):
    audio_path = 'temp_audio.wav'
    audio_array = np.frombuffer(audio_data, dtype=np.float32)

    # Check if audio is empty
    if audio_array.size == 0:
        raise ValueError("Received empty audio data.")

    # Ensure minimum audio length (for stable processing)
    min_duration = 0.5  # 0.5 seconds
    min_samples = int(min_duration * 16000)  # Assuming 16 kHz sample rate
    if len(audio_array) < min_samples:
        raise ValueError("Audio data too short for processing.")

    sf.write(audio_path, audio_array, samplerate=16000)
    output_video = sadtalker_model.infer(
        source_image='./examples/source_image/full_body_1.png',
        driven_audio=audio_path
    )
    return output_video

# WebSocket handler
async def handler(websocket, path):
    logger.info("Client connected")
    try:
        async for audio_data in websocket:
            video_data = process_audio(audio_data)
            await websocket.send(video_data)
    except websockets.exceptions.ConnectionClosed:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"Error in handler: {e}")

async def main():
    # Disable IPv6
    socket.setdefaulttimeout(60)
    server = await websockets.serve(
        handler,
        "0.0.0.0",  # Listen on all available interfaces
        8765,
        reuse_port=True
    )
    
    logger.info("Server started on ws://0.0.0.0:8765")
    await server.wait_closed()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server shutting down...")
    except Exception as e:
        logger.error(f"Server error: {e}")