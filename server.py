import asyncio
import websockets
import numpy as np
import soundfile as sf
from inference import SadTalker
import logging
import socket
import cv2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Load SadTalker Model
sadtalker_model = SadTalker()
logger.info("SadTalker model loaded successfully")

# Function to process audio and generate lip-synced video
async def process_audio(audio_data):
    try:
        audio_path = 'temp_audio.wav'
        audio_array = np.frombuffer(audio_data, dtype=np.float32)

        if audio_array.size == 0:
            return None

        min_duration = 0.5
        min_samples = int(min_duration * 16000)
        if len(audio_array) < min_samples:
            return None

        sf.write(audio_path, audio_array, samplerate=16000)
        output_video = sadtalker_model.infer(
            source_image='./examples/source_image/full_body_1.png',
            driven_audio=audio_path
        )
        
        # Read the generated video frame
        cap = cv2.VideoCapture(output_video)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            # Convert frame to bytes
            _, buffer = cv2.imencode('.jpg', frame)
            return buffer.tobytes()
        return None
        
    except Exception as e:
        logger.error(f"Error processing audio: {e}")
        return None

# WebSocket handler
async def handler(websocket):
    logger.info("Client connected")
    try:
        async for audio_data in websocket:
            try:
                frame_data = await process_audio(audio_data)
                if frame_data:
                    await websocket.send(frame_data)
            except Exception as e:
                logger.error(f"Error in processing: {e}")
                continue
    except websockets.exceptions.ConnectionClosed:
        logger.info("Client disconnected")
    except Exception as e:
        logger.error(f"Error in handler: {e}")

async def main():
    try:
        async with websockets.serve(
            handler,
            "0.0.0.0",
            8765,
            ping_interval=None,  # Disable ping/pong
            max_size=None  # No size limit for messages
        ) as server:
            logger.info("Server started on ws://0.0.0.0:8765")
            await asyncio.Future()  # Run forever
    except Exception as e:
        logger.error(f"Server error: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server shutting down...")
    except Exception as e:
        logger.error(f"Server error: {e}")