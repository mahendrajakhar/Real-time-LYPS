import cv2
import numpy as np
from flask import Flask, Response, render_template
import threading
import pyaudio
import wave
import time
from queue import Queue
import os
import asyncio
import websockets
from threading import Lock
import logging

# Configure logging and suppress ALSA errors completely
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logging.getLogger('alsa').setLevel(logging.CRITICAL)

# Create null ALSA configuration
with open('/etc/asound.conf', 'w') as f:
    f.write("""
pcm.!default {
    type null
}
ctl.!default {
    type null
}
""")

app = Flask(__name__)
# Suppress Flask development server logs
logging.getLogger('werkzeug').setLevel(logging.ERROR)

# Global variables
audio_queue = Queue()
frame_queue = Queue(maxsize=10)
is_recording = False
current_websocket = None
websocket_lock = Lock()

# Audio recording configuration
CHUNK = 1024
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 44100

class AudioHandler:
    def __init__(self):
        self.p = None
        self.stream = None
        self.is_virtual = True
        self._recording_started = False  # Add flag to track if we've logged start
        
    def start_recording(self):
        try:
            if not self.is_virtual:
                self.p = pyaudio.PyAudio()
                self.stream = self.p.open(
                    format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK
                )
            # Log only once when recording starts
            if not self._recording_started:
                logging.info("Audio recording started (virtual mode)")
                self._recording_started = True
        except Exception as e:
            logging.error(f"Audio initialization error: {e}")
            self.is_virtual = True
            
    def stop_recording(self):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        if self.p:
            self.p.terminate()
        self._recording_started = False  # Reset flag when stopping
            
    def get_audio_data(self):
        if self.is_virtual:
            # Generate synthetic audio data
            t = np.linspace(0, CHUNK/RATE, CHUNK)
            data = 0.5 * np.sin(2*np.pi*440*t)  # 440 Hz sine wave
            return data.astype(np.float32).tobytes()
        else:
            return self.stream.read(CHUNK, exception_on_overflow=False)

def generate_frames():
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        else:
            # Return a blank frame if queue is empty
            blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            ret, buffer = cv2.imencode('.jpg', blank_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.03)  # 30 FPS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

def start_audio_recording():
    global is_recording
    audio_handler = AudioHandler()
    
    while is_recording:
        if not audio_handler.stream:
            audio_handler.start_recording()
            
        audio_data = audio_handler.get_audio_data()
        audio_queue.put(audio_data)
        
    audio_handler.stop_recording()

def run_async_websocket():
    """Run websocket client in a new event loop"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(websocket_client())
    loop.close()

@app.route('/start_stream')
def start_stream():
    global is_recording
    is_recording = True
    threading.Thread(target=start_audio_recording).start()
    threading.Thread(target=run_async_websocket).start()
    return "Started streaming"

@app.route('/stop_stream')
def stop_stream():
    global is_recording
    is_recording = False
    return "Stopped streaming"

async def websocket_client():
    global current_websocket
    retry_count = 0
    max_retries = 3
    
    while retry_count < max_retries:
        try:
            async with websockets.connect(
                'ws://127.0.0.1:8765',
                ping_interval=None,  # Disable ping/pong
                max_size=None  # No size limit for messages
            ) as websocket:
                with websocket_lock:
                    current_websocket = websocket
                logging.info("WebSocket connected")
                
                response_task = asyncio.create_task(
                    process_server_response(websocket, frame_queue)
                )
                audio_task = asyncio.create_task(
                    process_audio_queue(websocket)
                )
                
                try:
                    await asyncio.gather(response_task, audio_task)
                except Exception as e:
                    logging.error(f"WebSocket error: {e}")
                finally:
                    response_task.cancel()
                    audio_task.cancel()
                    
                # If we get here without errors, break the retry loop
                break
                
        except Exception as e:
            retry_count += 1
            if retry_count < max_retries:
                logging.error(f"Connection error (attempt {retry_count}): {e}")
                await asyncio.sleep(2)  # Wait before retrying
            else:
                logging.error(f"Failed to connect after {max_retries} attempts")
        finally:
            with websocket_lock:
                current_websocket = None

async def process_server_response(websocket, frame_queue):
    video_writer = None
    try:
        async for message in websocket:
            try:
                frame_data = np.frombuffer(message, dtype=np.uint8)
                frame = cv2.imdecode(frame_data, cv2.IMREAD_COLOR)
                
                if frame is None:
                    continue
                    
                if not frame_queue.full():
                    frame_queue.put(frame)
                    
                if video_writer is None:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    output_path = f'outputs/output_{timestamp}.mp4'
                    video_writer = cv2.VideoWriter(
                        output_path, 
                        fourcc, 
                        30.0,
                        (frame.shape[1], frame.shape[0])
                    )
                    logging.info(f"Recording started: {output_path}")
                
                video_writer.write(frame)
                
            except Exception as e:
                logging.error(f"Frame processing error: {e}")
                continue
                
    except Exception as e:
        logging.error(f"Server response error: {e}")
    finally:
        if video_writer:
            video_writer.release()
            logging.info("Recording stopped")

async def process_audio_queue(websocket):
    try:
        while True:
            if not audio_queue.empty():
                audio_data = audio_queue.get()
                try:
                    await websocket.send(audio_data)
                except Exception as e:
                    logging.error(f"Error sending audio: {e}")
                    break
            await asyncio.sleep(0.01)
    except Exception as e:
        logging.error(f"Audio queue error: {e}")

if __name__ == '__main__':
    os.makedirs('outputs', exist_ok=True)
    logging.info("Starting Real-time Lip Sync server...")
    app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)