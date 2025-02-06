import cv2
import numpy as np
from flask import Flask, Response, render_template
import threading
import pyaudio
import wave
import time
from queue import Queue
import os

app = Flask(__name__)

# Global variables
audio_queue = Queue()
frame_queue = Queue(maxsize=10)
is_recording = False

# Audio recording configuration
CHUNK = 1024
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 44100

class AudioHandler:
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.stream = None
        
    def start_recording(self):
        self.stream = self.p.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK
        )
        
    def stop_recording(self):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            
    def get_audio_data(self):
        return self.stream.read(CHUNK)

def generate_frames():
    # For demo, using a sample video file
    # Replace this with your actual lip-sync video generation logic
    video_path = "/Users/indianrenters/SadTalker/examples/ref_video/WDA_AlexandriaOcasioCortez_000.mp4"  # Update with your video path
    cap = cv2.VideoCapture(video_path)
    
    while True:
        success, frame = cap.read()
        if not success:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Loop video
            continue
            
        # Convert frame to jpg format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

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

@app.route('/start_stream')
def start_stream():
    global is_recording
    is_recording = True
    threading.Thread(target=start_audio_recording).start()
    return "Started streaming"

@app.route('/stop_stream')
def stop_stream():
    global is_recording
    is_recording = False
    return "Stopped streaming"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)