import threading
import time

import matplotlib.pyplot as plt
import numpy as np
# Tobii related imports
import tobii_research as tr
from flask_socketio import emit
from tqdm import tqdm

from app import app, socketio

time.clock = time.time

# --- TOBII GAZE STREAMING + CALIBRATION ---

# Real-time streaming logic
def stream_gaze_data():
    # Get the Tobii Eye Tracker device
    eye_tracker = tr.find_all_eyetrackers()[0]
    def gaze_callback(gaze_data):
        if gaze_data.get("left_gaze_point_on_display_area"):
            x, y = gaze_data["left_gaze_point_on_display_area"]
            if 0 <= x <= 1 and 0 <= y <= 1:
                socketio.emit("gaze_data", {"x": x, "y": y})
    
    eye_tracker.subscribe_to(tr.EYETRACKER_GAZE_DATA, gaze_callback, as_dictionary=True)

@socketio.on("start_gaze")
def start_gaze():
    threading.Thread(target=stream_gaze_data).start()
    emit("gaze_status", {"status": "started"})
