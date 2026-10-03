import os
import cv2
import numpy as np
import sqlite3
import requests
import threading
from datetime import datetime
from flask import Flask, render_template, request, jsonify, Response
from werkzeug.utils import secure_filename
from ultralytics import YOLO
from hybrid_inference import analyze_video

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- TELEGRAM CONFIGURATION ---
TELEGRAM_BOT_TOKEN = "8824571130:AAFeciMV57pR_dKcJxlFXOiugGMuQzwzP_8"
TELEGRAM_CHAT_ID = "970923708"

def send_telegram_alert(message):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            "chat_id": TELEGRAM_CHAT_ID,
            "text": message,
            "parse_mode": "Markdown"
        }
        requests.post(url, json=payload, timeout=5)
    except Exception as e:
        print(f"Failed to send Telegram alert: {e}")

# --- DATABASE INITIALIZATION ---
def init_db():
    """Creates a local SQLite database for event logging if it doesn't exist."""
    conn = sqlite3.connect('surveillance.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS security_events
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT,
                  event_type TEXT,
                  track_id INTEGER)''')
    conn.commit()
    conn.close()

init_db()

# Load the model once when the server boots for live tracking
yolo_tracker = YOLO("yolov8s.pt")

# --- MODE 1: FORENSIC FILE UPLOAD (Your Existing Logic) ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded.'})
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected.'})
        
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        result = analyze_video(filepath)
        os.remove(filepath)
        return jsonify(result)

# --- CAMERA STATE MANAGEMENT ---
camera_active = False

@app.route('/toggle_camera', methods=['POST'])
def toggle_camera():
    global camera_active
    camera_active = not camera_active
    return jsonify({"status": "success", "camera_active": camera_active})

# --- MODE 2: LIVE SURVEILLANCE FEED (New Logic) ---
def generate_live_frames():
    """Captures live webcam feed, runs YOLOv8 tracking, checks fence intrusions, and logs events."""
    global camera_active
    import time
    cap = None
    logged_intruders = set() # Tracks who is currently in the zone to prevent DB spam
    alarm_cooldown = 0 # Keeps alarm active for a few frames to prevent flickering
    
    try:
        while True:
            if not camera_active:
                if cap is not None:
                    cap.release()
                    cap = None
                
                # Generate an offline placeholder frame so the stream connection doesn't break
                black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(black_frame, "CAMERA OFFLINE", (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(black_frame, "Click 'Initialize Camera Feed' to start", (100, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
                ret, buffer = cv2.imencode('.jpg', black_frame)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                time.sleep(0.5)
                continue
                
            if cap is None:
                cap = cv2.VideoCapture(0)
                
            success, frame = cap.read()
            if not success:
                time.sleep(0.1)
                continue
                
            h, w = frame.shape[:2]
            
            # 1. Define the Virtual Fence
            fence = np.array([[int(w*0.5), 0], [w, 0], [w, h], [int(w*0.5), h]], np.int32)
            fence = fence.reshape((-1, 1, 2))
            
            # 2. Run YOLOv8 Tracking
            # conf=0.6: Balanced confidence to prevent ghosts but allow partially visible people
            # iou=0.45: Aggressive overlap removal (if the knee's box overlaps with the body's box, discard it)
            # classes=[0]: Strictly ONLY detect humans (COCO class 0)
            results = yolo_tracker.track(frame, classes=[0], conf=0.6, iou=0.45, persist=True, verbose=False)
            annotated_frame = frame.copy() # Start with the clean frame
            zone_color = (255, 0, 0)
            intrusion_detected = False
            
            # 3. Spatial Intrusion Mathematics & Database Logging
            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                track_ids = results[0].boxes.id.cpu().numpy()
                for box, track_id in zip(boxes, track_ids):
                    x1, y1, x2, y2 = box
                    
                    # --- NEW: Size Filter ---
                    # Lowered the size filter so a peeking head (which is smaller) can still trigger the alarm
                    if (x2 - x1) < 50 or (y2 - y1) < 50:
                        continue
                    
                    # Draw the bounding box for the person manually
                    cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 165, 0), 2)
                    
                    cx = int((x1 + x2) / 2)
                    cy = int(y2)
                    
                    # Check if ANY part of the person's bounding box is inside the fence (Corners + Center)
                    points_to_test = [
                        (x1, y1), (x2, y1), (x1, y2), (x2, y2), # 4 Corners
                        (cx, cy), # Bottom center
                        (cx, int((y1+y2)/2)) # True center
                    ]
                    
                    inside = False
                    for pt in points_to_test:
                        if cv2.pointPolygonTest(fence, (int(pt[0]), int(pt[1])), False) >= 0:
                            inside = True
                            break
                    
                    if inside:
                        intrusion_detected = True
                        cv2.circle(annotated_frame, (cx, cy), 8, (0, 0, 255), -1)
                        cv2.putText(annotated_frame, f"ID:{int(track_id)} INTRUDER", (int(x1), int(y1)-30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                                    
                        # --- TRIGGER DATABASE WRITE ---
                        if track_id not in logged_intruders:
                            conn = sqlite3.connect('surveillance.db')
                            c = conn.cursor()
                            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            c.execute("INSERT INTO security_events (timestamp, event_type, track_id) VALUES (?, ?, ?)",
                                      (timestamp, "Virtual Fence Breach", int(track_id)))
                            conn.commit()
                            conn.close()
                            logged_intruders.add(track_id)
                            print(f"\n[!] ALERT LOGGED: Intruder ID {int(track_id)} breached zone at {timestamp}\n")
                            
                            # --- SEND TELEGRAM ALERT ---
                            alert_msg = f"🚨 *SECURITY ALERT* 🚨\n\n*Event:* Virtual Fence Breach\n*Intruder ID:* {int(track_id)}\n*Time:* {timestamp}"
                            threading.Thread(target=send_telegram_alert, args=(alert_msg,)).start()
                    else:
                        cv2.circle(annotated_frame, (cx, cy), 8, (0, 255, 0), -1)
                        cv2.putText(annotated_frame, f"ID:{int(track_id)}", (int(x1), int(y1)-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        # Reset the logger if they leave the zone, so they trigger a new log if they re-enter
                        if track_id in logged_intruders:
                            logged_intruders.remove(track_id)
                            
            # Update alarm cooldown
            if intrusion_detected:
                alarm_cooldown = 30 # Keep alarm active for 30 frames after last detection
                
            # 4. Trigger Alarm Visuals
            if alarm_cooldown > 0:
                zone_color = (0, 0, 255)
                cv2.putText(annotated_frame, "CRITICAL: VIRTUAL FENCE BREACHED", (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                alarm_cooldown -= 1
            else:
                cv2.putText(annotated_frame, "LIVE SURVEILLANCE: ACTIVE", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                            
            cv2.polylines(annotated_frame, [fence], isClosed=True, color=zone_color, thickness=3)
            
            ret, buffer = cv2.imencode('.jpg', annotated_frame)
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    finally:
        if cap is not None:
            cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_live_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, port=5000)