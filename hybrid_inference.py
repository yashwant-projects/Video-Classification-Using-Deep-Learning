import os
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from ultralytics import YOLO
import torch.nn as nn

device = "cuda" if torch.cuda.is_available() else "cpu"

# --- ARCHITECTURE (MUST MATCH EXACTLY) ---
CLASSES = ['Education', 'Entertainment', 'Human_Activities', 'Sports', 'Traffic', 'Violence', 'Wildlife']

class SIHActionBiLSTM(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, num_classes=len(CLASSES)):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True, dropout=0.4)
        self.fc1 = nn.Linear(hidden_dim * 2 * 2, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        mean_pool = torch.mean(lstm_out, dim=1)
        max_pool = torch.max(lstm_out, dim=1)[0]
        temporal_features = torch.cat((mean_pool, max_pool), dim=1)
        return self.fc2(self.dropout(self.relu(self.fc1(temporal_features))))

# --- 1. LOAD MODELS INTO MEMORY ---
print("[*] Booting Hybrid AI Surveillance Pipeline...")

lstm_model = SIHActionBiLSTM().to(device)
lstm_model.load_state_dict(torch.load("models/sih_lstm_model.pth", map_location=device, weights_only=True))
lstm_model.eval()

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True).to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Downloads standard YOLOv8s. COCO Class 43 is 'Knife'.
yolo_model = YOLO("yolov8s.pt")
THREAT_CLASSES = [43]

# --- 2. PIPELINE EXECUTION ---
def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Failed to open video file."}
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(int(total_frames*0.15), int(total_frames*0.85), 16, dtype=int)
    frames_bgr = []
    frames_pil = []
    
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames_bgr.append(frame)
            frames_pil.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((224, 224)))
    cap.release()
    
    if len(frames_pil) != 16:
        return {"error": "Corrupted video: Could not extract 16 frames."}

    # Stream A: YOLOv8 Spatial Weapon Detection
    threat_detected = False
    threat_details = []
    
    for img in [frames_bgr[0], frames_bgr[7], frames_bgr[15]]:
        results = yolo_model(img, verbose=False)[0]
        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            if cls_id in THREAT_CLASSES and conf > 0.40:
                threat_detected = True
                threat_details.append(f"Edged Weapon Detected ({conf*100:.1f}%)")

    # Stream B: CLIP + LSTM Temporal Action Recognition
    inputs = clip_processor(images=frames_pil, return_tensors="pt").to(device)
    with torch.no_grad():
        # Universal foolproof extraction bypassing version differences
        vision_outputs = clip_model.vision_model(pixel_values=inputs.pixel_values)
        features = clip_model.visual_projection(vision_outputs.pooler_output)
        features = features.cpu().numpy()
        
    # Unsqueeze adds the batch dimension, creating a perfect 3D tensor: (1, 16, 512)
    X = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        lstm_out = lstm_model(X)
        probs = torch.softmax(lstm_out, dim=1)[0]
        pred_idx = torch.argmax(probs).item()
        action_class = CLASSES[pred_idx]
        action_conf = probs[pred_idx].item() * 100

    # Stream C: Threat Synthesis Matrix
    final_status = "Safe"
    alert_color = "green"
    
    if threat_detected:
        final_status = "CRITICAL: Armed Threat Detected"
        alert_color = "red"
    elif action_class == "Violence":
        final_status = "CRITICAL: Physical Violence Detected"
        alert_color = "red"
    elif action_class in ["Traffic", "Wildlife"]:
        final_status = f"Monitor: {action_class} Activity"
        alert_color = "yellow"

    return {
        "status": "success",
        "action_prediction": action_class,
        "action_confidence": f"{action_conf:.2f}%",
        "yolo_threat": threat_detected,
        "yolo_details": threat_details,
        "system_status": final_status,
        "alert_color": alert_color
    }