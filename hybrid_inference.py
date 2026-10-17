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

class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
    def forward(self, lstm_out):
        attn_weights = self.attention(lstm_out)
        attn_weights = torch.softmax(attn_weights, dim=1)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        return context

class SIHActionBiLSTM(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, num_classes=len(CLASSES)):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, 
                            batch_first=True, bidirectional=True, dropout=0.5)
        
        self.attention = TemporalAttention(hidden_dim * 2)
        
        self.ln1 = nn.LayerNorm(hidden_dim * 2 * 3)
        self.fc1 = nn.Linear(hidden_dim * 2 * 3, 128)
        self.ln2 = nn.LayerNorm(128)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x) * 0.05
            x = x + noise
            mask = (torch.rand(x.size(0), x.size(1), 1, device=x.device) > 0.15).float()
            x = x * mask
            
        lstm_out, _ = self.lstm(x) 
        
        attn_pool = self.attention(lstm_out)
        mean_pool = torch.mean(lstm_out, dim=1)
        max_pool = torch.max(lstm_out, dim=1)[0]
        
        temporal_features = torch.cat((attn_pool, mean_pool, max_pool), dim=1) 
        
        x = self.ln1(temporal_features)
        x = self.act(self.ln2(self.fc1(x)))
        x = self.dropout(x)
        return self.fc2(x)

# --- 1. LOAD MODELS INTO MEMORY ---
print("[*] Booting Hybrid AI Surveillance Pipeline...")

lstm_model = SIHActionBiLSTM().to(device)
lstm_model.load_state_dict(torch.load("models/sih_lstm_model.pth", map_location=device, weights_only=True))
lstm_model.eval()

try:
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True, local_files_only=True).to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)
except Exception:
    print("[*] Local CLIP cache not found. Downloading openai/clip-vit-base-patch32...")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True).to(device)
    clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Downloads standard YOLOv8s. COCO Class 43 is 'Knife'.
yolo_model = YOLO("yolov8s.pt")
THREAT_CLASSES = [43]

# --- 2. PIPELINE EXECUTION ---
def letterbox_resize(img, target_size=(224, 224)):
    h, w = img.shape[:2]
    target_w, target_h = target_size
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return canvas

def analyze_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": "Failed to open video file."}
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 1. YOLO-gated extraction exactly like training
    scan_points = np.linspace(0, total_frames - 1, min(total_frames, 80), dtype=int)
    valid_frames = []
    
    # Use yolov8n.pt for gating, exactly like training
    yolo_gater = YOLO("yolov8n.pt")
    
    for idx in scan_points:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret: continue
        
        results = yolo_gater(frame, verbose=False, conf=0.3)
        if len(results[0].boxes) > 0:
            valid_frames.append(frame)
            
    cap.release()
    
    if len(valid_frames) < 16:
        if len(valid_frames) == 0:
            # Absolute worst-case scenario: no objects detected at all.
            # We must fallback to naive sampling just to get 16 frames, though it may misclassify.
            cap = cv2.VideoCapture(video_path)
            frame_indices = np.linspace(0, total_frames - 1, 16, dtype=int)
            valid_frames = []
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret: valid_frames.append(frame)
            cap.release()
        else:
            # We have SOME valid YOLO-gated frames, but fewer than 16.
            # Instead of naive sampling (which introduces blank frames and breaks the distribution),
            # we resample/duplicate the valid frames to get exactly 16.
            resampled_frames = []
            indices = np.linspace(0, len(valid_frames) - 1, 16, dtype=int)
            for idx in indices:
                resampled_frames.append(valid_frames[idx])
            valid_frames = resampled_frames
    if len(valid_frames) >= 16:
        # Uniformly sample exactly 16 frames from valid pool
        final_indices = np.linspace(0, len(valid_frames) - 1, 16, dtype=int)
    else:
        # If we went through the fallback, valid_frames is already exactly 16 frames
        final_indices = np.arange(16)
        
    frames_bgr = []
    frames_pil = []
    
    for idx in final_indices:
        frame = valid_frames[idx]
        frame_padded = letterbox_resize(frame, (224, 224))
        frames_bgr.append(frame) # Keep original for stream A weapon detection
        
        # Apply JPEG compression in-memory to perfectly match training distribution
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
        _, encimg = cv2.imencode('.jpg', frame_padded, encode_param)
        frame_jpeg = cv2.imdecode(encimg, 1)
        
        # Stream B CLIP needs RGB PIL Images
        frames_pil.append(Image.fromarray(cv2.cvtColor(frame_jpeg, cv2.COLOR_BGR2RGB)))


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