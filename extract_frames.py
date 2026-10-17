import os
import cv2
import random
import glob
import numpy as np
from ultralytics import YOLO

# --- CONFIGURATION ---
PROJECT_ROOT = os.getcwd()
RAW_DATASET_DIR = os.path.join(PROJECT_ROOT, "Dataset")
OUTPUT_FRAMES_DIR = os.path.join(PROJECT_ROOT, "Dataset_Frames")

TRAIN_RATIO = 0.8  
TEST_RATIO = 0.2   
NUM_FRAMES = 16
IMG_SIZE = (224, 224)
SEED = 42

random.seed(SEED)

def letterbox_resize(img, target_size=IMG_SIZE):
    """Resizes image to target_size while keeping aspect ratio and padding with black bars."""
    h, w = img.shape[:2]
    target_w, target_h = target_size
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    # Create black canvas
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    
    # Paste centered
    x_offset = (target_w - new_w) // 2
    y_offset = (target_h - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return canvas

def extract_yolo_gated_frames(video_path, output_folder, yolo_model):
    """Extracts exactly 16 uniform frames, strictly enforcing that each frame contains an object."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"      [!] Error: Could not open {os.path.basename(video_path)}")
        return False
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames < NUM_FRAMES:
        cap.release()
        print(f"      [!] Error: {os.path.basename(video_path)} is too short ({total_frames} frames).")
        return False

    # We want to scan across the whole video. We evaluate up to 80 evenly spaced frames.
    scan_points = np.linspace(0, total_frames - 1, min(total_frames, 80), dtype=int)
    
    valid_frames = []
    for idx in scan_points:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        
        # YOLO gate: Only keep frame if it has at least one detection (confidence > 0.3)
        results = yolo_model(frame, verbose=False, conf=0.3)
        if len(results[0].boxes) > 0:
            valid_frames.append(frame)
            
    cap.release()
    
    if len(valid_frames) < NUM_FRAMES:
        print(f"      [!] WARNING: {os.path.basename(video_path)} only had {len(valid_frames)} frames with objects. Skipping.")
        return False
        
    # Uniformly sample exactly 16 frames from the pool of object-confirmed frames
    final_indices = np.linspace(0, len(valid_frames) - 1, NUM_FRAMES, dtype=int)
    
    os.makedirs(output_folder, exist_ok=True)
    
    for i, idx in enumerate(final_indices):
        frame = valid_frames[idx]
        # Apply aspect-ratio padding to prevent warping
        frame_padded = letterbox_resize(frame, IMG_SIZE)
        frame_filename = os.path.join(output_folder, f"frame_{i:02d}.jpg")
        cv2.imwrite(frame_filename, frame_padded)
        
    return True

def process_phase_one():
    print("=" * 60)
    print("[*] PHASE 1: YOLO-GATED STRATIFIED FRAME EXTRACTION (80/20)")
    print("=" * 60)
    
    print("[*] Loading YOLOv8n for Object Gating...")
    yolo_model = YOLO("yolov8n.pt")
    
    classes = [d for d in os.listdir(RAW_DATASET_DIR) if os.path.isdir(os.path.join(RAW_DATASET_DIR, d))]
    
    if not classes:
        print(f"[ERROR] No class folders found in {RAW_DATASET_DIR}")
        return

    failed_videos = []

    for cls in classes:
        class_path = os.path.join(RAW_DATASET_DIR, cls)
        
        videos = []
        for ext in ('*.mp4', '*.avi', '*.mkv', '*.mov', '*.webm'):
            videos.extend(glob.glob(os.path.join(class_path, ext)))
            
        videos = sorted(videos)
        random.shuffle(videos)
        
        num_videos = len(videos)
        if num_videos == 0:
            continue
            
        split_idx = int(num_videos * TRAIN_RATIO)
        train_vids = videos[:split_idx]
        test_vids = videos[split_idx:]
        
        print(f"\n[*] Processing Class: {cls} | Train: {len(train_vids)} | Test: {len(test_vids)}")
        
        for split_name, vid_list in [("train", train_vids), ("test", test_vids)]:
            print(f"    -> Extracting {split_name.upper()} split...")
            for idx, vid_path in enumerate(vid_list, 1):
                vid_basename = os.path.splitext(os.path.basename(vid_path))[0]
                out_dir = os.path.join(OUTPUT_FRAMES_DIR, split_name, cls, vid_basename)
                
                # Check if this video has already been processed (contains exactly 16 frames)
                if os.path.exists(out_dir) and len(os.listdir(out_dir)) == NUM_FRAMES:
                    success = True
                else:
                    success = extract_yolo_gated_frames(vid_path, out_dir, yolo_model)
                    
                if not success:
                    failed_videos.append(vid_path)
                
                if idx % 10 == 0 or idx == len(vid_list):
                    print(f"       Processed {idx}/{len(vid_list)} videos...")
                    
    print("\n" + "=" * 60)
    if failed_videos:
        print(f"[!] COMPLETED WITH {len(failed_videos)} FAILED VIDEOS (No objects found).")
        print("Failed videos list saved to 'failed_videos.txt'. You must replace these!")
        with open("failed_videos.txt", "w") as f:
            for vid in failed_videos:
                f.write(vid + "\n")
    else:
        print("[SUCCESS] ALL FRAMES EXTRACTED. NO BLANK/NO-OBJECT FRAMES DETECTED.")
    print("=" * 60)

if __name__ == "__main__":
    process_phase_one()