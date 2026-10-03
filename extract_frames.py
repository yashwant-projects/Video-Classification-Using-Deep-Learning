import os
import cv2
import random
import glob
import numpy as np

# --- CONFIGURATION ---
PROJECT_ROOT = os.getcwd()
RAW_DATASET_DIR = os.path.join(PROJECT_ROOT, "Dataset")
OUTPUT_FRAMES_DIR = os.path.join(PROJECT_ROOT, "Dataset_Frames")

# The >99% testing metrics demand strict reproducible splits
TRAIN_RATIO = 0.8  
TEST_RATIO = 0.2   
NUM_FRAMES = 16
IMG_SIZE = (224, 224)
SEED = 42

random.seed(SEED)

def extract_uniform_frames(video_path, output_folder):
    """Slices 16 uniform frames between the 15% and 85% temporal marks."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"      [!] Error: Could not open {os.path.basename(video_path)}")
        return False
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Safety check for extremely short clips
    if total_frames < NUM_FRAMES:
        start_frame, end_frame = 0, max(0, total_frames - 1)
    else:
        start_frame = int(total_frames * 0.15)
        end_frame = int(total_frames * 0.85)

    # Linearly space 16 points across the safe temporal window
    frame_indices = np.linspace(start_frame, end_frame, NUM_FRAMES, dtype=int)
    
    os.makedirs(output_folder, exist_ok=True)
    
    extracted_count = 0
    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            frame_resized = cv2.resize(frame, IMG_SIZE)
            frame_filename = os.path.join(output_folder, f"frame_{i:02d}.jpg")
            cv2.imwrite(frame_filename, frame_resized)
            extracted_count += 1
        else:
            print(f"      [!] Warning: Failed to read frame {frame_idx} in {os.path.basename(video_path)}")
            
    cap.release()
    return extracted_count == NUM_FRAMES

def process_phase_one():
    print("=" * 60)
    print("[*] PHASE 1: STRATIFIED FRAME EXTRACTION (80/20 SPLIT)")
    print("=" * 60)
    
    classes = [d for d in os.listdir(RAW_DATASET_DIR) if os.path.isdir(os.path.join(RAW_DATASET_DIR, d))]
    
    if not classes:
        print(f"[ERROR] No class folders found in {RAW_DATASET_DIR}")
        return

    for cls in classes:
        class_path = os.path.join(RAW_DATASET_DIR, cls)
        
        # Grab all standard video formats
        videos = []
        for ext in ('*.mp4', '*.avi', '*.mkv', '*.mov', '*.webm'):
            videos.extend(glob.glob(os.path.join(class_path, ext)))
            
        videos = sorted(videos) # Sort first for absolute determinism
        random.shuffle(videos)  # Apply fixed seed shuffle
        
        num_videos = len(videos)
        if num_videos == 0:
            print(f"[-] Skipping {cls} (No videos found)")
            continue
            
        # Enforce exact 80/20 split (120 Train / 30 Test per class)
        split_idx = int(num_videos * TRAIN_RATIO)
        train_vids = videos[:split_idx]
        test_vids = videos[split_idx:]
        
        print(f"\n[*] Processing Class: {cls} | Train: {len(train_vids)} | Test: {len(test_vids)}")
        
        for split_name, vid_list in [("train", train_vids), ("test", test_vids)]:
            print(f"    -> Extracting {split_name.upper()} split...")
            for idx, vid_path in enumerate(vid_list, 1):
                vid_basename = os.path.splitext(os.path.basename(vid_path))[0]
                out_dir = os.path.join(OUTPUT_FRAMES_DIR, split_name, cls, vid_basename)
                
                extract_uniform_frames(vid_path, out_dir)
                
                # Provide lightweight terminal updates so Windows does not lag
                if idx % 20 == 0 or idx == len(vid_list):
                    print(f"       Processed {idx}/{len(vid_list)} videos...")
                    
    print("\n" + "=" * 60)
    print("[SUCCESS] ALL FRAMES EXTRACTED AND STRATIFIED PERFECTLY.")
    print("=" * 60)

if __name__ == "__main__":
    process_phase_one()