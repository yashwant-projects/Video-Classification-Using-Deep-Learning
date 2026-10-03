import os
import zipfile
import cv2
import numpy as np

# --- CONFIGURATION ---
PROJECT_ROOT = os.getcwd()
ZIP_FILE = os.path.join(PROJECT_ROOT, "archive (3).zip")
TARGET_DIR = os.path.join(PROJECT_ROOT, "Dataset", "Traffic")
TEMP_VIDEO = os.path.join(PROJECT_ROOT, "_temp_eval_video.mp4")

TARGET_COUNT = 150
SAMPLE_FRAMES = 8  # Number of frames to sample per video for blur calculation

os.makedirs(TARGET_DIR, exist_ok=True)

def calculate_video_sharpness(video_path, num_samples=8):
    """
    Samples frames across a video and returns the average Variance of the Laplacian.
    Higher value = Sharper image. Lower value = Blurry image.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return -1.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return -1.0

    # Pick evenly spaced frame indices
    step = max(1, total_frames // (num_samples + 1))
    sample_indices = [i * step for i in range(1, num_samples + 1)]

    scores = []
    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        # Convert to grayscale and compute the Laplacian variance
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        score = cv2.Laplacian(gray, cv2.CV_64F).var()
        scores.append(score)

    cap.release()
    return float(np.mean(scores)) if scores else -1.0

def process_traffic_videos():
    print("=" * 60)
    print("[*] STARTING TRAFFIC DATASET QUALITY RANKING & EXTRACTION")
    print("=" * 60)

    if not os.path.exists(ZIP_FILE):
        print(f"[ERROR] Could not find {ZIP_FILE}. Check the file name in your project folder.")
        return

    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    scored_videos = []

    with zipfile.ZipFile(ZIP_FILE, 'r') as z:
        # Collect all video members in the 'video' folder
        all_members = [
            m for m in z.infolist() 
            if not m.is_dir() and m.filename.lower().endswith(video_extensions)
        ]

        total_found = len(all_members)
        print(f"[*] Found {total_found} traffic videos in archive.")
        print(f"[*] Analyzing frame clarity using OpenCV (sampling {SAMPLE_FRAMES} frames per video)...")

        for idx, member in enumerate(all_members, 1):
            ext = os.path.splitext(member.filename)[1]
            temp_path = os.path.join(PROJECT_ROOT, f"_temp_eval{ext}")

            # Temporarily write file bytes to assess with cv2.VideoCapture
            with open(temp_path, "wb") as f:
                f.write(z.read(member))

            # Calculate sharpness score
            score = calculate_video_sharpness(temp_path, num_samples=SAMPLE_FRAMES)
            scored_videos.append((member, score))

            # Clean up temp file immediately
            if os.path.exists(temp_path):
                os.remove(temp_path)

            if idx % 25 == 0 or idx == total_found:
                print(f"    -> Scanned {idx}/{total_found} videos...")

    # Sort descending: sharpest videos first
    scored_videos.sort(key=lambda x: x[1], reverse=True)

    # Filter out unreadable videos
    valid_videos = [item for item in scored_videos if item[1] >= 0]
    print(f"\n[*] Completed quality assessment. Valid videos: {len(valid_videos)}/{total_found}")

    selected = valid_videos[:TARGET_COUNT]
    print(f"[*] Extracting top {len(selected)} highest clarity videos into: {TARGET_DIR}")

    with zipfile.ZipFile(ZIP_FILE, 'r') as z:
        for i, (member, score) in enumerate(selected, 1):
            base_filename = os.path.basename(member.filename)
            dest_path = os.path.join(TARGET_DIR, base_filename)
            
            with open(dest_path, "wb") as f:
                f.write(z.read(member))
            
            if i % 30 == 0 or i == len(selected):
                print(f"    -> Extracted {i}/{len(selected)} (Current score: {score:.2f})")

    print("\n" + "=" * 60)
    print(f"[SUCCESS] Extracted {len(selected)} high-clarity videos to /Dataset/Traffic")
    print(f"[*] Filtered out {total_found - len(selected)} lower-quality/blurry clips.")
    print("=" * 60)

if __name__ == "__main__":
    process_traffic_videos()