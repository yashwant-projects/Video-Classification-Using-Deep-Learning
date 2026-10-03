import os
import zipfile
import random

# --- CONFIGURATION ---
PROJECT_ROOT = os.getcwd()
TARGET_BASE = os.path.join(PROJECT_ROOT, "Dataset")

# We only need UCF101 (which is your 6.8 GB archive.zip)
UCF_ZIP = os.path.join(PROJECT_ROOT, "archive.zip")

# The precise SIH blueprint using ONLY UCF101
MAPPING = {
    "Human_Activities": [
        "ApplyEyeMakeup", "ApplyLipstick", "BabyCrawling", "BlowDryHair", "BlowingCandles", 
        "BrushingTeeth", "CuttingInKitchen", "Haircut", "Hammering", "MoppingFloor", 
        "ShavingBeard", "WalkingWithDog"
    ],
    "Sports": [
        "Archery", "Basketball", "Biking", "Bowling", "CliffDiving", "CricketShot", 
        "Fencing", "FrisbeeCatch", "GolfSwing", "HorseRiding", "IceDancing", "JumpRope", 
        "Kayaking", "RockClimbingIndoor", "Rowing", "SkateBoarding", "Skiing", "Surfing", 
        "TennisSwing", "VolleyballSpiking"
    ],
    "Entertainment": [
        "BandMarching", "Drumming", "JugglingBalls", "MilitaryParade", "PlayingCello", 
        "PlayingDhol", "PlayingFlute", "PlayingGuitar", "PlayingPiano", "PlayingViolin", 
        "SalsaSpin", "YoYo"
    ],
    "Education": [
        "WritingOnBoard", "Typing"
    ]
}

def extract_ucf101_only():
    print("=" * 60)
    print("[*] INITIATING UCF101-ONLY BALANCED EXTRACTION")
    print("=" * 60)
    
    if not os.path.exists(UCF_ZIP):
        print(f"[ERROR] Could not find {UCF_ZIP}. Check the file name in your folder!")
        return

    found_videos = {cat: {folder: [] for folder in folders} for cat, folders in MAPPING.items()}
    
    print(f"[*] Scanning memory of {os.path.basename(UCF_ZIP)}...")
    with zipfile.ZipFile(UCF_ZIP, 'r') as z:
        for file_info in z.infolist():
            if file_info.is_dir(): continue
            
            # Grabs the folder name dynamically 
            parent_folder = os.path.basename(os.path.dirname(file_info.filename))
            
            for target_class, valid_folders in MAPPING.items():
                if parent_folder in valid_folders:
                    found_videos[target_class][parent_folder].append(file_info)

    for target_class, folders_dict in found_videos.items():
        print(f"\n[*] Processing '{target_class}' (Target: 150 videos)")
        
        active_folders = {k: v for k, v in folders_dict.items() if len(v) > 0}
        num_folders = len(active_folders)
        
        if num_folders == 0:
            print(f"    -> [!] No videos found. Skipping.")
            continue
            
        base_count = 150 // num_folders
        remainder = 150 % num_folders
        final_selection = []
        
        for folder, files in active_folders.items():
            take_count = base_count + (1 if remainder > 0 else 0)
            if remainder > 0: remainder -= 1
            
            take_count = min(take_count, len(files))
            sampled = random.sample(files, take_count)
            final_selection.extend(sampled)
            
        print(f"    -> Selected {len(final_selection)} diverse videos from {num_folders} sub-folders.")
        
        out_dir = os.path.join(TARGET_BASE, target_class)
        os.makedirs(out_dir, exist_ok=True)
        
        extracted = 0
        with zipfile.ZipFile(UCF_ZIP, 'r') as z:
            for file_info in final_selection:
                source_stream = z.open(file_info)
                target_path = os.path.join(out_dir, os.path.basename(file_info.filename))
                with open(target_path, "wb") as f:
                    f.write(source_stream.read())
                extracted += 1
                
        print(f"    -> [SUCCESS] Extracted {extracted} clips directly to /Dataset/{target_class}")
        
    print("\n" + "=" * 60)
    print("[SUCCESS] ALL CATEGORIES FULLY EXTRACTED AND BALANCED!")
    print("=" * 60)

if __name__ == "__main__":
    extract_ucf101_only()