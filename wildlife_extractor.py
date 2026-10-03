import os
import ast
import random
import openpyxl
import tarfile

# --- CONFIGURATION ---
PROJECT_ROOT = os.getcwd()
METADATA_FILE = "AR_metadata.xlsx"
TAR_FILE = os.path.join(PROJECT_ROOT, "video.tar.gz") 
TARGET_FOLDER = os.path.join(PROJECT_ROOT, "Dataset", "Wildlife")
os.makedirs(TARGET_FOLDER, exist_ok=True)

def extract_wildlife_from_tar():
    print("=" * 50)
    print("[*] Initiating Pure Python Tarball Extraction (Extreme Diversity)...")
    
    # 1. Load the workbook without Pandas
    wb = openpyxl.load_workbook(METADATA_FILE, data_only=True)
    sheet = wb["AR"]
    
    targets = {
        'Mammal': {}, 'Bird': {}, 'Reptile': {}, 'Fish': {}, 'Amphibian': {}
    }
    
    print("[*] Parsing 30,000 rows silently to enforce species diversity...")
    # 2. Iterate through the spreadsheet manually
    for row in sheet.iter_rows(min_row=2, values_only=True):
        video_id = row[1]          # Column B
        animal_str = row[5]        # Column F (Specific Species)
        parent_class_str = row[6]  # Column G (Parent Class)
        
        if not video_id or not parent_class_str or not animal_str:
            continue
            
        try:
            parent_classes = ast.literal_eval(parent_class_str)
            species_list = ast.literal_eval(animal_str)
            
            if parent_classes and species_list:
                primary_parent = parent_classes[0]
                primary_species = species_list[0]
                
                if primary_parent in targets:
                    # If we don't have this specific species yet, add it!
                    if primary_species not in targets[primary_parent]:
                        targets[primary_parent][primary_species] = video_id
        except:
            continue

    selected_videos = []
    print("\n[*] Sampling exactly 1 video per unique species...")
    
    # 3. Sample the targets
    for animal_class, species_dict in targets.items():
        unique_videos = list(species_dict.values())
        sampled = random.sample(unique_videos, 30) if len(unique_videos) >= 30 else unique_videos
        selected_videos.extend(sampled)
        print(f"    - Secured {len(sampled)} highly diverse {animal_class}s")
        
    print(f"\n[*] Reaching into {TAR_FILE} to extract ONLY the needed files...")
    print("[*] (This might take a minute as it scans the 15GB archive. Do not close!)")
    
    # ---> THIS IS THE BLOCK THAT GOT DELETED <---
    # Build a set of exact filenames we want to find
    target_filenames = set()
    for vid in selected_videos:
        for ext in ['.mp4', '.mkv', '.avi']:
            target_filenames.add(f"{vid}{ext}")
            
    copied_count = 0
    
    # 4. Read directly from the .tar file without extracting it first
    try:
        with tarfile.open(TAR_FILE, "r") as tar:
            for member in tar: 
                base_name = os.path.basename(member.name)
                
                # If the file matches one of our 150 targets, extract it
                if base_name in target_filenames:
                    source_stream = tar.extractfile(member)
                    if source_stream:
                        target_path = os.path.join(TARGET_FOLDER, base_name)
                        with open(target_path, "wb") as f:
                            f.write(source_stream.read())
                        copied_count += 1
                        print(f"    -> Extracted: {base_name} ({copied_count}/150)")
                        
                        # Stop scanning early if we found all 150
                        if copied_count >= 150:
                            break
    except FileNotFoundError:
        print(f"\n[ERROR] Could not find {TAR_FILE}. Make sure it is in the exact folder as this script!")
        return

    print("\n" + "=" * 50)
    print(f"[SUCCESS] EXTRACTION COMPLETE!")
    print(f"Files Copied: {copied_count} / 150")
    print("=" * 50)

if __name__ == "__main__":
    extract_wildlife_from_tar()