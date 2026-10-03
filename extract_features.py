import os
import glob
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# --- CONFIGURATION ---
PROJECT_ROOT = os.getcwd()
FRAMES_DIR = os.path.join(PROJECT_ROOT, "Dataset_Frames")
FEATURES_DIR = os.path.join(PROJECT_ROOT, "Dataset_Features")

# Route computation to your local RTX 3050
device = "cuda" if torch.cuda.is_available() else "cpu"

def process_phase_two():
    print("=" * 60)
    print("[*] PHASE 2: OPENAI CLIP FEATURE SERIALIZATION")
    print("=" * 60)
    
    print(f"[*] Loading CLIP ViT-B/32 on {device.upper()}...")
    # Downloads the official OpenAI model weights securely via safetensors
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True).to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    for split in ["train", "test"]:
        split_path = os.path.join(FRAMES_DIR, split)
        if not os.path.exists(split_path):
            continue
            
        classes = os.listdir(split_path)
        for cls in classes:
            cls_path = os.path.join(split_path, cls)
            if not os.path.isdir(cls_path):
                continue
                
            out_cls_dir = os.path.join(FEATURES_DIR, split, cls)
            os.makedirs(out_cls_dir, exist_ok=True)
            
            videos = os.listdir(cls_path)
            print(f"\n[*] Processing {split.upper()} | Class: {cls} ({len(videos)} videos)")
            
            for idx, vid_folder in enumerate(videos, 1):
                vid_path = os.path.join(cls_path, vid_folder)
                
                # Verify exactly 16 frames exist
                frame_files = sorted(glob.glob(os.path.join(vid_path, "*.jpg")))
                if len(frame_files) != 16:
                    print(f"    [!] Warning: {vid_folder} has {len(frame_files)} frames. Skipping.")
                    continue
                    
                images = [Image.open(f).convert("RGB") for f in frame_files]
                
                # Pre-process and send batch to GPU
                inputs = processor(images=images, return_tensors="pt").to(device)
                
                with torch.no_grad():
                    # Extract embeddings
                    outputs = model.get_image_features(**inputs)
                    
                    # Robust version handler: extracts the raw tensor regardless of the Hugging Face version
                    if isinstance(outputs, torch.Tensor):
                        features = outputs
                    elif hasattr(outputs, "image_embeds"):
                        features = outputs.image_embeds
                    elif hasattr(outputs, "pooler_output"):
                        features = outputs.pooler_output
                    else:
                        features = outputs[0]
                        
                    features = features.cpu().numpy() # Shape: (16, 512)
                
                # Save the array directly to disk
                out_file = os.path.join(out_cls_dir, f"{vid_folder}.npy")
                np.save(out_file, features)
                
                # Terminal updates to prevent Windows from pausing the console
                if idx % 20 == 0 or idx == len(videos):
                    print(f"    -> Extracted {idx}/{len(videos)} sequence tensors...")

    print("\n" + "=" * 60)
    print("[SUCCESS] ALL FRAMES CONVERTED TO CLIP EMBEDDINGS.")
    print("=" * 60)

if __name__ == "__main__":
    process_phase_two()