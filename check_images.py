import os
from PIL import Image

dir_path = r"C:\Users\yashw\.gemini\antigravity-ide\brain\638235b8-0a62-4b89-bebf-12cf249d9871\.user_uploaded"
for f in os.listdir(dir_path):
    if f.lower().endswith(('.png', '.jpg', '.jpeg')):
        full_path = os.path.join(dir_path, f)
        img = Image.open(full_path)
        print(f"{f}: {img.size}, {img.format}")
