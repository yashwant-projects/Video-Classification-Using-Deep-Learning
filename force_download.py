import torch
import torchvision.models as models
from ultralytics import YOLO
import easyocr
from facenet_pytorch import MTCNN

print("=" * 50)
print("INITIATING HEAVY WEIGHT DOWNLOADS")
print("=" * 50)

print("\n1. Downloading Torchvision CNN/ViT Backbones...")
# Downloads the ResNet and ViT weights (~500 MB)
resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
vit = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)

print("\n2. Downloading YOLOv8 Surveillance Weights...")
# Downloads the nano and small real-time object detection weights (~200 MB)
yolo_n = YOLO("yolov8n.pt")
yolo_s = YOLO("yolov8s.pt")

print("\n3. Downloading EasyOCR (ANPR) Weights...")
# Downloads the CRAFT text detection and recognition weights (~200 MB)
reader = easyocr.Reader(['en'], gpu=True)

print("\n4. Downloading Face Detection Weights...")
# Downloads the MTCNN facial landmark weights (~150 MB)
mtcnn = MTCNN(device='cuda' if torch.cuda.is_available() else 'cpu')

print("\n" + "=" * 50)
print("[SUCCESS] ALL MODELS CACHED LOCALLY. SAFE TO GO OFFLINE.")
print("=" * 50)