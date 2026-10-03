import os
import torch
import numpy as np
import torch.nn as nn

# --- CONFIGURATION ---
PROJECT_ROOT = os.getcwd()
FEATURES_DIR = os.path.join(PROJECT_ROOT, "Dataset_Features")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "sih_lstm_model.pth")

CLASSES = sorted([d for d in os.listdir(os.path.join(FEATURES_DIR, "train")) if os.path.isdir(os.path.join(FEATURES_DIR, "train", d))])
CLASS_TO_IDX = {cls_name: idx for idx, cls_name in enumerate(CLASSES)}
IDX_TO_CLASS = {idx: cls_name for cls_name, idx in CLASS_TO_IDX.items()}

# --- ARCHITECTURE ---
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

def find_culprits():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SIHActionBiLSTM().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    print("=" * 60)
    print("[*] HUNTING FOR THE 3 FAILED VIDEOS...")
    print("=" * 60)

    test_dir = os.path.join(FEATURES_DIR, "test")
    failed_count = 0
    
    with torch.no_grad():
        for cls_name in CLASSES:
            cls_dir = os.path.join(test_dir, cls_name)
            if not os.path.isdir(cls_dir): continue
            
            for npy_file in os.listdir(cls_dir):
                if not npy_file.endswith(".npy"): continue
                
                npy_path = os.path.join(cls_dir, npy_file)
                features = np.load(npy_path)
                # Add the batch dimension required by PyTorch
                X = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device) 
                
                out = model(X)
                pred_idx = torch.argmax(out, dim=1).item()
                pred_class = IDX_TO_CLASS[pred_idx]
                
                if pred_class != cls_name:
                    failed_count += 1
                    vid_name = npy_file.replace('.npy', '')
                    print(f"[{failed_count}] TRUE: {cls_name} | PREDICTED: {pred_class} | FILE: {vid_name}")

if __name__ == "__main__":
    find_culprits()