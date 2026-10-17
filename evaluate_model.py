import os
import glob
import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report

# --- CONFIGURATION ---
PROJECT_ROOT = os.getcwd()
FEATURES_DIR = os.path.join(PROJECT_ROOT, "Dataset_Features")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "sih_lstm_model.pth")

# Load exactly as trained
CLASSES = sorted([d for d in os.listdir(os.path.join(FEATURES_DIR, "train")) if os.path.isdir(os.path.join(FEATURES_DIR, "train", d))])
CLASS_TO_IDX = {cls_name: idx for idx, cls_name in enumerate(CLASSES)}

class VideoFeatureDataset(Dataset):
    def __init__(self, split):
        self.data = []
        split_dir = os.path.join(FEATURES_DIR, split)
        for cls_name in CLASSES:
            cls_dir = os.path.join(split_dir, cls_name)
            for npy_file in glob.glob(os.path.join(cls_dir, "*.npy")):
                self.data.append((npy_file, CLASS_TO_IDX[cls_name]))
                
    def __len__(self): return len(self.data)
    
    def __getitem__(self, idx):
        npy_path, label = self.data[idx]
        return torch.tensor(np.load(npy_path), dtype=torch.float32), torch.tensor(label, dtype=torch.long)

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

def run_diagnostics():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test_loader = DataLoader(VideoFeatureDataset("test"), batch_size=32, shuffle=False)
    
    model = SIHActionBiLSTM().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()
    
    all_preds, all_labels = [], []
    print("=" * 60)
    print("[*] RUNNING TEST SET DIAGNOSTICS...")
    print("=" * 60)
    
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device)
            preds = torch.argmax(model(X), dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())
            
    print("\n[*] CLASSIFICATION REPORT")
    print(classification_report(all_labels, all_preds, target_names=CLASSES, digits=4))
    
    # Generate HOD-ready visualization
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
    plt.title('SIH Civilian Safety - Test Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True Class', fontsize=12, fontweight='bold')
    plt.xlabel('Predicted Class', fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    cm_path = os.path.join(PROJECT_ROOT, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    print("=" * 60)
    print(f"[SUCCESS] Graphic saved to: {cm_path}")
    print("=" * 60)

if __name__ == "__main__":
    run_diagnostics()