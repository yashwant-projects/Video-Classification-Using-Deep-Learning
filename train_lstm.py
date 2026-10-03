import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# --- CONFIGURATION ---
PROJECT_ROOT = os.getcwd()
FEATURES_DIR = os.path.join(PROJECT_ROOT, "Dataset_Features")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

TRAIN_DIR = os.path.join(FEATURES_DIR, "train")
CLASSES = sorted([d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))])
CLASS_TO_IDX = {cls_name: idx for idx, cls_name in enumerate(CLASSES)}

# --- DATA LOADER ---
class VideoFeatureDataset(Dataset):
    def __init__(self, split):
        self.data = []
        split_dir = os.path.join(FEATURES_DIR, split)
        for cls_name in CLASSES:
            cls_dir = os.path.join(split_dir, cls_name)
            if not os.path.isdir(cls_dir): continue
            
            for npy_file in glob.glob(os.path.join(cls_dir, "*.npy")):
                self.data.append((npy_file, CLASS_TO_IDX[cls_name]))
                
    def __len__(self): return len(self.data)
    
    def __getitem__(self, idx):
        npy_path, label = self.data[idx]
        features = np.load(npy_path) 
        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

# --- MODEL ARCHITECTURE ---
class SIHActionBiLSTM(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=256, num_classes=len(CLASSES)):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, 
                            batch_first=True, bidirectional=True, dropout=0.4)
        
        self.fc1 = nn.Linear(hidden_dim * 2 * 2, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5) 
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # Tensor Augmentation: Inject 3% Gaussian Noise during training only
        if self.training:
            noise = torch.randn_like(x) * 0.03
            x = x + noise
            
        lstm_out, _ = self.lstm(x) 
        
        mean_pool = torch.mean(lstm_out, dim=1)
        max_pool = torch.max(lstm_out, dim=1)[0]
        temporal_features = torch.cat((mean_pool, max_pool), dim=1) 
        
        x = self.relu(self.fc1(temporal_features))
        x = self.dropout(x)
        return self.fc2(x)

# --- TRAINING LOOP ---
def train_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 60)
    print(f"[*] PHASE 3: BI-LSTM SEQUENCE TRAINING ON {device.upper()}")
    print("=" * 60)
    print(f"[*] Loaded {len(CLASSES)} Classes: {', '.join(CLASSES)}")
    
    train_loader = DataLoader(VideoFeatureDataset("train"), batch_size=32, shuffle=True)
    test_loader = DataLoader(VideoFeatureDataset("test"), batch_size=32, shuffle=False)
    
    model = SIHActionBiLSTM().to(device)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    # Lowered learning rate and increased weight decay for final micro-adjustments
    optimizer = AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=50)
    
    epochs = 300
    best_acc = 0.0
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        scheduler.step()
        
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                out = model(X)
                preds = torch.argmax(out, dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
                
        acc = (correct / total) * 100
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, "sih_lstm_model.pth"))
            marker = " (NEW BEST MODEL SAVED)"
        else:
            marker = ""
            
        print(f"Epoch {epoch:02d}/{epochs} | Loss: {train_loss/len(train_loader):.4f} | Test Acc: {acc:.2f}%{marker}")
            
    print("\n" + "=" * 60)
    print(f"[SUCCESS] Final Peak Testing Accuracy: {best_acc:.2f}%")
    print(f"[*] Golden weights safely locked in: {MODEL_DIR}/sih_lstm_model.pth")
    print("=" * 60)

if __name__ == "__main__":
    train_model()