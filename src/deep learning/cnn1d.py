# cnn1d.py
import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
import math
from datetime import datetime

# -------- Dataset --------
class SSTDataset(Dataset):
    def __init__(self, spectra, labels, transform=None):
        # spectra: (N, L) numpy
        self.X = spectra.astype(np.float32)
        self.y = labels.astype(np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        x = self.X[idx]
        if self.transform:
            x = self.transform(x)
        # add channel dim
        x = np.expand_dims(x, axis=0)   # (1, L)
        return torch.from_numpy(x), torch.tensor(self.y[idx])
    
# -------- Model --------
class Simple1DCNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10, base_filters=64):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(in_channels, base_filters, kernel_size=7, padding=3),
            nn.BatchNorm1d(base_filters),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(base_filters, base_filters*2, kernel_size=5, padding=2),
            nn.BatchNorm1d(base_filters*2),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(base_filters*2, base_filters*4, kernel_size=3, padding=1),
            nn.BatchNorm1d(base_filters*4),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # -> (B, C, 1)
            )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(base_filters*4, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.net(x)
        x = self.classifier(x)
        return x


# -------- Training loop --------
def train_model(spectra, labels, num_classes, config):
    device = config.get("device", "cpu")
    device = torch.device(device)
    log_path = config.get("log_path")

    def _log(message: str):
        print(message)
        if log_path:
            try:
                with open(log_path, "a") as f:
                    f.write(message + "\n")
            except Exception:
                # If logging fails, continue without interrupting training
                pass

    # Initialize log header
    if log_path:
        try:
            with open(log_path, "w") as f:
                f.write(f"[{datetime.now().isoformat()}] 1D CNN Training Start\n")
                f.write(f"Config: epochs={config.get('epochs')}, batch_size={config.get('batch_size')}, lr={config.get('lr')}, base_filters={config.get('base_filters', 64)}, device={device}\n")
                f.write(f"Num samples={spectra.shape[0]}, Num classes={num_classes}, Feature length={spectra.shape[1]}\n")
        except Exception:
            pass

    X_train, X_val, y_train, y_val = train_test_split(
        spectra, labels, test_size=0.15, stratify=labels, random_state=42)

    train_dataset = SSTDataset(X_train, y_train)
    val_dataset = SSTDataset(X_val, y_val)

    # Weighted sampler for imbalance
    class_counts = np.bincount(y_train)
    class_weights = 1. / (class_counts + 1e-9)
    sample_weights = class_weights[y_train]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], sampler=sampler, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=2)

    model = Simple1DCNN(in_channels=1, num_classes=num_classes, base_filters=config.get("base_filters", 64))
    model.to(device)

    # compute class weights for loss (alternative)
    weights = torch.tensor((labels.size / (num_classes * class_counts))).float() if False else None
    criterion = nn.CrossEntropyLoss()  # or with weight=torch.tensor(...)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=4, factor=0.5)

    best_val_f1 = 0.0
    best_val_acc = 0.0
    best_model = None

    for epoch in range(config['epochs']):
        model.train()
        running_loss = 0.0
        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}"):
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
        avg_train_loss = running_loss / len(train_loader.dataset)