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