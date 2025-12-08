import os
import json
import numpy as np
import torch
import torch.nn.functional as F
import joblib
from typing import Tuple, Dict, Any
from src.feature_extraction.sst_transform import apply_sst
from src.utils.resampler import resample_signal
from src.deep_learning.cnn1d import Simple1DCNN
from src.config import CONFIG


def load_artifacts(
    model_path: str = "best_cnn1d.pth",
    scaler_path: str = "scaler.pkl",
    encoder_path: str = "label_encoder.pkl",
    sst_params_path: str = "sst_params.json",
    device: str = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")
    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"Label encoder not found: {encoder_path}")
    if not os.path.exists(sst_params_path):
        raise FileNotFoundError(f"SST params file not found: {sst_params_path}")

    scaler = joblib.load(scaler_path)
    label_encoder: Dict[str, Any] = joblib.load(encoder_path)
    with open(sst_params_path, "r") as f:
        sst_params = json.load(f)

    num_classes = len(label_encoder["str_to_int"])  # type: ignore
    model = Simple1DCNN(in_channels=1, num_classes=num_classes)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()