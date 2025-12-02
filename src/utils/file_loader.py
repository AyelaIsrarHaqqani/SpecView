import numpy as np
import os
from typing import List, Tuple, Optional
from PIL import Image

def load_binary_file(filepath: str) -> np.ndarray:
    with open(filepath, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data

def load_image_as_signal(filepath: str) -> np.ndarray:
    img = Image.open(filepath).convert('L')  # grayscale
    arr = np.asarray(img, dtype=np.float32).reshape(-1)
    return arr

def load_file_as_signal(filepath: str) -> np.ndarray:
    ext = os.path.splitext(filepath)[1].lower()
    if ext in {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}:
    return load_image_as_signal(filepath)
return load_binary_file(filepath)
     """
    Load dataset either from a labels CSV (filename,label) or by inferring labels
    from a directory structure of the form: data_dir/<label>/<file>.
    """
    X: List[np.ndarray] = []
    y: List[str] = []

# Path via labels.csv
    if label_file and os.path.exists(label_file):
    import pandas as pd
        df = pd.read_csv(label_file)
        for _, row in df.iterrows():
            filepath = os.path.join(data_dir, row['filename'])
            X.append(load_file_as_signal(filepath))
            y.append(str(row['label']))
        return X, y

# Fallback: infer labels from subdirectories
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")






def load_dataset(data_dir: str, label_file: Optional[str] = None) -> Tuple[List[np.ndarray], List[str]]:


