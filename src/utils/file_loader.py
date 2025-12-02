import numpy as np
import os
from typing import List, Tuple, Optional
from PIL import Image

def load_binary_file(filepath: str) -> np.ndarray:
    with open(filepath, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data
    

