import numpy as np
from typing import List, Tuple, Dict, Any
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def _encode_labels(labels: List[str]) -> Tuple[np.ndarray, Dict[str, int], Dict[int, str]]:
    unique_labels = sorted(set(labels))
    str_to_int = {s: i for i, s in enumerate(unique_labels)}
    int_to_str = {i: s for s, i in str_to_int.items()}
    y_int = np.array([str_to_int[s] for s in labels], dtype=np.int64)
    return y_int, str_to_int, int_to_str