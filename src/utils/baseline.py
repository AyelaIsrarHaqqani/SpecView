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

def _train_centroids(X: np.ndarray, y: np.ndarray, num_classes: int) -> np.ndarray:
    centroids = []
    for c in range(num_classes):
        class_mask = (y == c)
        if not np.any(class_mask):
            # If a class is missing in train split, create a zero centroid
            centroids.append(np.zeros(X.shape[1], dtype=np.float32))
        else:
            centroids.append(X[class_mask].mean(axis=0))
    return np.vstack(centroids)

def _predict_centroids(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    # Compute squared Euclidean distances to each centroid
    # X shape: (n_samples, n_features), centroids: (n_classes, n_features)
    # Using (x - mu)^2 = x^2 - 2x·mu + mu^2
    x2 = (X ** 2).sum(axis=1, keepdims=True)                # (n, 1)
    mu2 = (centroids ** 2).sum(axis=1, keepdims=True).T     # (1, k)
    cross = X @ centroids.T                                 # (n, k)
    d2 = x2 - 2 * cross + mu2                               # (n, k)
    return np.argmin(d2, axis=1)