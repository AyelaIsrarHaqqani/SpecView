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

def train_and_evaluate(features: List[np.ndarray], labels: List[str], test_size: float = 0.2, random_state: int = 42):
    X = np.vstack([np.asarray(f, dtype=np.float32).reshape(1, -1) for f in features])
    y_int, str_to_int, int_to_str = _encode_labels(labels)

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_int, test_size=test_size, random_state=random_state, stratify=y_int
    )

    # Standardize features using train statistics
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)

    num_classes = len(str_to_int)
    centroids = _train_centroids(X_train_std, y_train, num_classes)
    y_pred_int = _predict_centroids(X_test_std, centroids)

    acc = float((y_pred_int == y_test).mean()) if y_test.size > 0 else 0.0

    # Baselines
    rng = np.random.RandomState(random_state)
    # Majority baseline from training distribution
    values, counts = np.unique(y_train, return_counts=True)
    majority_class = int(values[np.argmax(counts)])
    majority_pred = np.full_like(y_test, fill_value=majority_class)
    majority_acc = float((majority_pred == y_test).mean()) if y_test.size > 0 else 0.0

    # Random baseline sampled from training label prior
    probs = counts / counts.sum()
    random_pred = rng.choice(values, size=y_test.shape[0], p=probs)
    random_acc = float((random_pred == y_test).mean()) if y_test.size > 0 else 0.0
