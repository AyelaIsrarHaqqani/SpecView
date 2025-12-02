import numpy as np
from scipy.linalg import svd

def singular_spectrum_transform(signal, window=30, lag=10, n_components=10):
    """