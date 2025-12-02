import numpy as np
from scipy.linalg import svd

def singular_spectrum_transform(signal, window=30, lag=10, n_components=10):
    """
    Custom Singular Spectrum Transformation (SST)
    Based on SpecView's spectral change-point formulation
    """
    N = len(signal)
    scores = np.zeros(N)
    # Ensure feasible parameters and symmetric windows
    if window >= N or lag > window:
        return scores
    if N - window <= window:
        return scores

    for t in range(window, N - window):
        # Build past and future matrices
        X_past = np.lib.stride_tricks.sliding_window_view(signal[t - window:t], lag)
        X_future = np.lib.stride_tricks.sliding_window_view(signal[t:t + window], lag)
