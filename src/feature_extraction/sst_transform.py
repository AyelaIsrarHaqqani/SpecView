from src.utils.sst_custom import singular_spectrum_transform

def apply_sst(signal, window, lag, n_components):
    # Clamp parameters to valid ranges for sliding windows
    n = len(signal)
    if n <= 1:
        return []