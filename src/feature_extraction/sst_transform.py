from src.utils.sst_custom import singular_spectrum_transform

def apply_sst(signal, window, lag, n_components):
    # Clamp parameters to valid ranges for sliding windows
    n = len(signal)
    if n <= 1:
        return []
    
    # Prefer symmetric past/future windows: cap at ~half the signal
    half_cap = max(1, (n // 2) - 1)
    window = int(max(1, min(int(window), half_cap)))
    lag = int(max(1, min(int(lag), window)))  # lag must not exceed window
    # Clamp n_components to safe range: 1..min(window, 40)
    n_components = int(max(1, min(int(n_components), int(min(window, 40)))))

    return singular_spectrum_transform(signal, window, lag, n_components)

