from sko.PSO import PSO
import numpy as np
from scipy.spatial.distance import euclidean
from .sst_transform import apply_sst

def fitness_function(params, signals):
    w, k, l = map(int, params)  # window, n_components, lag
    # Enforce robust bounds before evaluation
w = int(max(10, min(w, 100)))
    k = int(max(5, min(k, 20)))
l = int(max(1, min(l, 10)))
    spectra = [apply_sst(s, w, l, k) for s in signals]
    total_dist = 0
