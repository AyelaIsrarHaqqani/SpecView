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
for i in range(len(spectra)):
        for j in range(i+1, len(spectra)):
            total_dist += euclidean(spectra[i], spectra[j])
    return total_dist

def optimize_sst_params(signals, lb, ub, particles, iterations):
    # Intersect provided bounds with realistic bounds: [10,100], [5,20], [1,10]
realistic_lb = np.array([10.0, 5.0, 1.0])
    realistic_ub = np.array([100.0, 20.0, 10.0])

 lb_arr = np.array(lb, dtype=float).reshape(3)
lb_final = np.maximum(lb_arr, realistic_lb)
    ub_final = np.minimum(ub_arr, realistic_ub)


    ub_arr = np.array(ub, dtype=float).reshape(3)
# Fix inversions by snapping to realistic bounds
    for i in range(3):
     if lb_final[i] > ub_final[i]:
            lb_final[i] = realistic_lb[i]

            ub_final[i] = realistic_ub[i]



