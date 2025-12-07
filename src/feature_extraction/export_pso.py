import argparse
import os
import numpy as np
from src.utils.file_loader import load_dataset
from src.utils.resampler import resample_signal
from src.config import CONFIG
from .pso_optimizer import optimize_sst_params_and_save

def main():
    parser = argparse.ArgumentParser(description="Run PSO to optimize SST params and save them to a file.")
    parser.add_argument("--data", default="malimg_dataset/train", help="Path to dataset root (label subdirs).")
    parser.add_argument("--out", default="artifacts/pso_params.json", help="Path to write JSON with best params.")
    parser.add_argument("--num", type=int, default=64, help="Number of samples to use for PSO (subset).")
    args = parser.parse_args()

    X, y = load_dataset(args.data)
    if len(X) == 0:
        raise RuntimeError(f"No samples found in {args.data}")
    
    # Resample and subset
    X_resampled = [resample_signal(x, CONFIG["resample_length"]) for x in X]
    subset = X_resampled[: max(1, min(args.num, len(X_resampled)))]
    
    # Use CONFIG pso settings
    pso_cfg = CONFIG["pso"]
    lb = pso_cfg["lb"]
    ub = pso_cfg["ub"]
    particles = pso_cfg["particles"]
    iterations = pso_cfg["iterations"]

    # Optimize and save
    best_params, save_path = optimize_sst_params_and_save(
        subset, lb=lb, ub=ub, particles=particles, iterations=iterations, save_path=args.out
    )

    w, k, l = map(int, best_params)
    print(f"Saved PSO params to {save_path}")
    print(f"Optimized SST Params -> window={w}, lag={l}, n_components={k}")

