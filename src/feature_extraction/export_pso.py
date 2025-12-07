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
    