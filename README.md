
# SpecView: Malware Spectrum Visualization Framework

**SpecView** is a universal, efficient, and accurate malware detection and classification framework. It utilizes **Singular Spectrum Transformation (SST)** to visualize malware behavior as 1D time-series spectra and employs **Particle Swarm Optimization (PSO)** to fine-tune features for Machine Learning classification.

## 📌 Overview

Traditional malware visualization methods (like binary image conversion) often fail against polymorphic, metamorphic, and packed/encrypted malware variants. SpecView addresses this by treating malware binaries as signal data, extracting deep structural and textural information that resists obfuscation.

### Key Features
* **Cross-Platform Analysis:** Effective on both Windows (PE) and Android (APK) malware.
* **Novel Feature Extraction:** Uses Singular Spectrum Transformation (SST) to generate Change Point (CP) scores.
* **Optimized Performance:** Implements Particle Swarm Optimization (PSO) to tune SST parameters (window size, order, lag).
* **High Accuracy:** Outperforms traditional static and dynamic analysis methods.

## ⚙️ Methodology

The SpecView framework follows a multi-stage pipeline:

1.  **Binary to Time-Series Conversion:** The malware binary is read as a sequence of bytes and converted into a 1D time-series signal to capture raw structural characteristics.
2.  **Signal Resampling:** The signal is resampled to ensure uniform input length across different samples.
3.  **Feature Extraction (SST):** SST analyzes the signal to detect structural changes, generating "Change Point scores" that visualize the malware's unique internal evolution (the SST Spectrum).
4.  **Parameter Optimization (PSO):** PSO fine-tunes the SST parameters to minimize intra-class variance and maximize classification accuracy.
5.  **Classification:** Extracted features are fed into ML algorithms including Random Forest, SVM, KNN, and a Voting Classifier.

## 🚀 Getting Started

### Prerequisites
* Python 3.8+
* Required libraries: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `scipy` 

### Installation

1.  Clone the repository:
    ```bash
    git clone [https://github.com/AyelaIsrarHaqqani/SpecView.git](https://github.com/AyelaIsrarHaqqani/SpecView.git)
    cd SpecView
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### Usage

1.  **Preprocessing:** Convert your dataset binaries to time-series signals.
    ```bash
    python src/preprocess.py --input_dir /path/to/malware
    ```
2.  **Feature Extraction:** Run SST to generate spectrum features.
    ```bash
    python src/sst_extraction.py
    ```
3.  **Classification:** Train and test the model.
    ```bash
    python src/train_model.py --classifier random_forest
    ```

    ## 👥 Project Team
* **Umar Tariq** (2022604)
* **Ayela Israr** (2022130)
* **M Zeeshan** (2022644)

**Course:** CY-471 Malware Analysis  
**Institution:** GIK Institute of Engineering Sciences and Technology
