# SpecView

# SpecView: Malware Spectrum Visualization Framework

**SpecView** is a universal, efficient, and accurate malware detection and classification framework. It utilizes **Singular Spectrum Transformation (SST)** to visualize malware behavior as 1D time-series spectra and employs **Particle Swarm Optimization (PSO)** to fine-tune features for Machine Learning classification.

## 📌 Overview

Traditional malware visualization methods (like binary image conversion) often fail against polymorphic, metamorphic, and packed/encrypted malware variants. SpecView addresses this by treating malware binaries as signal data, extracting deep structural and textural information that resists obfuscation.

### Key Features
* **Cross-Platform Analysis:** Effective on both Windows (PE) and Android (APK) malware.
* **Novel Feature Extraction:** Uses Singular Spectrum Transformation (SST) to generate Change Point (CP) scores.
* **Optimized Performance:** Implements Particle Swarm Optimization (PSO) to tune SST parameters (window size, order, lag).
* **High Accuracy:** Outperforms traditional static and dynamic analysis methods.
