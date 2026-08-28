# Raman Spectroscopy Preprocessing & Synthetic Data Generation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end research framework and production pipeline for **Raman spectral preprocessing**, **synthetic spectra generation**, and **deep neural network architectures**. 

This project tackles major analytical challenges in Raman spectroscopy:
- **Severe Fluorescence Baselines**: High-order polynomial, exponentially decaying, and arbitrary background interference.
- **Detector and Shot Noise**: Low signal-to-noise ratios (SNR) in low-exposure and high-throughput spectral mappings.
- **Cosmic Ray Artifacts & Dead Pixels**: Narrow, high-intensity spike contaminants.
- **Substrate Interference**: Characteristic non-linear background signals (such as second-order optical phonons in silicon substrates).
- **Arbitrary Spectral Resolutions & Bandwidths**: Dynamic, input-size agnostic neural networks capable of processing Raman spectra of any sequence length $L$ without resizing or interpolation artifacts.

---

## Table of Contents

- [Key Architecture & Innovations](#key-architecture--innovations)
  - [1. PolyGaussNet & PolyGaussNet2 (Input-Size Agnostic Preprocessing)](#1-polygaussnet--polygaussnet2-input-size-agnostic-preprocessing)
  - [2. Dual-Supervised ResNet](#2-dual-supervised-resnet)
  - [3. Synthetic Raman Spectra Generator](#3-synthetic-raman-spectra-generator)
  - [4. Classical / Conventional Preprocessing Pipeline](#4-classical--conventional-preprocessing-pipeline)
- [Project Directory Structure](#project-directory-structure)
- [Deep Dive into Project Components](#deep-dive-into-project-components)
  - [Core Model Architectures](#core-model-architectures)
  - [Data Simulation Engine](#data-simulation-engine)
  - [Conventional Preprocessing Library](#conventional-preprocessing-library)
  - [Training Rounds Evolution (Rounds 1–4)](#training-rounds-evolution-rounds-14)
  - [Validation, Unit Tests, and Diagnostic Utilities](#validation-unit-tests-and-diagnostic-utilities)
- [Installation & Environment Setup](#installation--environment-setup)
- [Usage Workflows](#usage-workflows)
  - [1. Generating Synthetic Datasets](#1-generating-synthetic-datasets)
  - [2. Training PolyGaussNet Models](#2-training-polygaussnet-models)
  - [3. Benchmarking & Evaluating Preprocessing](#3-benchmarking--evaluating-preprocessing)
  - [4. Running the Classical Preprocessing Pipeline](#4-running-the-classical-preprocessing-pipeline)
- [Loss Functions & Optimization](#loss-functions--optimization)
- [Requirements](#requirements)

---

## Key Architecture & Innovations

### 1. PolyGaussNet & PolyGaussNet2 (Input-Size Agnostic Preprocessing)

`polygaussnet.py` and `polygaussnet2.py` implement deep learning architectures that are **100% input-length agnostic**. Unlike standard fully connected networks or fixed-grid autoencoders, PolyGaussNet processes spectra with arbitrary sequence lengths $L$ (e.g., 1016, 1200, 2048 bins) directly:

```
Raw Spectrum x (Batch, 1, L)
       │
       ▼
┌────────────────────────────────────────────────────────┐
│  Stage 1: PolynomialBaselineEstimator                  │
│  - 1D CNN Feature Extractor + AdaptiveAvgPool1d(1)     │
│  - Fully Connected Head -> Polynomial Coefficients     │
│  - Horner's Method Grid Evaluation over [-1.0, 1.0]   │
└────────────────────────────────────────────────────────┘
       │
       ├─────────────────────────► Predicted Baseline B(x)
       ▼
 Baseline-Corrected Spectrum: x - B(x)
       │
       ▼
┌────────────────────────────────────────────────────────┐
│  Stage 2: FilterParameterPredictor                     │
│  - Dilated Residual Blocks (dilations: 1, 2, 4, 8)     │
│  - Preserves exact sequence length L                   │
│  - Predicts per-bin physical parameters:               │
│      • Sigma σ(x) (adaptive local bandwidth)           │
│      • [Optional in V1] Amplitude A(x) & Beta β(x)     │
└────────────────────────────────────────────────────────┘
       │
       ▼
┌────────────────────────────────────────────────────────┐
│  Stage 3: AdaptiveGaussianFilter1D                     │
│  - Differentiable per-bin Gaussian convolution         │
│  - Local energy conservation                           │
└────────────────────────────────────────────────────────┘
       │
       ▼
 Clean Raman Spectrum (Batch, 1, L)
```

- **Polynomial Baseline Estimator**: Extracts global shape representations using multi-stage 1D convolutions and `AdaptiveAvgPool1d(1)` before projecting to polynomial coefficients $(c_0, c_1, \dots, c_k)$. The baseline is evaluated in normalized Chebyshev-domain space $[-1.0, 1.0]$ using Horner's rule, ensuring numerical stability and fast execution.
- **Dilated Residual CNN**: A receptive field expansion network that maintains full resolution without downsampling or pooling.
- **Differentiable Adaptive Filter**: Performs localized, sample-specific convolutional filtering using physically constrained kernel parameters.

### 2. Dual-Supervised ResNet

`dual_supervised_resnet.py` provides a cooperative, 2-stage deep residual network:
- **Encoder Sub-Network (`BaselineNet`)**: 15 residual blocks across 5 stages with large-to-small convolution kernels ($F_i \in \{30, 15, 9, 3\}$) for global baseline extraction.
- **Decoder Sub-Network (`DenoisingNet`)**: 3 residual stages with multi-channel convolutions for noise and spike filtering.
- **Dual Supervision**: Supervised intermediate loss on baseline-corrected spectra and final loss on pure clean spectra.

### 3. Synthetic Raman Spectra Generator

`spectra_generator.py` simulates realistic Raman spectra with customizable parameters:
- **Parametric Peaks**: Randomly blends Gaussian, Lorentzian, Voigt, and Moffat profiles using `astropy.modeling`.
- **High-Order Polynomial Baselines**: Sampled over Chebyshev windows $[-1, 1]$ with controlled min-offset shifts.
- **Signal-Dependent Gaussian Noise**: Noise standard deviation dynamically scaled relative to the minimum peak amplitude.
- **Cosmic Ray Artifacts**: Poisson-distributed intensity spikes with adjustable probability and magnitude.
- **Dual/Triple Matrix Outputs**: Simultaneously outputs `(pure, pure_noise_cosmic, full)` for multi-stage supervised learning.

### 4. Classical / Conventional Preprocessing Pipeline

`test_files/conventional/pre.py` contains a modular, state-of-the-art classical Raman preprocessing toolbox:
- **Cosmic Ray & Dead Pixel Despiking**: Modified Whittaker Z-score and median absolute deviation (MAD) filtering.
- **Wavelet Denoising**: Soft BayesShrink thresholding with Coiflet wavelets (`coif3`).
- **Baseline Correction Algorithms**:
  - `arPLS`: Asymmetrically Reweighted Penalized Least Squares (Baek et al., 2015).
  - `airPLS`: Adaptive Iteratively Reweighted Penalized Least Squares (Zhang et al., 2010).
  - `modpoly` / `IModPoly`: Iterative Modified Polynomial Fitting (Zhao et al., 2007).
  - `peakutils`: Adaptive polynomial baseline fitting.
  - `combo`: Multi-stage hybrid combination (PeakUtils polynomial + ZhangFit).
- **Silicon Background Subtraction**: Non-linear curve fitting of second-order silicon optical phonon bands (800–1150 cm⁻¹) using triple-Voigt convolution profiles.
- **Decorated API (`@process_spectra_data`)**: Seamlessly accepts single `spectrum` objects, lists of spectra, or high-density `spectra` map matrices.

---

## Project Directory Structure

```text
raman-neural-network-preprocessing/
│
├── README.md                           # Comprehensive project documentation
├── requirements.txt                    # Project dependencies
├── .gitignore                          # Git exclusion rules
│
├── spectra_generator.py                # Synthetic Raman spectral simulation engine
├── polygaussnet.py                     # PolyGaussNet V1 (generalized Gaussian parameters: σ, A, β)
├── polygaussnet2.py                    # PolyGaussNet V2 (refined σ-only adaptive Gaussian filtering)
├── dual_supervised_resnet.py           # Cooperative Dual-Supervised ResNet & Log-Cosh loss
├── plot_training_progress.py           # Training curve parser & PDF report visualizer
├── polygaussnet4.pth                   # Pretrained weights for PolyGaussNet2 (Round 4)
│
├── test_files/                         # Test suites, unit tests & experimental scripts
│   ├── conventional/                   # Classical algorithmic preprocessing
│   │   ├── pre.py                      # Complete classical preprocessing pipeline
│   │   ├── despike_test.py             # Cosmic ray despiking test script
│   │   ├── despike_test2.py            # Comparative despiking evaluation
│   │   └── pipeline_test.py            # End-to-end classical pipeline test
│   │
│   ├── polygauss/                      # Deep learning architecture tests
│   │   └── amplitude_effect.py         # Amplitude scaling & energy conservation test
│   │
│   └── spectra_generation/             # Simulation module unit tests
│       ├── test_peak_generation.py     # Peak profile model tests (Gaussian/Lorentzian/Voigt/Moffat)
│       ├── test_baseline_generation.py # Polynomial baseline generator tests
│       ├── test_noise_generation.py    # Gaussian noise & cosmic ray tests
│       ├── test_composite_generation.py# Full synthetic batch generation tests
│       ├── test_parameters.py          # Generator parameter sweep & validation
│       └── shift_problem.py            # Y-shift & normalization invariant analysis
│
├── training_round1/                    # Round 1: Initial ResNet architecture & dataset design
│   ├── create_dataset1.txt             # Dataset generation specification
│   ├── training1.txt                   # Early training logs and parameters
│   └── testing1.py                     # Round 1 test and evaluation script
│
├── training_round2/                    # Round 2: Dual-supervised ResNet training
│   ├── create_dataset2.py              # Multiprocess chunked dataset creator
│   ├── training2.py                    # Dual-supervised ResNet training loop
│   ├── testing2.py                     # Round 2 validation & evaluation script
│   └── training_history.txt            # Training history & loss logs
│
├── training_round3/                    # Round 3: PolyGaussNet V1 input-agnostic model
│   ├── training3.py                    # PolyGaussNet training loop with multi-parameter filter
│   ├── testing3.py                     # Model evaluation & metrics calculation
│   ├── compare_conventional.py         # Head-to-head comparison vs. arPLS/airPLS/wavelets
│   ├── plot_real_wdf_comparison.py     # Evaluation on real experimental instrument maps (.wdf)
│   └── polygaussnet3.pth               # Pretrained weights for PolyGaussNet (Round 3)
│
├── training_round4/                    # Round 4: PolyGaussNet2 with Asymmetric Penalty
│   ├── create_dataset4.py              # High-throughput multiprocess dataset generator
│   ├── training4.py                    # PolyGaussNet2 training with asymmetric baseline loss
│   ├── testing4.py                     # Comprehensive benchmark suite & interactive viewer
│   ├── overfit_check.py                # Visual generalization & overfitting diagnostics
│   ├── trial_spectra4.py               # Single-spectrum trial and inference script
│   └── polygaussnet4.pth               # Production weights (Degree 7 + Gaussian + Asym Loss)
│
└── pdfs/                               # Design documents, architecture references & papers
    ├── preprocessing design brief.pdf  # Project requirements and theoretical foundation
    ├── techrxiv.19435718.v1.pdf        # Deep learning for Raman spectroscopy reference paper
    └── Convolutional block (n,,,.pdf   # ResNet convolutional block architectural design
```

---

## Deep Dive into Project Components

### Core Model Architectures

#### 1. `polygaussnet2.py` (Current Production Model)
- **`PolynomialBaselineEstimator(poly_order=7)`**:
  - Encodes the input spectrum into a fixed 128-dimensional embedding via strided 1D convolutions and `AdaptiveAvgPool1d(1)`.
  - Maps embeddings through multi-layer perceptrons to output $k+1$ polynomial coefficients.
  - Dynamically builds a linspace grid over $[-1.0, 1.0]$ of length $L$ and computes the baseline via Horner's scheme.
- **`FilterParameterPredictor`**:
  - Utilizes 4 dilated 1D residual blocks ($d=1, 2, 4, 8$) with GELU activations to predict per-channel physical bandwidth parameters $\sigma \in [\sigma_{\min}, \sigma_{\max}]$.
- **`AdaptiveGaussianFilter1D`**:
  - Constructs per-channel Gaussian convolution kernels dynamically and computes local convolutions via sliding window `F.unfold`.

#### 2. `polygaussnet.py` (Generalized Adaptive Gaussian Filter)
- Extends the filtering stage to predict three per-bin parameters:
  - $\sigma$ (kernel width / standard deviation)
  - $A$ (amplitude gain with global energy conservation)
  - $\beta$ (generalized Gaussian shape parameter, bridging Lorentzian, Gaussian, and Super-Gaussian profiles)

#### 3. `dual_supervised_resnet.py`
- Implements deep residual convolutional blocks with identity mappings and strided skip connections.
- Implements `LogCoshLoss` for robust regression against non-Gaussian noise and outliers.

---

### Data Simulation Engine

`spectra_generator.py` provides pure NumPy/Astropy vectorized routines for high-speed synthetic spectra synthesis:

```python
from spectra_generator import generate_spectra

pure, pure_noise_cosmic, full = generate_spectra(
    batch_size=1024,
    wavenum_range=[0, 1015],
    num_peaks_range=[0, 100],
    amplitude_range=[0.001, 0.1],
    width_range=[1, 100],
    degree_range=[1, 16],
    offset_range=[0.0, 1.0],
    max_coeff=1.0,
    min_peak_ratio=2.5,
    std_range=[1, 5],
    probability_cosmic=1 / 24000,
    intensity_range_cosmic=[5.0, 15.0],
    normalize=True
)
```

- **Vectorized Generation**: Generates batches of spectra directly as 2D NumPy arrays `(batch_size, bins)`.
- **Multi-Level Supervision**: Returns ground-truth components for both intermediate baseline subtraction and final denoised spectral reconstruction.

---

### Conventional Preprocessing Library

`test_files/conventional/pre.py` contains standard algorithmic methods for Raman processing:

| Function | Method | Description |
| :--- | :--- | :--- |
| `despike_spectra` | Modified Whittaker Z-Score | Detects narrow outliers/spikes and replaces them using valid local neighbors |
| `denoise_spectra` | BayesShrink Wavelet | Deconstructs spectra into wavelet domains (`coif3`) with soft thresholding |
| `remove_baseline` | `arPLS` / `airPLS` / `modpoly` / `combo` | Baseline subtraction via penalized least squares or iterative polynomial fitting |
| `background_removal_silicon` | Multi-Voigt Curve Fitting | Removes 2nd-order silicon bands (800–1150 cm⁻¹) via non-linear optimization |
| `normalize_spectra` | Standard Z-score | Normalizes intensity distributions per spectrum |
| `shift_to_zero` | Minimum Intensity Shifting | Non-negative constraint shift setting $\min(I) = 0$ |
| `preprocess_pipeline` | Composite Pipeline | Configurable sequence executing all above preprocessing stages |

---

### Training Rounds Evolution (Rounds 1–4)

```
┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│ Training Rd. 1  │  ──►  │ Training Rd. 2  │  ──►  │ Training Rd. 3  │  ──►  │ Training Rd. 4  │
├─────────────────┤       ├─────────────────┤       ├─────────────────┤       ├─────────────────┤
│ Fixed-size ResNet│      │ Dual-Supervised │       │ PolyGaussNet V1 │       │ PolyGaussNet2   │
│ Baseline study  │       │ Encoder-Decoder │       │ Input Agnostic  │       │ Asymmetric Loss │
│ Initial dataset │       │ Log-Cosh loss   │       │ Multi-parameter │       │ Degree 7 Poly   │
└─────────────────┘       └─────────────────┘       └─────────────────┘       └─────────────────┘
```

1. **Round 1 (`training_round1/`)**: Explored single-stage 1D convolutional ResNets for baseline subtraction on fixed-length spectra.
2. **Round 2 (`training_round2/`)**: Introduced dual-supervision (`BaselineNet` + `DenoisingNet`) trained with `LogCoshLoss` on chunked `.npz` datasets.
3. **Round 3 (`training_round3/`)**: Developed the first fully input-size agnostic `PolyGaussNet` architecture with adaptive generalized Gaussian filtering, benchmarked against classical arPLS and airPLS on real `.wdf` Raman map data.
4. **Round 4 (`training_round4/`)**: Engineered `PolyGaussNet2` featuring:
   - Degree 7 polynomial baseline head for optimal stiffness.
   - Fixed Gaussian filter ($\text{exponent}=2$, unit amplitude) with adaptive $\sigma$ prediction.
   - **Asymmetric Baseline Penalty Loss**: Heavily penalizes baseline overprediction ($\hat{B}(x) > X_{\text{raw}}(x)$) to prevent peak clipping.
   - Checkpoint pause/resume, multi-threaded `.npz` streaming, and interactive diagnostic slideshow viewers.

---

### Validation, Unit Tests, and Diagnostic Utilities

The repository includes a comprehensive set of test suites under `test_files/`:
- `test_peak_generation.py`: Verifies mathematical integrity, amplitude bounds, and FWHM sampling across Gaussian, Lorentzian, Voigt, and Moffat profiles.
- `test_baseline_generation.py`: Checks high-order polynomial sampling, Chebyshev domain mappings, and offset constraints.
- `test_noise_generation.py`: Validates Gaussian noise scaling relative to minimum peak intensities and Poisson cosmic ray frequencies.
- `amplitude_effect.py`: Evaluates energy conservation and scaling properties of the adaptive filtering layers.
- `testing4.py`: Comprehensive benchmarking script comparing PolyGaussNet2 vs. the classical pipeline across Cosine Similarity, MSE, MAE, RMSE, throughput (spectra/sec), and latency.

---

## Installation & Environment Setup

### Prerequisites
- Python 3.10 or higher
- Git
- PyTorch 2.0+ (supports Apple Silicon MPS, NVIDIA CUDA, or CPU)

### Setup Instructions

```bash
# 1. Clone the repository
git clone https://github.com/bogomilkissiov/Raman-Neural-Network-Preprocessing.git
cd Raman-Neural-Network-Preprocessing

# 2. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install required dependencies
pip install -r requirements.txt

# (Optional for local development of Raman-Plotting-Pipeline):
# pip install -e "../Raman-Plotting-Pipeline"
```

---

## Usage Workflows

### 1. Generating Synthetic Datasets

Generate large-scale, chunked training datasets using multi-core multiprocessing:

```bash
python training_round4/create_dataset4.py
```
This produces `.npz` chunk files containing `pure_matrix`, `pure_noise_cosmic_matrix`, and `full_matrix`.

### 2. Training PolyGaussNet Models

Train PolyGaussNet2 with automatic device detection (MPS / CUDA / CPU), learning rate scheduling, and pause/resume support:

```bash
# Train from training_round4 with custom parameters
python training_round4/training4.py --epochs 160 --batch-size 512 --lr 1e-4 --poly-order 7
```

- **Pause & Resume**: Press `Ctrl+C` at any time to safely save `training_checkpoint4.pth`. Re-running the command automatically resumes training from the last saved epoch.

### 3. Benchmarking & Evaluating Preprocessing

Run the head-to-head benchmarking suite comparing PolyGaussNet2 against classical pipelines:

```bash
# Run quantitative benchmark (Cosine Similarity, MSE, Throughput)
python training_round4/testing4.py

# Launch interactive 4-stage residual and decomposition slideshow
python training_round4/testing4.py --residual-slideshow

# Launch interactive 4-panel spectral comparison viewer
python training_round4/testing4.py --interactive
```

### 4. Running the Classical Preprocessing Pipeline

```python
import numpy as np
from test_files.conventional.pre import preprocess_pipeline

# Example: Process a batch of raw Raman spectra (shape: N x Bins)
raw_spectra = np.load("test_data.npz")["full_matrix"]

# Run complete classical pipeline (despike, denoise, baseline removal, normalize, shift)
cleaned_spectra = preprocess_pipeline(
    raw_spectra,
    despike=True,
    denoise=True,
    baseline=True,
    normalize=True,
    shift=True
)
```

---

## Loss Functions & Optimization

PolyGaussNet is trained with a composite multi-objective loss:

$$\mathcal{L}_{\text{total}} = \lambda_{\text{BC}} \mathcal{L}_{\text{LogCosh}}(X_{\text{BC}}, \hat{X}_{\text{BC}}) + \lambda_{\text{Clean}} \mathcal{L}_{\text{LogCosh}}(X_{\text{pure}}, \hat{X}_{\text{clean}}) + \lambda_{\text{asym}} \mathcal{L}_{\text{asym}}$$

Where:
- **Log-Cosh Loss**: A smooth, robust surrogate for Mean Absolute Error (MAE):
  $$\mathcal{L}_{\text{LogCosh}}(y, \hat{y}) = \frac{1}{N}\sum_{i=1}^N \ln\left(\cosh(\hat{y}_i - y_i)\right)$$
- **Asymmetric Baseline Penalty**: Penalizes any instance where the estimated baseline $\hat{B}(x)$ exceeds the raw spectral signal $X_{\text{raw}}(x)$, preventing synthetic peak attenuation:
  $$\mathcal{L}_{\text{asym}} = \frac{1}{N} \sum_{i} \left(\max(0, \hat{B}_i - X_{\text{raw}, i})\right)^2$$

---

## Requirements

Core dependencies specified in `requirements.txt`:
- `numpy >= 1.24.0`
- `torch >= 2.0.0`
- `scipy >= 1.10.0`
- `astropy >= 5.0`
- `matplotlib >= 3.7.0`
- `seaborn >= 0.12.0`
- `scikit-image >= 0.20.0`
- `PyWavelets >= 1.4.0`
- `peakutils >= 1.3.0`
- `BaselineRemoval >= 0.1.6`
- `Raman-Plotting-Pipeline` (installed directly from GitHub repository)
