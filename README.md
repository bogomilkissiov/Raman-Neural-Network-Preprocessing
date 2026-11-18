# Raman Neural Network Preprocessing (`PolyGaussNet`)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end research codebase for **physics-informed, input-length-agnostic Raman spectral preprocessing** (baseline subtraction, denoising, and despiking), **synthetic Raman spectra simulation**, **classical baseline/wavelet benchmarking**, and **downstream real-world spectral classification**.

> **TL;DR — Current Production Model (`PolyGaussNet V3.4-8sec`):**  
> Implemented in [`polygaussnet3_4.py`](polygaussnet3_4.py) with `num_quadrants=8` (**108,394 parameters**), trained for 97 epochs in **Round 12** (`training_round12/`):
> - **−60.32% Raw MSE** (`0.00351` vs `0.00885`) and **−65.55% Baseline-Shifted MSE** (`0.00100` vs `0.00291`) relative to the Round 6 `PolyGaussNet V3` baseline (`148,426` params) on `N = 20,480` held-out multi-resolution spectra.
> - **−60.55% Raw MSE** and **51.0× faster** (`20,505.5` vs `402.0` spectra/s) than the conventional CPU pipeline (`test_files/conventional/pre.py`).
> - Improves downstream classification accuracy across **all 3 classifier families** on real-world **Bacteria-ID** spectra (`Linear`: **52.50%**, `XGBoost`: **46.93%**, `ResNet1D`: **60.63%**, McNemar $p \le 7.15 \times 10^{-4}$ vs both Raw and Conventional) while running preprocessing **30.5× faster** than the conventional pipeline.

---

## Table of Contents

- [Repository Map](#repository-map)
- [How PolyGaussNet V3.4 Works](#how-polygaussnet-v34-works)
- [Model Architecture Evolution (V1 → V3.4)](#model-architecture-evolution-v1--v34)
- [Training Rounds & Experimental Results](#training-rounds--experimental-results)
  - [Summary of Rounds 1–8](#summary-of-rounds-18)
  - [Rounds 9–10: Spatial Pooling & Length-Scalar Ablation](#rounds-910-spatial-pooling--length-scalar-ablation)
  - [Round 11: 4-Section vs. 8-Section Spatial Pooling (10 Seeds)](#round-11-4-section-vs-8-section-spatial-pooling-10-seeds)
  - [Round 12: Full-Scale Training & Head-to-Head Benchmark](#round-12-full-scale-training--head-to-head-benchmark)
  - [Round 12 Downstream Benchmark: Real-World Raman Classification](#round-12-downstream-benchmark-real-world-raman-classification)
- [Synthetic Spectra Engine & Conventional Baseline](#synthetic-spectra-engine--conventional-baseline)
- [Installation & Quickstart](#installation--quickstart)
- [Loss Functions](#loss-functions)
- [Requirements](#requirements)

---

## Repository Map

```text
Raman-Neural-Network-Preprocessing/
├── README.md                           # Project overview, architecture guide & benchmark results
├── requirements.txt                    # Python dependencies
│
├── polygaussnet3_4.py                  # ★ Production architecture: PolyGaussNet V3.4 (num_quadrants=8 default in R11/R12)
├── polygaussnet3.py                    # V3.0 (degree-7 polynomial + 31-bin Gaussian + bounded amplitude)
├── polygaussnet3_1.py                  # V3.1 (32-ch baseline trunk + 3-layer 31-bin filter + length scalar)
├── polygaussnet3_2.py                  # V3.2 (128-ch trunk + contracting funnel + length scalar)
├── polygaussnet3_3.py                  # V3.3 (32-ch trunk + 4-quadrant pooling + funnel + length scalar)
├── polygaussnet3_3_nolen.py            # V3.3 ablation without length scalar (identical topology to V3.4-4sec)
├── polygaussnet3_len.py                # V3.0 + length scalar ablation
├── polygaussnet3+.py                   # Round 5 experimental V3+ (wider kernel / extra filter capacity)
├── polygaussnet2.py                    # V2 (unit-amplitude σ-only Gaussian filter)
├── polygaussnet.py                     # V1 (generalized Gaussian filter: σ, A, β)
├── dual_supervised_resnet.py           # Early fixed-size Dual-Supervised ResNet & LogCoshLoss
│
├── spectra_generator.py                # Vectorized synthetic Raman spectra generator
├── plot_training_progress.py           # Training curve parser & plot utility
│
├── classifiers/                        # Shared downstream classification benchmark suite
│   ├── README.md                       # Classifier specifications & metrics overview
│   ├── linear.py                       # LinearSpectralClassifier (StandardScaler + LogisticRegression)
│   ├── xgb.py                          # XGBSpectralClassifier (XGBoost histogram trees + early stopping)
│   ├── resnet1d.py                     # ResNet1D / BacteriaResNetClassifier (Ho et al., Nat. Commun. 2019)
│   └── benchmark.py                    # Shared runner, McNemar tests & report/plot exporter
│
├── test_files/
│   ├── conventional/pre.py             # Classical pipeline (Whittaker despiking, BayesShrink wavelets, arPLS/airPLS/modpoly)
│   ├── polygauss/                      # Filter energy-conservation tests
│   └── spectra_generation/             # Unit tests for peaks, baselines, noise & cosmic rays
│
├── training_round1/ ... training_round8/  # Early explorations, V1–V3.1 benchmarks & saved .pth checkpoints (R3–R7)
├── training_round9/                    # V3.0 vs V3.1 vs V3.2 vs V3.3 multi-seed benchmark (3 seeds, 16 epochs)
├── training_round10/                   # 5-seed fresh-dataset validation & V3.3 length-scalar ablation (25 runs)
├── training_round11/                   # 10-seed statistical proof: V3.4-8sec vs V3.4-4sec vs V3.0 (30 runs)
└── training_round12/                   # Full 97-epoch V3.4-8sec training, OSCAR runbook & downstream benchmarks
    ├── train_round12.py                # High-throughput CUDA/MPS/CPU trainer (RAM/VRAM dataset caching)
    ├── testing12.py                    # 3-way synthetic benchmark (Conventional vs V3 vs V3.4-8sec)
    ├── train_downstream12.py           # 18-run downstream benchmark trainer (Bacteria-ID & MLROD)
    ├── test_downstream12.py            # Held-out downstream evaluation, McNemar significance & plots
    ├── submit_round12.sh               # Slurm job script for Brown OSCAR GPU cluster
    ├── submit_downstream_train12.sh    # Slurm job script for downstream classification suite
    ├── OSCAR_RUNBOOK.md                # Step-by-step OSCAR cluster deployment & sync guide
    ├── round12_summary.txt             # Round 12 training & synthetic benchmark summary
    ├── validation_results_round12.txt  # Full per-resolution synthetic benchmark report
    └── downstream_results/             # Held-out Bacteria-ID & MLROD CSV/JSON/TXT reports & plots
```

> **Note on large files:** Raw synthetic datasets (`training_data*`, `testing_data*`, `*.npz`), public downstream datasets (`datasets/bacteria-ID/`, `datasets/MLROD/`), cached preprocessed splits (`training_round12/downstream_cache/`), and large checkpoints outside Rounds 3–7 are excluded via `.gitignore`. Tracked weights exist in `training_round3/`–`training_round7/`, while Round 12 scripts (`training_round12/train_round12.py` and `OSCAR_RUNBOOK.md`) reproduce `polygaussnet12.pth` from `training_round6/training_data6`.

---

## How PolyGaussNet V3.4 Works

Unlike standard autoencoders or MLPs tied to a fixed spectral grid, **`PolyGaussNet V3.4`** (`polygaussnet3_4.py`) processes raw spectra of **arbitrary bin length $L$** (tested across 800–1,836+ bins) in three physically interpretable, end-to-end differentiable stages:

```text
Raw Spectrum x  (Batch, 1, L)
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Stage 1: PolynomialBaselineEstimator (num_quadrants=8, 32-ch trunk)     │
│  • 4× strided stem (1 → 32 → 32) + 3 DilatedResidualBlock1D (d=1,2,4)   │
│    → 251-bin receptive field                                            │
│  • AdaptiveAvgPool1d(8) → octant spatial anchors across [-1.0, 1.0]     │
│  • Contracting FC funnel (256 → 64 → 8) → polynomial coeffs c₀..c₇      │
│  • Horner's scheme evaluation on linspace(-1.0, 1.0, L)                 │
└─────────────────────────────────────────────────────────────────────────┘
       │
       ├──────────────────────────────────────► Predicted Baseline B(x)
       ▼
 Baseline-Corrected Spectrum:  x_BC = x - B(x)
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Stage 2: FilterParameterPredictor (3-layer stride-1 CNN, 31-bin RF)     │
│  • Conv1d(1→32, k=15) → Conv1d(32→64, k=11) → Conv1d(64→64, k=7)        │
│  • Predicts per-bin bounded physical parameters for every bin i ∈ [0,L):│
│      σᵢ ∈ [0.2, 6.0]    (adaptive local smoothing width)                │
│      Aᵢ ∈ [0.85, 1.15]  (bounded local amplitude modulation)            │
└─────────────────────────────────────────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ Stage 3: AdaptiveGaussianFilter1D (kernel_size=31)                      │
│  • Per-bin normalized Gaussian kernel weights from σᵢ via F.unfold      │
│  • Output: smoothed(x_BC) * A                                           │
└─────────────────────────────────────────────────────────────────────────┘
       │
       ▼
 Clean Raman Spectrum  (Batch, 1, L)
```

### Why V3.4 Works So Well
1. **Spatial Anchoring Aligned to Polynomial Degree:** Global 1-bucket pooling (`AdaptiveAvgPool1d(1)` in V3.0–V3.2) averages away left-vs-right slope and center-vs-edge curvature. Partitioning the spectrum into **8 spatial octants** (`AdaptiveAvgPool1d(8)`) provides 8 localized anchors across $[-1, 1]$—matching the 8 degrees of freedom of a 7th-order polynomial.
2. **True Scale Invariance Without Length Injection:** Ablations in Rounds 8 and 10 proved that concatenating $L / 1600$ to the FC head is unnecessary once spatial pooling preserves relative geometry across $[-1, 1]$.
3. **Clean Contracting Funnel:** Replacing the `33 -> 128 -> 64 -> 8` expansion bottleneck from V3.1 with `256 -> 64 -> 8` cut parameter count by **27.0%** vs V3.0 (`108,394` vs `148,426`) and sharply reduced across-seed variance.
4. **Targeted Local Filtering:** A lightweight 3-layer stride-1 CNN with a **31-bin receptive field** matches physical Raman peak widths and the 31-bin adaptive Gaussian kernel, avoiding vanishing gradients and overfitting.

---

## Model Architecture Evolution (V1 → V3.4)

| Model | File | Baseline Trunk & Pool | Baseline FC Head | Length Scalar | Filter Stage | Params | Status |
| :--- | :--- | :--- | :--- | :---: | :--- | :---: | :--- |
| **DualResNet** | `dual_supervised_resnet.py` | 15-block encoder | Fixed-grid decoder | No | 3-stage ResNet | — | Legacy (R1–R2) |
| **PolyGaussNet V1** | `polygaussnet.py` | 128-ch conv + `Pool(1)` | `128→128→64→16` | No | Generalized Gaussian ($\sigma, A, \beta$) | — | Legacy (R3) |
| **PolyGaussNet V2** | `polygaussnet2.py` | 128-ch conv + `Pool(1)` | `128→128→64→8` | No | Gaussian ($\sigma$ only, unit $A$) | — | Legacy (R4) |
| **PolyGaussNet V3.0** | `polygaussnet3.py` | 128-ch conv + `Pool(1)` | `128→128→64→8` | No | 4-block dilated ResNet ($\sigma, A$) | 148,426 | Baseline (R6–R12) |
| **PolyGaussNet V3.1** | `polygaussnet3_1.py` | 32-ch + 3 dilated blocks + `Pool(1)` | `33→128→64→8` | Yes | 3-layer 31-bin CNN ($\sigma, A$) | 104,554 | Superseded (R7–R10) |
| **PolyGaussNet V3.2** | `polygaussnet3_2.py` | 128-ch + 3 dilated blocks + `Pool(1)` | `129→64→8` | Yes | 3-layer 31-bin CNN ($\sigma, A$) | 104,746 | Superseded (R9–R10) |
| **PolyGaussNet V3.3** | `polygaussnet3_3.py` | 32-ch + 3 dilated blocks + `Pool(4)` | `129→64→8` | Yes | 3-layer 31-bin CNN ($\sigma, A$) | 100,266 | Ablated (R9–R10) |
| **PolyGaussNet V3.4 (4-sec)** | `polygaussnet3_4.py` | 32-ch + 3 dilated blocks + `Pool(4)` | `128→64→8` | No | 3-layer 31-bin CNN ($\sigma, A$) | 100,202 | Lightweight (R11) |
| **PolyGaussNet V3.4 (8-sec)** | `polygaussnet3_4.py` | 32-ch + 3 dilated blocks + `Pool(8)` | `256→64→8` | No | 3-layer 31-bin CNN ($\sigma, A$) | **108,394** | **Production Default (R11–R12)** |

---

## Training Rounds & Experimental Results

### Summary of Rounds 1–8

| Rounds | Focus & Key Milestones |
| :--- | :--- |
| **Rounds 1–2** | Fixed-length ResNet exploration and `DualSupervisedResNet` trained with `LogCoshLoss` on chunked `.npz` datasets. |
| **Rounds 3–4** | Introduced input-length-agnostic `PolyGaussNet` V1/V2, degree-7 Horner baseline estimation, asymmetric baseline overshoot penalty ($\lambda_{\text{asym}} = 15.0$), and real Renishaw `.wdf` map comparisons (`training_round3/plot_real_wdf_comparison.py`). |
| **Rounds 5–6** | Scaled training to **655,360 spectra** across 5 bin resolutions (`800, 1016, 1200, 1400, 1600`); established **dual evaluation** (Raw + `mean`-aligned Baseline-Shifted shape fidelity) in `training_round6/testing6.py`. At 97 epochs, `PolyGaussNet V3` (`polygaussnet6.pth`) matched or slightly beat Conventional MSE (`0.00885` vs `0.00891` raw; `0.00291` vs `0.00302` shifted) at **5.6×** speedup. |
| **Rounds 7–8** | Introduced `V3.1` (32-ch 251-bin RF baseline trunk + 3-layer 31-bin filter), cutting MSE by ~29% at 104,554 params, but multi-seed testing (seeds `42, 101, 202`) exposed 10× higher variance caused by `AdaptiveAvgPool1d(1)` spatial blindness and the `33→128` FC expansion. Length-scalar ablation (`V3-Len`) showed `<3%` effect (`training_round8/architectural_findings_summary.txt`). |

---

### Rounds 9–10: Spatial Pooling & Length-Scalar Ablation

Rounds 9 and 10 tested the architectural diagnoses from Round 8 across **16-epoch** multi-seed runs with identical deterministic batch sequences:
- **Round 9 (`training_round9/`):** 3 seeds (`203, 204, 205`) on `testing_data7` (`5,120` spectra).
- **Round 10 (`training_round10/`):** 5 fresh unseen seeds (`300–304`) on a newly generated dataset (`40,960` train / `10,240` held-out test spectra across 5 resolutions) plus a paired length-scalar ablation (`V3.3` vs `V3.3-NoLen`).

**Round 10 Aggregate Held-Out Results (`N = 10,240` spectra, 5 seeds, 16 epochs — [`validation_results_round10.txt`](training_round10/validation_results_round10.txt)):**

| Model | Params | Raw Pure MSE ↓ | Baseline-Shifted MSE ↓ | Baseline Est. MSE ↓ | Cosine Similarity ↑ | Δ Raw MSE vs V3.0 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `V3.0` | 148,426 | `0.02565 ± 0.00285` | `0.00931 ± 0.00028` | `0.04403 ± 0.00477` | `0.77144 ± 0.00743` | — |
| `V3.1` | 104,554 | `0.02124 ± 0.00399` | `0.00641 ± 0.00038` | `0.03466 ± 0.00639` | `0.78928 ± 0.00589` | −16.71% ± 16.40% |
| `V3.2` | 104,746 | `0.01910 ± 0.00499` | `0.00542 ± 0.00035` | `0.03127 ± 0.00827` | `0.79440 ± 0.01138` | −24.12% ± 26.23% |
| **`V3.3`** | 100,266 | **`0.00748 ± 0.00068`** | `0.00223 ± 0.00006` | **`0.01217 ± 0.00111`** | `0.83112 ± 0.00236` | **−70.53% ± 4.20%** |
| **`V3.3-NoLen`** | 100,202 | `0.00765 ± 0.00080` | **`0.00220 ± 0.00012`** | `0.01243 ± 0.00119` | **`0.83243 ± 0.00234`** | **−69.60% ± 6.63%** |

**Key Takeaways:**
- Switching from 1-bucket pooling to **4-quadrant spatial pooling + contracting funnel** (`V3.3`) reduced Raw MSE by **~70%** and Baseline-Shifted MSE by **~76%** over `V3.0`, while shrinking cross-seed standard deviation by **4×** (`0.00285 → 0.00068`).
- Removing the length scalar (`V3.3-NoLen` vs `V3.3`) yielded **4/5** seed wins on Baseline-Shifted MSE (`−1.37%`) and **3/5** on Cosine Similarity (`+0.13p`), proving spatial pooling alone achieves full length invariance → formalized as **`PolyGaussNet V3.4`**.

---

### Round 11: 4-Section vs. 8-Section Spatial Pooling (10 Seeds)

Round 11 (`training_round11/`) ran **30 paired trainings** (**3 models × 10 seeds: `400–409`**, 16 epochs each on `training_data10` / `testing_data10`) to test whether doubling spatial partitions from `num_quadrants=4` to `num_quadrants=8` better resolves 7th-degree polynomial inflection points ([`results_analysis.txt`](training_round11/results_analysis.txt)):

| Model | Params | Raw Pure MSE ↓ | Baseline-Shifted MSE ↓ | Baseline Est. MSE ↓ | Cosine Similarity ↑ |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `V3.0` (Reference) | 148,426 | `0.027585 ± 0.006187` | `0.009340 ± 0.000512` | `0.046876 ± 0.010068` | `0.773029 ± 0.007012` |
| `V3.4 (4-section)` | 100,202 | `0.008065 ± 0.000937` | `0.002274 ± 0.000297` | `0.013206 ± 0.001440` | `0.831854 ± 0.006194` |
| **`V3.4 (8-section)`** | **108,394** | **`0.007480 ± 0.000686`** | **`0.002070 ± 0.000178`** | **`0.012292 ± 0.001173`** | **`0.836196 ± 0.005338`** |

**Paired Significance (`V3.4-8sec` vs `V3.4-4sec`, $n=10$ seeds):**
- **Raw Pure MSE:** **−6.61%** (95% CI `[-13.17%, -0.05%]`, $p = 0.0485$, **8/10** wins)
- **Baseline-Shifted MSE:** **−8.24%** (95% CI `[-14.48%, -2.00%]`, $p = 0.0153$, **8/10** wins)
- **Baseline Est. MSE:** **−6.42%** (95% CI `[-12.62%, -0.22%]`, $p = 0.0436$, **8/10** wins)
- **Cosine Similarity:** **+0.43p** (95% CI `[+0.13p, +0.73p]`, $p = 0.0104$, **8/10** wins)
- Across-seed standard deviation dropped by **−26.8%** (Raw MSE) and **−40.1%** (Shifted MSE), and `V3.4-8sec` swept `V3.0` **10/10** seeds (**−71.66%** Raw MSE, **−77.72%** Shifted MSE).

---

### Round 12: Full-Scale Training & Head-to-Head Benchmark

Round 12 (`training_round12/`) trained **`PolyGaussNet V3.4-8sec`** (`108,394` params) under the exact **97-epoch** Round 6 regime (`655,360` spectra across 40 chunks; batch size `512`; `AdamW lr=1e-4`; `ReduceLROnPlateau`) on Brown University's OSCAR GPU cluster (Slurm Job `9146833`, `02:25:45` wall time, final best training loss **`0.006084`** vs `V3` Round 6's `0.012257`).

Evaluated on **`N = 20,480` held-out spectra** (`training_round6/testing_data6`: 5 resolutions × 4,096 spectra) on an **NVIDIA A40** ([`round12_summary.txt`](training_round12/round12_summary.txt), [`validation_results_round12.txt`](training_round12/validation_results_round12.txt)):

#### Part A — Raw (Unshifted) Reconstruction Metrics

| Metric | Conventional (`pre.py`) | `V3` (Round 6) | **`V3.4-8sec` (Round 12)** | Δ vs `V3` | Δ vs Conventional |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Cosine Similarity (↑)** | `0.84364` | `0.83406` | **`0.88076`** | **+4.67p** | **+3.71p** |
| **MSE (↓)** | `0.00891` | `0.00885` | **`0.00351`** | **−60.32%** | **−60.55%** |
| **Log-Cosh Loss (↓)** | `0.00441` | `0.00440` | **`0.00175`** | **−60.28%** | **−60.37%** |
| **MAE (↓)** | `0.06329` | `0.06979` | **`0.03722`** | **−46.66%** | **−41.19%** |
| **RMSE (↓)** | `0.07603` | `0.08154` | **`0.04353`** | **−46.61%** | **−42.74%** |
| **Peak Summit MAE (↓)** | `0.06344` | `0.05625` | **`0.03590`** | **−36.17%** | **−43.40%** |
| **Baseline Est. MSE (↓)** | N/A | `0.01737` | **`0.00672`** | **−61.29%** | N/A |

#### Part B — Baseline-Shifted (`mean`-Aligned) Shape Fidelity & Throughput

| Metric | Conventional (`pre.py`) | `V3` (Round 6) | **`V3.4-8sec` (Round 12)** | Δ vs `V3` |
| :--- | :---: | :---: | :---: | :---: |
| **Shifted Cosine Similarity (↑)** | `0.76780` | `0.76092` | **`0.86093`** | **+10.00p** |
| **Shifted MSE (↓)** | `0.00302` | `0.00291` | **`0.00100`** | **−65.55%** |
| **Shifted Peak Summit MAE (↓)** | `0.04024` | `0.04289` | **`0.02545`** | **−40.68%** |
| **Inference Time (`N=20,480`)** | `50.942 s` (8 CPU workers) | `1.267 s` | **`0.999 s`** | — |
| **Throughput (spectra/s)** | `402.0` | `16,168.6` | **`20,505.5`** (**51.0× vs Conv**) | **+26.8%** |

**Per-Resolution Win Counts (`800, 1016, 1200, 1400, 1600` bins):** `V3.4-8sec` won **5/5** resolutions on Raw MSE, Shifted MSE, and Raw Cosine against both `V3` and `Conventional`.

---

### Round 12 Downstream Benchmark: Real-World Raman Classification

To verify that synthetic preprocessing gains translate to real experimental spectroscopy, [`classifiers/`](classifiers/README.md) and [`training_round12/train_downstream12.py`](training_round12/train_downstream12.py) / [`test_downstream12.py`](training_round12/test_downstream12.py) benchmarked **Raw (`[0, 1]` min-max)** vs **Conventional (`pre.py`)** vs **`PolyGaussNet V3.4-8sec`** across three distinct classifier families—`LinearSpectralClassifier`, `XGBSpectralClassifier`, and `ResNet1D` (Ho et al., 2019)—under matched epoch budgets ([`downstream_results_summary12.txt`](training_round12/downstream_results/downstream_results_summary12.txt)).

#### 1. Bacteria-ID (Ho et al., *Nature Communications* 2019)
*30 bacterial isolate classes, 1,000 bins (`train=60,000`, `val=3,000`, `test=3,000`; clinical splits excluded)*

| Classifier | Preprocessor | Test Acc (Best Val Ckpt) ↑ | Macro F1 ↑ | Top-3 Acc ↑ | McNemar $p$-value vs Raw / Conv (`V3.4`) |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Linear** (`LogisticRegression`) | Raw | `50.33%` | `0.4309` | `70.10%` | — |
| **Linear** (`LogisticRegression`) | Conventional | `48.97%` | `0.4279` | `72.10%` | — |
| **Linear** (`LogisticRegression`) | **`PolyGaussNet V3.4-8sec`** | **`52.50%`** | **`0.4562`** | **`72.23%`** | $6.85 \times 10^{-4}$ / $3.16 \times 10^{-9}$ (`***`) |
| **XGBoost** (`hist`, 128 trees) | Raw | `33.73%` | `0.2875` | `55.47%` | — |
| **XGBoost** (`hist`, 128 trees) | Conventional | `39.27%` | `0.3344` | `63.13%` | — |
| **XGBoost** (`hist`, 128 trees) | **`PolyGaussNet V3.4-8sec`** | **`46.93%`** | **`0.4141`** | **`69.40%`** | $1.52 \times 10^{-52}$ / $5.51 \times 10^{-22}$ (`***`) |
| **ResNet1D** (14 epochs) | Raw | `57.50%` | `0.5188` | `76.20%` | — |
| **ResNet1D** (14 epochs) | Conventional | `58.23%` | `0.5236` | `78.83%` | — |
| **ResNet1D** (14 epochs) | **`PolyGaussNet V3.4-8sec`** | **`60.63%`** | **`0.5522`** | **`81.23%`** | $4.19 \times 10^{-5}$ / $7.15 \times 10^{-4}$ (`***`) |

- **Preprocessing Speed (66,000 spectra):** `PolyGaussNet V3.4-8sec` finished in **`3.01 s`** (`21,916.6` spec/s) vs **`91.92 s`** (`718.0` spec/s) for Conventional (**30.5× faster**).

#### 2. MLROD (Berlanga et al., 2022)
*12 pure single-mineral classes, 1,836 bins (`train=71,275`, `val=16,449`, `test=21,931`; stratified 65/15/20)*

| Classifier | Preprocessor | Test Acc (Best Val Ckpt) ↑ | Macro F1 ↑ | Top-3 Acc ↑ |
| :--- | :--- | :---: | :---: | :---: |
| **Linear** | Raw / Conventional / `V3.4-8sec` | `97.67%` / **`98.16%`** / `97.69%` | `0.9784` / **`0.9829`** / `0.9780` | `99.57%` / **`99.64%`** / `99.57%` |
| **XGBoost** | Raw / Conventional / `V3.4-8sec` | **`99.14%`** / `99.04%` / `99.12%` | **`0.9923`** / `0.9916` / `0.9922` | `99.90%` / **`99.92%`** / `99.90%` |
| **ResNet1D** | Raw / Conventional / `V3.4-8sec` | **`99.33%`** / `99.14%` / `99.08%` | **`0.9941`** / `0.9924` / `0.9919` | **`99.95%`** / `99.93%` / `99.93%` |

- On clean, high-SNR mineral spectra (`MLROD`), all methods saturate near ~98–99.3% accuracy, while **`V3.4-8sec`** preprocessed all **109,655 spectra in `6.78 s`** (`16,170.5` spec/s) compared to **`267.38 s`** (`410.1` spec/s) for Conventional (**39.4× faster**).

---

## Synthetic Spectra Engine & Conventional Baseline

- **Synthetic Data Engine ([`spectra_generator.py`](spectra_generator.py)):** Vectorized NumPy/Astropy generator producing `(pure, pure_noise_cosmic, full)` tuples with randomized Gaussian/Lorentzian/Voigt/Moffat peaks, Chebyshev-domain polynomial baselines (degree up to 16), signal-dependent noise, and Poisson cosmic-ray spikes.
- **Conventional Preprocessing ([`test_files/conventional/pre.py`](test_files/conventional/pre.py)):** Modular classical pipeline implementing modified Whittaker Z-score despiking (`despike_spectra`), `coif3` BayesShrink wavelet denoising (`denoise_spectra`), `arPLS`/`airPLS`/`modpoly`/`combo` baseline removal (`remove_baseline`), triple-Voigt silicon background subtraction (`background_removal_silicon`), and composite `preprocess_pipeline`.

---

## Installation & Quickstart

### 1. Environment Setup

```bash
git clone https://github.com/bogomilkissiov/Raman-Neural-Network-Preprocessing.git
cd Raman-Neural-Network-Preprocessing

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Optional: needed for downstream classifier benchmarks
pip install scikit-learn xgboost joblib
```

### 2. Inference with `PolyGaussNet V3.4`

```python
import torch
from polygaussnet3_4 import PolyGaussNet

model = PolyGaussNet(
    poly_order=7,
    filter_kernel_size=31,
    min_amplitude=0.85,
    max_amplitude=1.15,
    num_quadrants=8,  # Round 11/12 production configuration (108,394 params)
)
# Accepts (Batch, L) or (Batch, 1, L) for arbitrary spectral length L
raw = torch.rand(16, 1, 1000)
clean, pred_baseline, baseline_corrected, latent = model(raw)
```

### 3. Training & Benchmarking (Round 12 & OSCAR Cluster)

```bash
# Train PolyGaussNet V3.4-8sec (supports CUDA/MPS/CPU, auto-caching, Ctrl+C resume)
python training_round12/train_round12.py

# Run 3-way synthetic benchmark (Conventional vs V3 vs V3.4-8sec)
python training_round12/testing12.py

# On Brown OSCAR Slurm cluster (see training_round12/OSCAR_RUNBOOK.md):
sbatch training_round12/submit_round12.sh
```

### 4. Downstream Classification Benchmark

```bash
# Train & evaluate Linear, XGBoost, and ResNet1D on Bacteria-ID and MLROD
python training_round12/train_downstream12.py
python training_round12/test_downstream12.py --checkpoint-mode best
```

### 5. Running the Conventional Pipeline

```python
import numpy as np
from test_files.conventional.pre import preprocess_pipeline

raw_spectra = np.load("test_data.npz")["full_matrix"]
cleaned = preprocess_pipeline(
    raw_spectra, despike=True, denoise=True, baseline=True, normalize=True, shift=True
)
```

---

## Loss Functions

PolyGaussNet is trained with a dual-supervised composite objective:

$$\mathcal{L}_{\text{total}} = \lambda_{\text{BC}} \mathcal{L}_{\text{LogCosh}}(X_{\text{BC}}, \hat{X}_{\text{BC}}) + \lambda_{\text{Clean}} \mathcal{L}_{\text{LogCosh}}(X_{\text{pure}}, \hat{X}_{\text{clean}}) + \lambda_{\text{asym}} \mathcal{L}_{\text{asym}}$$

- **Log-Cosh Loss:** $\mathcal{L}_{\text{LogCosh}}(y, \hat{y}) = \frac{1}{N}\sum_{i=1}^N \ln\cosh(\hat{y}_i - y_i)$ — smooth, outlier-robust surrogate for MAE ($\lambda_{\text{BC}} = 1.0$, $\lambda_{\text{Clean}} = 1.0$).
- **Asymmetric Baseline Penalty ($\lambda_{\text{asym}} = 15.0$):** $\mathcal{L}_{\text{asym}} = \frac{1}{N}\sum_i \left(\max(0, \hat{B}_i - X_{\text{raw}, i})\right)^2$ — strongly penalizes baseline overprediction above the raw spectrum to prevent peak clipping.

---

## Requirements

Core dependencies in [`requirements.txt`](requirements.txt):
- `numpy>=1.24.0`, `torch>=2.0.0`, `scipy>=1.10.0`, `astropy>=5.0`
- `matplotlib>=3.7.0`, `seaborn>=0.12.0`, `scikit-image>=0.20.0`
- `PyWavelets>=1.4.0`, `peakutils>=1.3.0`, `BaselineRemoval>=0.1.6`
- `Raman-Plotting-Pipeline` (installed via `git+https://github.com/bogomilkissiov/Raman-Plotting-Pipeline.git`)
- Downstream benchmarks additionally use `scikit-learn`, `xgboost`, and `joblib`.
