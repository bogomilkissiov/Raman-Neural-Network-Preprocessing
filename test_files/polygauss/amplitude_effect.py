"""
AMPLITUDE EFFECT EXPERIMENT - POLYGAUSSNET ADAPTIVE GAUSSIAN FILTER
===================================================================
This script evaluates and visualizes the effect of the amplitude parameter in
PolyGaussNet's AdaptiveGenGaussianFilter1D.

It generates a random synthetic Raman spectrum and runs it through the filter
while keeping sigma and exponent (beta) constant, comparing filter passes
with dramatically different amplitude values.

Can be executed directly from the project root or from inside `test_files/polygauss/`.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch

# Configure paths so imports resolve seamlessly from root or subdirectories
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
for path in [PROJECT_ROOT, SCRIPT_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

from spectra_generator import generate_spectra
from polygaussnet import AdaptiveGenGaussianFilter1D


def main():
    # -------------------------------------------------------------------------
    # 1. GENERATE ONE RANDOM SYNTHETIC RAMAN SPECTRUM
    # -------------------------------------------------------------------------
    wavenum_range = [0, 1015]
    num_peaks_range = [10, 80]
    amplitude_range = [0.01, 0.1]
    width_range = [3, 25]
    degree_range = [2, 6]
    offset_range = [1.0, 5.0]
    max_coeff = 1.0
    min_peak_ratio = 4
    std_range = [1, 2]
    probability_cosmic = 1 / 10000
    intensity_range_cosmic = [3.0, 8.0]
    domain_mapping = [-1.0, 1.0]
    min_value = 0

    print("Generating random Raman spectrum...")
    pure_batch, noisy_batch, raw_batch = generate_spectra(
        batch_size=1,
        wavenum_range=wavenum_range,
        num_peaks_range=num_peaks_range,
        amplitude_range=amplitude_range,
        width_range=width_range,
        degree_range=degree_range,
        offset_range=offset_range,
        max_coeff=max_coeff,
        min_peak_ratio=min_peak_ratio,
        std_range=std_range,
        probability_cosmic=probability_cosmic,
        intensity_range_cosmic=intensity_range_cosmic,
        domain_mapping=domain_mapping,
        min_value=min_value,
    )

    pure = pure_batch[0]
    noisy = noisy_batch[0]
    raw = raw_batch[0]
    wavenumbers = np.arange(wavenum_range[0], wavenum_range[1] + 1, 1, dtype=np.float64)
    length = len(wavenumbers)

    print(f"Generated spectrum with {length} bins across [{wavenum_range[0]}, {wavenum_range[1]}] cm⁻¹.\n")

    # -------------------------------------------------------------------------
    # 2. INITIALIZE ADAPTIVE GAUSSIAN FILTER LAYER
    # -------------------------------------------------------------------------
    kernel_size = 31
    filter_layer = AdaptiveGenGaussianFilter1D(kernel_size=kernel_size)
    filter_layer.eval()

    # Prepare input tensor: shape (Batch=1, Channels=1, Length=L)
    # Using the noisy (peaks + noise + cosmic) spectrum as typical input to filter
    x_tensor = torch.tensor(noisy, dtype=torch.float32).view(1, 1, length)

    # -------------------------------------------------------------------------
    # 3. SET FIXED SIGMA AND EXPONENT (BETA), AND SPLIT-TEST AMPLITUDES
    # -------------------------------------------------------------------------
    # Fixed parameters
    fixed_sigma_val = 2.5
    fixed_beta_val = 2.0  # beta=2 is standard Gaussian shape exponent

    sigma_tensor = torch.full((1, 1, length), fixed_sigma_val, dtype=torch.float32)
    beta_tensor = torch.full((1, 1, length), fixed_beta_val, dtype=torch.float32)

    # Split test two dramatically different amplitude configurations:
    # A) Constant scalar amplitude comparison (e.g. 0.01 vs 100.0)
    # Note: In AdaptiveGenGaussianFilter1D, A_conserved = amplitude / amplitude.mean()
    # For constant values, A_conserved normalizes to 1.0 (global energy conservation).
    #
    # B) Spatially-varying amplitude profiles (small amplitude modulation vs large amplitude modulation / gain)
    # to demonstrate the spatial redistribution behavior of the filter.

    small_amp_val = 0.05
    large_amp_val = 50.0

    amp_small_const = torch.full((1, 1, length), small_amp_val, dtype=torch.float32)
    amp_large_const = torch.full((1, 1, length), large_amp_val, dtype=torch.float32)

    with torch.no_grad():
        out_small_const = filter_layer(x_tensor, sigma_tensor, amp_small_const, beta_tensor)
        out_large_const = filter_layer(x_tensor, sigma_tensor, amp_large_const, beta_tensor)

    # Convert to 1D numpy arrays
    filtered_small_const = out_small_const.squeeze().cpu().numpy()
    filtered_large_const = out_large_const.squeeze().cpu().numpy()

    # Spatially varying amplitude test:
    # Small amplitude profile (subtle variation) vs Large amplitude profile (strong peak-weighted amplification)
    # Simulating per-bin predicted amplitudes from neural network:
    norm_peaks = (pure - pure.min()) / (pure.max() - pure.min() + 1e-8)
    amp_small_profile = torch.tensor(0.5 + 0.1 * norm_peaks, dtype=torch.float32).view(1, 1, length)
    amp_large_profile = torch.tensor(0.1 + 2.5 * (norm_peaks ** 2), dtype=torch.float32).view(1, 1, length)

    with torch.no_grad():
        out_small_profile = filter_layer(x_tensor, sigma_tensor, amp_small_profile, beta_tensor)
        out_large_profile = filter_layer(x_tensor, sigma_tensor, amp_large_profile, beta_tensor)

    filtered_small_profile = out_small_profile.squeeze().cpu().numpy()
    filtered_large_profile = out_large_profile.squeeze().cpu().numpy()

    print("Filter Evaluation Summary:")
    print(f"  - Kernel Size : {kernel_size}")
    print(f"  - Fixed Sigma : {fixed_sigma_val}")
    print(f"  - Fixed Beta  : {fixed_beta_val} (Gaussian Exponent)")
    print(f"  - Small Amp   : {small_amp_val}")
    print(f"  - Large Amp   : {large_amp_val}")
    print(f"  - Uniform Amplitude Max Absolute Difference: {np.max(np.abs(filtered_small_const - filtered_large_const)):.6e}")
    print("    (Due to energy conservation A_conserved = A / mean(A), uniform amplitude scales are scale-invariant)")
    print(f"  - Profile-Modulated Max Absolute Difference : {np.max(np.abs(filtered_small_profile - filtered_large_profile)):.6f}\n")

    # -------------------------------------------------------------------------
    # 4. PLOT COMPARISONS
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True, dpi=120)

    # Panel 1: Input Spectrum vs Constant Small/Large Amplitude Filter Passes
    axes[0].plot(wavenumbers, noisy, color="#7c6f64", alpha=0.55, lw=1.0, label="Input Noisy Spectrum")
    axes[0].plot(wavenumbers, pure, color="#98971a", linestyle=":", lw=1.5, label="Ground Truth Pure Spectrum")
    axes[0].plot(wavenumbers, filtered_small_const, color="#458588", lw=1.8, label=f"Filter Pass (Small Amp = {small_amp_val})")
    axes[0].plot(wavenumbers, filtered_large_const, color="#cc241d", lw=1.4, linestyle="--", label=f"Filter Pass (Large Amp = {large_amp_val})")
    axes[0].set_title(
        f"Adaptive Gaussian Filter Pass: Uniform Amplitude Split Test (σ={fixed_sigma_val}, β={fixed_beta_val})",
        fontsize=13,
        fontweight="bold",
        pad=8,
    )
    axes[0].set_ylabel("Intensity", fontsize=11)
    axes[0].legend(loc="upper right", framealpha=0.9)
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Panel 2: Spatially-Varying / Peak-Modulated Amplitude Filter Passes
    axes[1].plot(wavenumbers, noisy, color="#7c6f64", alpha=0.45, lw=0.9, label="Input Noisy Spectrum")
    axes[1].plot(wavenumbers, filtered_small_profile, color="#076678", lw=1.8, label="Filter Pass (Low-Modulation Amplitude Profile)")
    axes[1].plot(wavenumbers, filtered_large_profile, color="#d65d0e", lw=1.8, label="Filter Pass (High-Modulation Amplitude Profile)")
    axes[1].set_title(
        "Adaptive Gaussian Filter Pass: Spatially Modulated Amplitude Split Test",
        fontsize=13,
        fontweight="bold",
        pad=8,
    )
    axes[1].set_ylabel("Intensity", fontsize=11)
    axes[1].legend(loc="upper right", framealpha=0.9)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    # Panel 3: Difference / Residual Between the Two Filter Passes
    diff_profile = filtered_large_profile - filtered_small_profile
    axes[2].plot(wavenumbers, diff_profile, color="#b16286", lw=1.6, label="Difference (High Mod Pass - Low Mod Pass)")
    axes[2].axhline(0, color="black", linestyle="--", alpha=0.6, lw=1.0)
    axes[2].set_title("Filter Pass Residual Difference Across Wavenumbers", fontsize=13, fontweight="bold", pad=8)
    axes[2].set_xlabel("Wavenumber (cm⁻¹)", fontsize=11)
    axes[2].set_ylabel("Δ Intensity", fontsize=11)
    axes[2].legend(loc="upper right", framealpha=0.9)
    axes[2].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
