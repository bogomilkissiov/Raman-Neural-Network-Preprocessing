import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Configure paths so imports resolve whether running from project root or subdirectories
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
for path in [PROJECT_ROOT, SCRIPT_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

from spectra_generator import generate_spectra

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
GENERATION_PARAMS = {
    "batch_size": 1,
    "wavenum_range": [0, 1015],
    "num_peaks_range": [10, 100],
    "amplitude_range": [0.009, 0.1],
    "width_range": [2, 30],
    "degree_range": [1, 7],
    "offset_range": [2.0, 10.0],
    "max_coeff": 1.0,
    "min_peak_ratio": 2.5,
    "std_range": [0.3, 0.7],
    "probability_cosmic": 1 / 12000,
    "intensity_range_cosmic": [3.0, 10.0],
    "domain_mapping": [-1.0, 1.0],
    "min_value": 0,
}


def main():
    # 1. Generate 1 spectrum (returns 2D arrays of shape (batch_size, bins))
    pure_batch, noisy_batch, raw_batch = generate_spectra(**GENERATION_PARAMS)

    # 2. Extract the first (and only) spectrum as 1D vectors
    pure = pure_batch[0]
    noisy = noisy_batch[0]
    raw = raw_batch[0]

    # 3. Create wavenumber axis matching spectra_generator (step size = 1)
    wn_min, wn_max = GENERATION_PARAMS["wavenum_range"]
    wavenumbers = np.arange(wn_min, wn_max + 1, 1, dtype=np.float64)

    # 4. Plot both an overlay and individual components for clarity
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    # Subplot 1: Raw spectrum (all components)
    axes[0].plot(wavenumbers, raw, color="tab:blue", lw=1.2, label="Raw Spectrum (Peaks + Baseline + Noise + Cosmic)")
    axes[0].set_ylabel("Intensity")
    axes[0].set_title("Generated Synthetic Raman Spectra Components", fontsize=14, fontweight="bold")
    axes[0].legend(loc="upper right")
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # Subplot 2: Noisy spectrum (peaks + noise + cosmic, no baseline)
    axes[1].plot(wavenumbers, noisy, color="tab:orange", lw=1.2, label="Noisy Spectrum (Peaks + Noise + Cosmic)")
    axes[1].set_ylabel("Intensity")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, linestyle="--", alpha=0.5)

    # Subplot 3: Pure spectrum (peaks only)
    axes[2].plot(wavenumbers, pure, color="tab:green", lw=1.5, label="Pure Spectrum (Peaks Only)")
    axes[2].set_xlabel("Wavenumber ($\text{cm}^{-1}$)")
    axes[2].set_ylabel("Intensity")
    axes[2].legend(loc="upper right")
    axes[2].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()

    # Also show overlaid comparison in a separate figure
    fig_overlay, ax_overlay = plt.subplots(figsize=(12, 6))
    ax_overlay.plot(wavenumbers, raw, label="Raw (Signal + Baseline + Noise + Cosmic)", alpha=0.7, color="tab:blue")
    ax_overlay.plot(wavenumbers, noisy, label="Noisy (Signal + Noise + Cosmic)", alpha=0.7, color="tab:orange")
    ax_overlay.plot(wavenumbers, pure, label="Pure (Ground Truth Signal)", lw=1.8, color="tab:green")
    ax_overlay.set_xlabel("Wavenumber ($\text{cm}^{-1}$)")
    ax_overlay.set_ylabel("Intensity")
    ax_overlay.set_title("Overlaid Spectra Comparison", fontsize=14, fontweight="bold")
    ax_overlay.legend(loc="upper right")
    ax_overlay.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()


