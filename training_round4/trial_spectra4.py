import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from spectra_generator import generate_spectra

GENERATION_PARAMS = {
    "wavenum_range": [0, 1015],
    "num_peaks_range": [0, 200],
    "amplitude_range": [0.001, 0.2],
    "width_range": [1, 200],
    "degree_range": [1, 16],
    "offset_range": [0.0, 1.0],
    "max_coeff": 1.0,
    "min_peak_ratio": 2.0,
    "std_range": [1, 10],
    "probability_cosmic": 0,
    "intensity_range_cosmic": [5.0, 20.0],
    "domain_mapping": [-1.0, 1.0],
    "min_value": 0,
    "normalize" : True}

if __name__ == "__main__":
    # Generate 9 random spectra
    rng = np.random.default_rng()
    pure, pure_noise_cosmic, full = generate_spectra(batch_size=9, rng=rng, **GENERATION_PARAMS)

    w_start, w_end = GENERATION_PARAMS["wavenum_range"]
    x = np.arange(w_start, w_end + 1, 1)

    # Plot 3x3 grid with independent axes for full zoom/pan control
    fig, axes = plt.subplots(3, 3, figsize=(15, 11), dpi=120, sharex=False, sharey=False)
    axes = axes.flatten()

    for i in range(9):
        axes[i].plot(x, full[i], label="Full Spectrum", color="#2ca02c", lw=1.2)
        axes[i].plot(x, pure[i], label="Pure Peaks", color="#1f77b4", lw=1.0, alpha=0.7, linestyle="--")
        axes[i].set_title(f"Sample #{i + 1}", fontsize=11, pad=6)
        axes[i].set_xlabel("Wavenumber (cm⁻¹)", fontsize=9)
        axes[i].grid(True, linestyle="--", alpha=0.5)
        if i % 3 == 0:
            axes[i].set_ylabel("Intensity", fontsize=9)
        if i == 0:
            axes[i].legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.show()
