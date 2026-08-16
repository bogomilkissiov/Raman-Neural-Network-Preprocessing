import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import spectra_generator as g
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 1016, 1)
rng = np.random.default_rng()

# Parameters
batch_size = 100
wavenum_range = [0, 1015]
num_peaks_range = [10, 100]
amplitude_range = [0.009, 0.1]
width_range = [2, 30]
degree_range = [1, 7]
offset_range = [0.0, 2]
max_coeff = 1.0
min_peak_ratio = 2.5
std_range = [0.3, 0.7]
probability_cosmic = 1/12000
intensity_range_cosmic = [3, 10]
domain_mapping = [-1.0, 1.0]
min_value = 0

print("Testing generate_spectra function...")

pure_list, pure_noise_cosmic_list, full_list = g.generate_spectra(
    batch_size=batch_size,
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
    rng=rng,
    min_value=min_value)

# Pick 9 random indices from full_list
sample_indices = rng.choice(batch_size, size=9, replace=False)

# Plot 9 subplots in a 3x3 grid for fully mixed spectra
fig, axes = plt.subplots(3, 3, figsize=(15, 11), dpi=120, sharex=True, sharey=False)
axes = axes.flatten()

for i, idx in enumerate(sample_indices):
    axes[i].plot(x, full_list[idx], color="#2ca02c")
    axes[i].set_title(f"Spectrum #{idx}", pad=8)
    axes[i].set_xlabel("Wavenumber (cm⁻¹)")
    axes[i].set_ylim(bottom=0)
    if i % 3 == 0:
        axes[i].set_ylabel("Intensity")
    axes[i].grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()
