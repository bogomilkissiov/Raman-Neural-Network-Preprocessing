import os
import sys
import copy
import numpy as np
import matplotlib.pyplot as plt

# Add project root to sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from pre import preprocess_pipeline
from spectra_generator import generate_spectra
from spectra_class import spectra
from pipeline_test import calculate_metrics


def gen_cosmic_rays(
    batch: np.ndarray,
    probability: float,
    intensity_range: list = [10.0, 100.0],
    rng: np.random.Generator = None) -> np.ndarray:
    """
    Generates a matrix of pure cosmic rays (zeros everywhere except at cosmic ray spike locations)
    matching the shape of `batch`.
    """
    if rng is None:
        rng = np.random.default_rng()
    result = np.zeros_like(batch)

    # 1. Generate boolean mask
    mask = rng.random(size=batch.shape) < probability
    num_hits = np.count_nonzero(mask)

    # 2. Sample intensities only for the locations that triggered True
    if num_hits > 0:
        ray_intensities = rng.uniform(intensity_range[0], intensity_range[1], size=num_hits)
        result[mask] = ray_intensities
    return result


wavenum_range = [0, 1023]
num_peaks_range = [10, 100]
amplitude_range = [0.009, 0.1]
width_range = [2, 30]
degree_range = [2, 7]
offset_range = [0.0, 2.0]
max_coeff = 1.0
min_peak_ratio = 2.5
std_range = [0.3, 0.7]
probability_cosmic = 0
intensity_range_cosmic = [5, 15]
domain_mapping = [-1.0, 1.0]
min_value = 0

pure, noisy, raw = generate_spectra(
    batch_size=1024,
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
    min_value=min_value
)

cosmic_rays = gen_cosmic_rays(raw, probability=1/512, intensity_range=intensity_range_cosmic)
raw_with_cosmic = raw + cosmic_rays

wavenumbers = np.arange(wavenum_range[0], wavenum_range[1] + 1)
wavenumber_matrix = np.tile(wavenumbers, (raw_with_cosmic.shape[0], 1))

raw_spectra = spectra.from_matrices(wavenumber_matrix, raw_with_cosmic)

despiked_spectra = copy.deepcopy(raw_spectra)
despiked_spectra = preprocess_pipeline(despiked_spectra, denoise=False, baseline=False, normalize=False, shift=False)

raw_mat = raw
raw_cosmic_mat = raw_spectra.intensity_matrix
despiked_mat = despiked_spectra.intensity_matrix
removed_cosmic_rays = raw_cosmic_mat - despiked_mat

metrics_spectra = calculate_metrics(raw_mat, despiked_mat)
metrics_cosmic = calculate_metrics(cosmic_rays, removed_cosmic_rays)

print("\n" + "=" * 90)
print(f"{'DESPIKING BENCHMARK & EVALUATION REPORT (N = ' + str(raw.shape[0]) + ' Spectra)':^90}")
print("=" * 90)
print(f"{'Metric':<22} | {'Raw (Clean) vs. Despiked':<30} | {'True vs. Removed Cosmic Rays':<30}")
print("-" * 90)
for metric in ["Cosine Similarity", "MSE", "Log-Cosh"]:
    spec_val, spec_std = metrics_spectra[metric]
    cos_val, cos_std = metrics_cosmic[metric]
    print(f"{metric:<22} | {spec_val:10.6f} (±{spec_std:.4f}) {'':<10} | {cos_val:10.6f} (±{cos_std:.4f})")
print("=" * 90 + "\n")

sample_indices = np.random.choice(raw.shape[0], size=4, replace=False)

fig, axes = plt.subplots(3, 4, figsize=(16, 9), dpi=120, sharex=True)

for col, idx in enumerate(sample_indices):
    # Row 1: Raw with Cosmic Rays
    axes[0, col].plot(wavenumbers, raw_cosmic_mat[idx], color="#d65d0e", linewidth=1.2)
    axes[0, col].set_title(f"Spectrum #{idx + 1} (Raw + Cosmic)", fontsize=9, fontweight="bold", pad=5)
    axes[0, col].grid(True, linestyle="--", alpha=0.4)

    # Row 2: Raw Ground Truth (No Cosmic)
    axes[1, col].plot(wavenumbers, raw_mat[idx], color="#458588", linewidth=1.2)
    axes[1, col].set_title(f"Spectrum #{idx + 1} (Raw Ground Truth)", fontsize=9, fontweight="bold", pad=5)
    axes[1, col].grid(True, linestyle="--", alpha=0.4)

    # Match Row 1 (Raw + Cosmic) vertical scale to Row 2 (Raw Ground Truth)
    axes[0, col].set_ylim(axes[1, col].get_ylim())

    # Row 3: Despiked Spectra
    axes[2, col].plot(wavenumbers, despiked_mat[idx], color="#98971a", linewidth=1.2)
    axes[2, col].set_title(f"Spectrum #{idx + 1} (Despiked)", fontsize=9, fontweight="bold", pad=5)
    axes[2, col].set_xlabel("Wavenumber (cm⁻¹)", fontsize=9)
    axes[2, col].grid(True, linestyle="--", alpha=0.4)

# Set Y-axis labels on the leftmost column
axes[0, 0].set_ylabel("Raw + Cosmic", fontsize=9)
axes[1, 0].set_ylabel("Raw (Clean)", fontsize=9)
axes[2, 0].set_ylabel("Despiked", fontsize=9)

fig.suptitle(
    "Cosmic Ray Removal Evaluation (3x4 Grid)\nRow 1: Raw + Cosmic | Row 2: Raw Ground Truth | Row 3: Despiked",
    fontsize=12,
    fontweight="bold",
    y=0.98,
)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
