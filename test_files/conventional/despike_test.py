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


wavenum_range = [0, 1027]
num_peaks_range = [10, 100]
amplitude_range = [0.009, 0.1]
width_range = [2, 30]
degree_range = [2, 7]
offset_range = [0.0, 2.0]
max_coeff = 1.0
min_peak_ratio = 2.5
std_range = [0.3, 0.7]
probability_cosmic = 0.0
intensity_range_cosmic = [0, 1]
domain_mapping = [-1.0, 1.0]
min_value = 0

pure, noisy, raw = generate_spectra(
    batch_size=512,
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


print(f"Running despike_spectra on spike-free spectra...")
wavenumbers = np.arange(wavenum_range[0], wavenum_range[1] + 1)
wavenumber_matrix = np.tile(wavenumbers, (raw.shape[0], 1))

raw_spectra = spectra.from_matrices(wavenumber_matrix, raw)

despiked_spectra = copy.deepcopy(raw_spectra)
despiked_spectra = preprocess_pipeline(despiked_spectra, denoise=False, baseline=False, normalize=False, shift=False)

metrics = calculate_metrics(raw_spectra.intensity_matrix, despiked_spectra.intensity_matrix, return_unaltered=True)


print("\n" + "=" * 70)
print(f"{'SPIKE REMOVAL FIDELITY ON CLEAN SPECTRA (N = ' + str(len(raw_spectra.intensity_matrix)) + ')':^70}")
print("=" * 70)
print(f"Cosine Similarity (Original vs Despiked): {metrics['Cosine Similarity'][0]:.8f} (±{metrics['Cosine Similarity'][1]:.6f})")
print(f"Mean Squared Error (MSE)              : {metrics['MSE'][0]:.8e} (±{metrics['MSE'][1]:.6e})")
print(f"Log-Cosh Loss                          : {metrics['Log-Cosh'][0]:.8e} (±{metrics['Log-Cosh'][1]:.6e})")
print(f"Perfectly Unaltered Spectra            : {metrics['Unaltered Spectra'][0]} / {metrics['Unaltered Spectra'][1]} ({metrics['Unaltered Spectra'][0]/metrics['Unaltered Spectra'][1]*100:.1f}%)")
print("=" * 70 + "\n")
