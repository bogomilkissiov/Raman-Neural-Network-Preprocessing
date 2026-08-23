import os
import sys
import copy
import time
import numpy as np
import matplotlib.pyplot as plt

# Add project root to sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from pre import preprocess_pipeline
from spectra_generator import generate_spectra
from spectra_class import spectra


def calculate_metrics(y_true, y_pred, return_unaltered: bool = False):
    """
    Computes spectral metrics between ground truth and predicted/processed matrices.
    y_true, y_pred shape: (N, L)
    """
    # 1. Cosine Similarity per spectrum
    dot_product = np.sum(y_true * y_pred, axis=1)
    norm_true = np.linalg.norm(y_true, axis=1)
    norm_pred = np.linalg.norm(y_pred, axis=1)

    # Handle zero vectors (e.g. comparing sparse cosmic ray matrices)
    valid_mask = (norm_true > 1e-12) & (norm_pred > 1e-12)
    both_zero_mask = (norm_true <= 1e-12) & (norm_pred <= 1e-12)

    cosine_sim = np.zeros(y_true.shape[0], dtype=np.float64)
    cosine_sim[both_zero_mask] = 1.0
    cosine_sim[valid_mask] = dot_product[valid_mask] / (norm_true[valid_mask] * norm_pred[valid_mask])

    # 2. Mean Squared Error (MSE) per spectrum
    mse = np.mean((y_true - y_pred) ** 2, axis=1)

    # 3. Log-Cosh Loss per spectrum (numerically stable: |x| + log1p(exp(-2|x|)) - log(2))
    diff = y_true - y_pred
    abs_diff = np.abs(diff)
    log_cosh_elementwise = abs_diff + np.log1p(np.exp(-2.0 * abs_diff)) - np.log(2.0)
    log_cosh = np.mean(log_cosh_elementwise, axis=1)

    results = {
        "Cosine Similarity": (np.mean(cosine_sim), np.std(cosine_sim)),
        "MSE": (np.mean(mse), np.std(mse)),
        "Log-Cosh": (np.mean(log_cosh), np.std(log_cosh)),
    }

    if return_unaltered:
        identical_count = np.sum(np.all(np.isclose(y_true, y_pred, atol=1e-10), axis=1))
        results["Unaltered Spectra"] = (identical_count, len(y_true))

    return results


if __name__ == "__main__":
    wavenum_range = [0, 1023]
    num_peaks_range = [10, 100]
    amplitude_range = [0.009, 0.1]
    width_range = [2, 30]
    degree_range = [2, 7]
    offset_range = [0.0, 2.0]
    max_coeff = 1.0
    min_peak_ratio = 2.5
    std_range = [0.3, 0.7]
    probability_cosmic = 1 / 8192
    intensity_range_cosmic = [5, 15]
    domain_mapping = [-1.0, 1.0]
    min_value = 0

    print("Generating synthetic Raman spectra dataset...")
    t0_gen = time.perf_counter()
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
    t1_gen = time.perf_counter()
    print(f"Generated {raw.shape[0]:,} spectra ({raw.shape[1]} points) in {t1_gen - t0_gen:.2f}s.\n")

    wavenumbers = np.arange(wavenum_range[0], wavenum_range[1] + 1)
    wavenumber_matrix = np.tile(wavenumbers, (raw.shape[0], 1))

    raw_spectra = spectra.from_matrices(wavenumber_matrix, raw)

    print(f"Running conventional preprocessing pipeline on {raw.shape[0]:,} spectra...")
    processed_spectra = copy.deepcopy(raw_spectra)

    t0_pipe = time.perf_counter()
    preprocess_pipeline(processed_spectra)
    t1_pipe = time.perf_counter()
    duration = t1_pipe - t0_pipe
    throughput = raw.shape[0] / duration if duration > 0 else 0

    print(f"Preprocessing completed in {duration:.2f}s ({throughput:.1f} spectra/s).\n")

    processed_mat = processed_spectra.intensity_matrix
    metrics = calculate_metrics(pure, processed_mat)

    print("=" * 76)
    print(f"{'CONVENTIONAL PREPROCESSING EVALUATION REPORT (N = ' + f'{raw.shape[0]:,}' + ' Spectra)':^76}")
    print("=" * 76)
    print(f"{'Metric / Parameter':<28} | {'Value (Mean ± Std)':<43}")
    print("-" * 76)
    print(f"{'Cosine Similarity':<28} | {metrics['Cosine Similarity'][0]:10.6f} (±{metrics['Cosine Similarity'][1]:.4f})")
    print(f"{'Mean Squared Error (MSE)':<28} | {metrics['MSE'][0]:10.6f} (±{metrics['MSE'][1]:.4f})")
    print(f"{'Log-Cosh Loss':<28} | {metrics['Log-Cosh'][0]:10.6f} (±{metrics['Log-Cosh'][1]:.4f})")
    print("-" * 76)
    print(f"{'Total Execution Time':<28} | {duration:10.2f} s")
    print(f"{'Throughput':<28} | {throughput:10.1f} spectra/s")
    print(f"{'Time per Spectrum':<28} | {(duration / raw.shape[0]) * 1000:10.2f} ms")
    print("=" * 76 + "\n")


    sample_indices = np.random.choice(raw.shape[0], size=3, replace=False)

    fig, axes = plt.subplots(3, 3, figsize=(15, 10), dpi=120, sharex=True)

    for col, idx in enumerate(sample_indices):
        # Row 1: Raw Unprocessed
        axes[0, col].plot(wavenumbers, raw[idx], color="#d65d0e", linewidth=1.2)
        axes[0, col].set_title(f"Spectrum #{idx + 1} (Raw Unprocessed)", fontsize=10, fontweight="bold", pad=5)
        axes[0, col].grid(True, linestyle="--", alpha=0.4)

        # Row 2: Preprocessed Output
        axes[1, col].plot(wavenumbers, processed_mat[idx], color="#98971a", linewidth=1.2)
        axes[1, col].set_title(f"Spectrum #{idx + 1} (Preprocessed Pipeline)", fontsize=10, fontweight="bold", pad=5)
        axes[1, col].grid(True, linestyle="--", alpha=0.4)

        # Row 3: Ground Truth (Pure Peaks)
        axes[2, col].plot(wavenumbers, pure[idx], color="#458588", linewidth=1.2)
        axes[2, col].set_title(f"Spectrum #{idx + 1} (Ground Truth Pure)", fontsize=10, fontweight="bold", pad=5)
        axes[2, col].set_xlabel("Wavenumber (cm⁻¹)", fontsize=9)
        axes[2, col].grid(True, linestyle="--", alpha=0.4)

    # Set Y-axis labels on the leftmost column
    axes[0, 0].set_ylabel("Raw Intensity", fontsize=10)
    axes[1, 0].set_ylabel("Preprocessed Intensity", fontsize=10)
    axes[2, 0].set_ylabel("Pure Intensity", fontsize=10)

    fig.suptitle(
        "Conventional Preprocessing Pipeline Results (3x3 Grid)\nRow 1: Raw Input | Row 2: Preprocessed Output | Row 3: Ground Truth Pure",
        fontsize=12,
        fontweight="bold",
        y=0.98,
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()