"""
TESTING & BENCHMARKING SCRIPT - ROUND 3 (PolyGaussNet)
------------------------------------------------------
Head-to-head evaluation of PolyGaussNet3 vs. Conventional Preprocessing Pipeline (pre.py)
on the `test_spectra3` dataset.

EVALUATES:
1. Cosine Similarity vs. Ground Truth Pure Spectra
2. Mean Squared Error (MSE) vs. Ground Truth Pure Spectra
3. Numerically Stable Log-Cosh Loss vs. Ground Truth Pure Spectra
4. Mean Absolute Error (MAE) & Root Mean Squared Error (RMSE)
5. Execution Time, Processing Throughput (spectra/s), and Latency (ms/spectrum)
6. Speedup Factor of PolyGaussNet3 relative to Conventional Preprocessing

GENERATES:
- `testing3_comparison_overlay.png`: 4-panel visual comparison across diverse spectra
- `testing3_residuals_and_breakdown.png`: PolyGaussNet multi-stage decomposition & error residuals
- `testing3_metrics_benchmark.png`: Head-to-head quantitative metrics bar chart
- Interactive `SpectrumSlideshow` viewer for 4-panel spectrum comparisons
- Interactive `ResidualSlideshow` viewer for 4-stage decomposition and error residual inspection

USAGE:
  # 1. Run full evaluation with default parameters:
  python testing3.py

  # 2. Launch 4-stage residual & decomposition slideshow:
  python testing3.py --residual_slideshow

  # 3. Launch both slideshows with a subset of samples:
  python testing3.py --max-samples 1024 --residual_slideshow
"""

import os
import sys
import time
import glob
import copy
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed

# Configure paths so imports resolve cleanly whether running from project root or inside training_round3
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
CONVENTIONAL_DIR = os.path.join(PROJECT_ROOT, "test_files", "conventional")

for p in [PROJECT_ROOT, SCRIPT_DIR, CONVENTIONAL_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Imports from project
from spectra_class import spectrum, spectra
import pre
from polygaussnet import PolyGaussNet
import gruvbox_theme
from gruvbox_theme import GRUVBOX


# =====================================================================
# 1. CONFIGURATION & CLI ARGUMENT PARSER
# =====================================================================
DEFAULT_DATA_DIR = "test_spectra3"
DEFAULT_MODEL_PATH = "polygaussnet3.pth"
DEFAULT_BATCH_SIZE = 512
DEFAULT_N_JOBS = min(os.cpu_count() or 4, 8)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Head-to-head evaluation: PolyGaussNet3 vs Conventional Preprocessing (pre.py)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR,
                        help="Directory containing test dataset .npz files")
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH,
                        help="Path to trained PolyGaussNet3 model weights (.pth)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum number of test spectra to evaluate (default: all)")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help="Mini-batch size for PyTorch neural network inference")
    parser.add_argument("--n-jobs", type=int, default=DEFAULT_N_JOBS,
                        help="Number of CPU worker processes for conventional pipeline")
    parser.add_argument("--no-plots", action="store_true", default=False,
                        help="Skip generating PNG visual plots")
    parser.add_argument("--interactive", action="store_true", default=False,
                        help="Launch interactive matplotlib slideshow viewer")
    parser.add_argument("--residual-slideshow", "--residual_slideshow", action="store_true", default=False,
                        dest="residual_slideshow",
                        help="Launch interactive 4-stage decomposition & error residual slideshow viewer")
    parser.add_argument("--poly-order", type=int, default=15,
                        help="Polynomial order for PolyGaussNet baseline model")
    parser.add_argument("--filter-kernel-size", type=int, default=31,
                        help="Adaptive filter kernel size for PolyGaussNet")
    return parser.parse_args()


# =====================================================================
# 2. NUMERICAL METRIC FUNCTIONS
# =====================================================================
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Computes rigorous spectral evaluation metrics between ground truth and predictions.
    y_true, y_pred shape: (N, L)
    """
    # 1. Cosine Similarity per spectrum
    dot_product = np.sum(y_true * y_pred, axis=1)
    norm_true = np.linalg.norm(y_true, axis=1)
    norm_pred = np.linalg.norm(y_pred, axis=1)

    valid_mask = (norm_true > 1e-12) & (norm_pred > 1e-12)
    both_zero_mask = (norm_true <= 1e-12) & (norm_pred <= 1e-12)

    cosine_sim = np.zeros(y_true.shape[0], dtype=np.float64)
    cosine_sim[both_zero_mask] = 1.0
    cosine_sim[valid_mask] = dot_product[valid_mask] / (norm_true[valid_mask] * norm_pred[valid_mask])

    # 2. Mean Squared Error (MSE) per spectrum
    mse = np.mean((y_true - y_pred) ** 2, axis=1)

    # 3. Log-Cosh Loss per spectrum (numerically stable: |x| + log1p(exp(-2|x|)) - log(2))
    diff = y_pred - y_true
    abs_diff = np.abs(diff)
    log_cosh_elementwise = abs_diff + np.log1p(np.exp(-2.0 * abs_diff)) - np.log(2.0)
    log_cosh = np.mean(log_cosh_elementwise, axis=1)

    # 4. Mean Absolute Error (MAE) per spectrum
    mae = np.mean(abs_diff, axis=1)

    # 5. Root Mean Squared Error (RMSE) per spectrum
    rmse = np.sqrt(mse)

    return {
        "Cosine Similarity": (float(np.mean(cosine_sim)), float(np.std(cosine_sim))),
        "MSE": (float(np.mean(mse)), float(np.std(mse))),
        "Log-Cosh": (float(np.mean(log_cosh)), float(np.std(log_cosh))),
        "MAE": (float(np.mean(mae)), float(np.std(mae))),
        "RMSE": (float(np.mean(rmse)), float(np.std(rmse))),
        "_arrays": {
            "cosine_sim": cosine_sim,
            "mse": mse,
            "log_cosh": log_cosh,
            "mae": mae,
            "rmse": rmse
        }
    }


# =====================================================================
# 3. CONVENTIONAL PIPELINE WORKER FUNCTION (MULTIPROCESSING)
# =====================================================================
def _process_chunk_worker(wavenumbers_chunk, raw_chunk):
    """Worker function for running conventional preprocessing on a chunk of spectra."""
    conv_data = spectra.from_matrices(wavenumbers_chunk, raw_chunk)
    pre.preprocess_pipeline(conv_data, normalize=False, shift=False)
    return conv_data.intensity_matrix


def run_conventional_pipeline(raw_np: np.ndarray, wavenumbers_np: np.ndarray, n_jobs: int = 4):
    """
    Runs conventional pipeline (Despike -> BayesShrink Wavelet -> Baseline Removal)
    across raw spectra in parallel across CPU cores.
    """
    n_samples = len(raw_np)
    if n_jobs <= 1 or n_samples < 32:
        conv_data = spectra.from_matrices(wavenumbers_np, np.copy(raw_np))
        pre.preprocess_pipeline(conv_data, normalize=False, shift=False)
        return conv_data.intensity_matrix

    # Chunk dataset across worker processes
    chunk_size = int(np.ceil(n_samples / n_jobs))
    chunks = []
    for i in range(0, n_samples, chunk_size):
        end = min(i + chunk_size, n_samples)
        chunks.append((wavenumbers_np[i:end], np.copy(raw_np[i:end])))

    results = [None] * len(chunks)
    with ProcessPoolExecutor(max_workers=n_jobs) as pool:
        future_to_idx = {
            pool.submit(_process_chunk_worker, chunk_wn, chunk_raw): idx
            for idx, (chunk_wn, chunk_raw) in enumerate(chunks)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            results[idx] = future.result()

    return np.vstack(results)


def apply_gruvbox_styling():
    """Configures matplotlib rcParams for Gruvbox dark aesthetic."""
    plt.rcParams.update({
        "figure.facecolor": GRUVBOX.get("bg0", "#282828"),
        "axes.facecolor": GRUVBOX.get("bg0_hard", "#1d2021"),
        "axes.edgecolor": GRUVBOX.get("bg3", "#665c54"),
        "axes.linewidth": 1.0,
        "axes.labelcolor": GRUVBOX.get("fg", "#ebdbb2"),
        "axes.titlecolor": GRUVBOX.get("fg0", "#fbf1c7"),
        "xtick.color": GRUVBOX.get("fg4", "#a89984"),
        "ytick.color": GRUVBOX.get("fg4", "#a89984"),
        "grid.color": GRUVBOX.get("bg2", "#504945"),
        "grid.alpha": 0.45,
        "grid.linestyle": ":",
        "text.color": GRUVBOX.get("fg", "#ebdbb2"),
        "legend.facecolor": GRUVBOX.get("bg0_soft", "#32302f"),
        "legend.edgecolor": GRUVBOX.get("bg3", "#665c54"),
        "legend.labelcolor": GRUVBOX.get("fg1", "#ebdbb2"),
        "legend.framealpha": 0.92,
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    })


# =====================================================================
# 4. MAIN BENCHMARK & EVALUATION ROUTINE
# =====================================================================
def main():
    args = parse_args()

    print("\n" + "=" * 78)
    print(" PolyGaussNet3 vs. Conventional Pipeline Head-to-Head Benchmark ")
    print("=" * 78)

    # -----------------------------------------------------------------
    # 4.1 Load Test Dataset
    # -----------------------------------------------------------------
    data_dir = args.data_dir
    if not os.path.isdir(data_dir):
        candidate_dir = os.path.join(SCRIPT_DIR, data_dir)
        if os.path.isdir(candidate_dir):
            data_dir = candidate_dir
        else:
            candidate_dir = os.path.join(PROJECT_ROOT, "training_round3", data_dir)
            if os.path.isdir(candidate_dir):
                data_dir = candidate_dir

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Cannot find test directory '{args.data_dir}'")

    test_files = sorted(
        glob.glob(os.path.join(data_dir, "*.npz")),
        key=lambda f: int(os.path.splitext(os.path.basename(f))[0].split('_')[-1]) if os.path.splitext(os.path.basename(f))[0].split('_')[-1].isdigit() else f
    )

    if not test_files:
        raise FileNotFoundError(f"No .npz files found in '{data_dir}'")

    print(f"Loading {len(test_files)} dataset files from '{data_dir}':")
    for f in test_files:
        print(f"  • {os.path.basename(f)}")

    pure_list = []
    pure_noise_cosmic_list = []
    full_list = []

    for filepath in test_files:
        with np.load(filepath) as data:
            pure_list.append(data["pure_matrix"])
            pure_noise_cosmic_list.append(data["pure_noise_cosmic_matrix"])
            full_list.append(data["full_matrix"])

    pure_np = np.vstack(pure_list).astype(np.float32)
    bc_true_np = np.vstack(pure_noise_cosmic_list).astype(np.float32)
    raw_np = np.vstack(full_list).astype(np.float32)

    if args.max_samples is not None and args.max_samples < len(raw_np):
        print(f"\nSubsetting to first {args.max_samples:,} spectra (from total {len(raw_np):,})...")
        pure_np = pure_np[:args.max_samples]
        bc_true_np = bc_true_np[:args.max_samples]
        raw_np = raw_np[:args.max_samples]

    total_samples, spectral_length = raw_np.shape
    wavenumbers_np = np.tile(np.arange(spectral_length, dtype=np.float32), (total_samples, 1))

    print(f"\nDataset Statistics:")
    print(f"  - Total Test Spectra: {total_samples:,}")
    print(f"  - Spectral Resolution: {spectral_length} bins (cm⁻¹)")
    print(f"  - Total Data Points:  {total_samples * spectral_length:,}")

    # -----------------------------------------------------------------
    # 4.2 Conventional Preprocessing Pipeline Benchmark (pre.py)
    # -----------------------------------------------------------------
    print("\n" + "-" * 78)
    print(f"1. Running Conventional Preprocessing Pipeline (pre.py) on {args.n_jobs} CPU workers...")
    print("-" * 78)

    t0_conv = time.perf_counter()
    conv_clean_np = run_conventional_pipeline(raw_np, wavenumbers_np, n_jobs=args.n_jobs)
    conv_duration = time.perf_counter() - t0_conv
    conv_throughput = total_samples / conv_duration if conv_duration > 0 else 0
    conv_latency_ms = (conv_duration / total_samples) * 1000 if total_samples > 0 else 0

    print(f"✓ Conventional pipeline finished in {conv_duration:.3f} s")
    print(f"  Throughput: {conv_throughput:8.1f} spectra/s  |  Latency: {conv_latency_ms:6.2f} ms/spectrum")

    # -----------------------------------------------------------------
    # 4.3 PolyGaussNet3 Model Inference Benchmark
    # -----------------------------------------------------------------
    print("\n" + "-" * 78)
    print("2. Running PolyGaussNet3 Neural Network Inference...")
    print("-" * 78)

    # Hardware device detection
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Hardware Acceleration Device: {device}")

    model = PolyGaussNet(
        poly_order=args.poly_order,
        filter_kernel_size=args.filter_kernel_size
    ).to(device)

    # Locate model checkpoint weights
    possible_model_paths = [
        args.model_path,
        os.path.join(SCRIPT_DIR, args.model_path),
        os.path.join(PROJECT_ROOT, "training_round3", args.model_path),
        os.path.join(SCRIPT_DIR, "test_polygaussnet3.pth"),
        os.path.join(PROJECT_ROOT, "training_round3", "test_polygaussnet3.pth"),
        os.path.join(SCRIPT_DIR, "polygaussnet3.pth"),
        os.path.join(PROJECT_ROOT, "training_round3", "polygaussnet3.pth"),
    ]
    model_path = next((p for p in possible_model_paths if os.path.exists(p)), None)
    if model_path is None:
        raise FileNotFoundError(f"Model weights file '{args.model_path}' not found.")

    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    model.eval()
    print(f"Loaded trained PolyGaussNet3 weights from '{model_path}'")

    raw_tensor = torch.tensor(raw_np, dtype=torch.float32)

    poly_clean_list = []
    poly_base_list = []
    poly_bc_list = []

    # Warmup pass (JIT compilation / MPS graph init)
    warmup_batch = raw_tensor[:min(32, total_samples)].to(device)
    with torch.no_grad():
        _ = model(warmup_batch)
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()

    # Benchmark timed inference loop
    t0_nn = time.perf_counter()
    with torch.no_grad():
        for i in range(0, total_samples, args.batch_size):
            batch_in = raw_tensor[i : i + args.batch_size].to(device)
            clean_pred, pred_baseline, bc_pred, _ = model(batch_in)
            poly_clean_list.append(clean_pred.cpu())
            poly_base_list.append(pred_baseline.cpu())
            poly_bc_list.append(bc_pred.cpu())

        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()

    nn_duration = time.perf_counter() - t0_nn
    nn_throughput = total_samples / nn_duration if nn_duration > 0 else 0
    nn_latency_ms = (nn_duration / total_samples) * 1000 if total_samples > 0 else 0

    poly_clean_np = torch.cat(poly_clean_list, dim=0).numpy()
    poly_base_np = torch.cat(poly_base_list, dim=0).numpy()
    poly_bc_np = torch.cat(poly_bc_list, dim=0).numpy()

    print(f"✓ PolyGaussNet3 inference finished in {nn_duration:.3f} s")
    print(f"  Throughput: {nn_throughput:8.1f} spectra/s  |  Latency: {nn_latency_ms:6.4f} ms/spectrum")

    # -----------------------------------------------------------------
    # 4.4 Compute Quantitative Evaluation Metrics
    # -----------------------------------------------------------------
    print("\n" + "-" * 78)
    print("3. Computing Head-to-Head Performance & Spectral Accuracy Metrics...")
    print("-" * 78)

    metrics_conv = compute_metrics(pure_np, conv_clean_np)
    metrics_nn = compute_metrics(pure_np, poly_clean_np)
    metrics_raw = compute_metrics(pure_np, raw_np)

    speedup = conv_duration / nn_duration if nn_duration > 0 else 0

    # -----------------------------------------------------------------
    # 4.5 Print Comprehensive Report Table
    # -----------------------------------------------------------------
    print("\n" + "=" * 82)
    print(f"{'HEAD-TO-HEAD BENCHMARK REPORT (N = ' + f'{total_samples:,}' + ' Spectra)':^82}")
    print("=" * 82)
    print(f"{'Metric / Parameter':<30} | {'Conventional (pre.py)':<24} | {'PolyGaussNet3':<22}")
    print("-" * 82)

    # Core requested metrics
    c_cos, c_cos_std = metrics_conv["Cosine Similarity"]
    n_cos, n_cos_std = metrics_nn["Cosine Similarity"]
    print(f"{'Cosine Similarity (↑)':<30} | {c_cos:9.6f} (±{c_cos_std:.4f})     | {n_cos:9.6f} (±{n_cos_std:.4f})")

    c_mse, c_mse_std = metrics_conv["MSE"]
    n_mse, n_mse_std = metrics_nn["MSE"]
    print(f"{'Mean Squared Error (MSE ↓)':<30} | {c_mse:9.6f} (±{c_mse_std:.4f})     | {n_mse:9.6f} (±{n_mse_std:.4f})")

    c_lc, c_lc_std = metrics_conv["Log-Cosh"]
    n_lc, n_lc_std = metrics_nn["Log-Cosh"]
    print(f"{'Log-Cosh Loss (↓)':<30} | {c_lc:9.6f} (±{c_lc_std:.4f})     | {n_lc:9.6f} (±{n_lc_std:.4f})")

    c_mae, c_mae_std = metrics_conv["MAE"]
    n_mae, n_mae_std = metrics_nn["MAE"]
    print(f"{'Mean Absolute Error (MAE ↓)':<30} | {c_mae:9.6f} (±{c_mae_std:.4f})     | {n_mae:9.6f} (±{n_mae_std:.4f})")

    c_rmse, c_rmse_std = metrics_conv["RMSE"]
    n_rmse, n_rmse_std = metrics_nn["RMSE"]
    print(f"{'Root Mean Sq. Error (RMSE ↓)':<30} | {c_rmse:9.6f} (±{c_rmse_std:.4f})     | {n_rmse:9.6f} (±{n_rmse_std:.4f})")

    print("-" * 82)
    print(f"{'Total Execution Time':<30} | {conv_duration:9.3f} s             | {nn_duration:9.3f} s")
    print(f"{'Processing Throughput':<30} | {conv_throughput:9.1f} spec/s         | {nn_throughput:9.1f} spec/s")
    print(f"{'Latency per Spectrum':<30} | {conv_latency_ms:9.2f} ms             | {nn_latency_ms:9.4f} ms")
    print(f"{'Speedup Factor':<30} | {'1.0x (Baseline)':<24} | {f'{speedup:.1f}x Faster':<22}")
    print("=" * 82 + "\n")

    # -----------------------------------------------------------------
    # 4.6 Generate Publication-Quality Comparison Plots (Gruvbox Theme)
    # -----------------------------------------------------------------
    if not args.no_plots:
        print("Generating visual comparison plots (Gruvbox Dark Theme)...")
        apply_gruvbox_styling()

        # Gruvbox Palette
        COLOR_RAW = GRUVBOX.get("gray", "#928374")       # Raw noisy input
        COLOR_CONV = GRUVBOX.get("orange", "#fe8019")    # Conventional pipeline
        COLOR_POLY = GRUVBOX.get("green", "#b8bb26")     # PolyGaussNet3
        COLOR_PURE = GRUVBOX.get("blue", "#83a598")      # Ground truth pure
        COLOR_BASE = GRUVBOX.get("yellow", "#fabd2f")    # Inferred baseline
        COLOR_LATENT = GRUVBOX.get("purple", "#d3869b")  # Latent baseline-corrected
        COLOR_BG = GRUVBOX.get("bg0_hard", "#1d2021")
        COLOR_FG = GRUVBOX.get("fg0", "#fbf1c7")

        # -------------------------------------------------------------
        # PLOT 1: Multi-Spectrum Comparison Overlay (Gruvbox Dark)
        # -------------------------------------------------------------
        sample_indices = [0, min(10, total_samples - 1), min(50, total_samples - 1), min(100, total_samples - 1)]
        if total_samples <= 4:
            sample_indices = list(range(total_samples))

        fig, axes = plt.subplots(len(sample_indices), 1, figsize=(14, 3.2 * len(sample_indices)), sharex=True, dpi=150, facecolor=GRUVBOX.get("bg0", "#282828"))
        if len(sample_indices) == 1:
            axes = [axes]

        for ax_i, idx in enumerate(sample_indices):
            ax = axes[ax_i]
            ax.set_facecolor(COLOR_BG)
            wn = wavenumbers_np[idx]

            # Compute per-sample metrics
            poly_cos = metrics_nn["_arrays"]["cosine_sim"][idx]
            conv_cos = metrics_conv["_arrays"]["cosine_sim"][idx]
            poly_mse = metrics_nn["_arrays"]["mse"][idx]
            conv_mse = metrics_conv["_arrays"]["mse"][idx]

            ax.plot(wn, raw_np[idx], label='Raw Noisy Input (Peaks + Baseline + Noise + Cosmic)', color=COLOR_RAW, alpha=0.55, linewidth=0.7)
            ax.plot(wn, conv_clean_np[idx], label=f'Conventional Pipeline (Cos: {conv_cos:.4f}, MSE: {conv_mse:.5f})', color=COLOR_CONV, alpha=0.85, linewidth=0.85)
            ax.plot(wn, poly_clean_np[idx], label=f'PolyGaussNet3 Inferred (Cos: {poly_cos:.4f}, MSE: {poly_mse:.5f})', color=COLOR_POLY, alpha=0.95, linewidth=1.0)
            ax.plot(wn, pure_np[idx], label='Ground Truth Pure Spectrum', color=COLOR_PURE, linestyle='--', alpha=0.9, linewidth=0.9)

            ax.set_title(f"Test Spectrum #{idx + 1} — PolyGaussNet3 (Cosine: {poly_cos:.4f}) vs Conventional (Cosine: {conv_cos:.4f})", fontsize=11, fontweight='bold', color=COLOR_FG, pad=6)
            ax.set_ylabel("Intensity (a.u.)", fontsize=10, color=GRUVBOX.get("fg", "#ebdbb2"))
            ax.grid(True, linestyle=':', alpha=0.45, color=GRUVBOX.get("bg2", "#504945"))
            if ax_i == 0:
                ax.legend(loc='upper right', framealpha=0.92, fontsize=9, facecolor=GRUVBOX.get("bg0_soft", "#32302f"), edgecolor=GRUVBOX.get("bg3", "#665c54"))

        axes[-1].set_xlabel("Raman Shift / Wavenumber Bin (cm⁻¹)", fontsize=11, color=GRUVBOX.get("fg", "#ebdbb2"))
        plt.tight_layout()

        plot1_path = "testing3_comparison_overlay.png"
        plt.savefig(plot1_path, dpi=180, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close()
        print(f"✓ Saved overlay comparison plot to: '{plot1_path}'")

        # -------------------------------------------------------------
        # PLOT 2: Multi-Stage Breakdown & Residuals (Gruvbox Dark)
        # -------------------------------------------------------------
        idx = 0  # Detailed examination of sample 0
        fig, axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True, dpi=150, facecolor=GRUVBOX.get("bg0", "#282828"))

        wn = wavenumbers_np[idx]
        for ax in axes:
            ax.set_facecolor(COLOR_BG)

        # Stage 1: Raw vs Inferred Baseline
        axes[0].plot(wn, raw_np[idx], color=COLOR_RAW, label='Raw Input Spectrum', alpha=0.6, linewidth=0.7)
        axes[0].plot(wn, poly_base_np[idx], color=COLOR_BASE, label='PolyGaussNet Inferred Polynomial Baseline (Order 15)', linewidth=1.1)
        axes[0].set_title("Stage 1: Polynomial Baseline Subtraction", fontsize=11, fontweight='bold', color=COLOR_FG)
        axes[0].set_ylabel("Intensity", fontsize=9, color=GRUVBOX.get("fg", "#ebdbb2"))
        axes[0].legend(loc="upper right", fontsize=9, facecolor=GRUVBOX.get("bg0_soft", "#32302f"), edgecolor=GRUVBOX.get("bg3", "#665c54"))
        axes[0].grid(True, linestyle=':', alpha=0.45, color=GRUVBOX.get("bg2", "#504945"))

        # Stage 2: Baseline-Corrected (Latent) vs True BC
        axes[1].plot(wn, poly_bc_np[idx], color=COLOR_LATENT, label='PolyGaussNet Stage 1 Output (Raw - Baseline)', alpha=0.85, linewidth=0.85)
        axes[1].plot(wn, bc_true_np[idx], color=GRUVBOX.get("fg4", "#a89984"), linestyle=':', label='Ground Truth Baseline-Corrected (Pure + Noise + Cosmic)', alpha=0.7, linewidth=0.8)
        axes[1].set_title("Stage 2: Latent Baseline-Corrected Signal", fontsize=11, fontweight='bold', color=COLOR_FG)
        axes[1].set_ylabel("Intensity", fontsize=9, color=GRUVBOX.get("fg", "#ebdbb2"))
        axes[1].legend(loc="upper right", fontsize=9, facecolor=GRUVBOX.get("bg0_soft", "#32302f"), edgecolor=GRUVBOX.get("bg3", "#665c54"))
        axes[1].grid(True, linestyle=':', alpha=0.45, color=GRUVBOX.get("bg2", "#504945"))

        # Stage 3: Final Denoised Output vs Pure
        axes[2].plot(wn, conv_clean_np[idx], color=COLOR_CONV, label='Conventional Pipeline Output', alpha=0.8, linewidth=0.85)
        axes[2].plot(wn, poly_clean_np[idx], color=COLOR_POLY, label='PolyGaussNet3 Final Inferred Clean Output', linewidth=1.0)
        axes[2].plot(wn, pure_np[idx], color=COLOR_PURE, linestyle='--', label='Ground Truth Pure Spectrum', linewidth=0.9)
        axes[2].set_title("Stage 3: Denoised & Purified Spectrum Comparison", fontsize=11, fontweight='bold', color=COLOR_FG)
        axes[2].set_ylabel("Intensity", fontsize=9, color=GRUVBOX.get("fg", "#ebdbb2"))
        axes[2].legend(loc="upper right", fontsize=9, facecolor=GRUVBOX.get("bg0_soft", "#32302f"), edgecolor=GRUVBOX.get("bg3", "#665c54"))
        axes[2].grid(True, linestyle=':', alpha=0.45, color=GRUVBOX.get("bg2", "#504945"))

        # Stage 4: Error Residuals vs Ground Truth
        poly_resid = poly_clean_np[idx] - pure_np[idx]
        conv_resid = conv_clean_np[idx] - pure_np[idx]
        axes[3].axhline(0, color=GRUVBOX.get("bg3", "#665c54"), linestyle='-', linewidth=0.7, alpha=0.8)
        axes[3].plot(wn, conv_resid, color=COLOR_CONV, label=f'Conventional Residuals (MAE: {np.mean(np.abs(conv_resid)):.5f})', alpha=0.75, linewidth=0.75)
        axes[3].plot(wn, poly_resid, color=COLOR_POLY, label=f'PolyGaussNet3 Residuals (MAE: {np.mean(np.abs(poly_resid)):.5f})', alpha=0.9, linewidth=0.9)
        axes[3].set_title("Stage 4: Reconstruction Error Residuals (Prediction − Ground Truth)", fontsize=11, fontweight='bold', color=COLOR_FG)
        axes[3].set_xlabel("Raman Shift / Wavenumber Bin (cm⁻¹)", fontsize=10, color=GRUVBOX.get("fg", "#ebdbb2"))
        axes[3].set_ylabel("Error Residual", fontsize=9, color=GRUVBOX.get("fg", "#ebdbb2"))
        axes[3].legend(loc="upper right", fontsize=9, facecolor=GRUVBOX.get("bg0_soft", "#32302f"), edgecolor=GRUVBOX.get("bg3", "#665c54"))
        axes[3].grid(True, linestyle=':', alpha=0.45, color=GRUVBOX.get("bg2", "#504945"))

        plt.tight_layout()
        plot2_path = "testing3_residuals_and_breakdown.png"
        plt.savefig(plot2_path, dpi=180, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close()
        print(f"✓ Saved multi-stage decomposition plot to: '{plot2_path}'")

        # -------------------------------------------------------------
        # PLOT 3: Metrics & Throughput Benchmark Bar Chart (Gruvbox)
        # -------------------------------------------------------------
        fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), dpi=150, facecolor=GRUVBOX.get("bg0", "#282828"))
        methods = ['Conventional\n(pre.py)', 'PolyGaussNet3']
        colors = [COLOR_CONV, COLOR_POLY]

        for ax in axes:
            ax.set_facecolor(COLOR_BG)

        # 1. Cosine Similarity
        bars0 = axes[0].bar(methods, [c_cos, n_cos], yerr=[c_cos_std, n_cos_std], capsize=5, color=colors, alpha=0.88, edgecolor=GRUVBOX.get("bg3", "#665c54"), error_kw=dict(ecolor=GRUVBOX.get("fg2", "#d5c4a1"), lw=1.2))
        axes[0].set_title("Cosine Similarity (↑)\nHigher is Better", fontsize=11, fontweight='bold', color=COLOR_FG)
        axes[0].set_ylabel("Similarity Score", color=GRUVBOX.get("fg", "#ebdbb2"))
        axes[0].set_ylim([max(0.0, min(c_cos, n_cos) - 0.05), 1.02])
        for i, val in enumerate([c_cos, n_cos]):
            axes[0].text(i, val * 0.95, f"{val:.4f}", ha='center', color=GRUVBOX.get("bg0_hard", "#1d2021"), fontweight='bold', fontsize=10)
        axes[0].grid(axis='y', linestyle=':', alpha=0.45, color=GRUVBOX.get("bg2", "#504945"))

        # 2. MSE Loss
        axes[1].bar(methods, [c_mse, n_mse], yerr=[c_mse_std, n_mse_std], capsize=5, color=colors, alpha=0.88, edgecolor=GRUVBOX.get("bg3", "#665c54"), error_kw=dict(ecolor=GRUVBOX.get("fg2", "#d5c4a1"), lw=1.2))
        axes[1].set_title("Mean Squared Error (↓)\nLower is Better", fontsize=11, fontweight='bold', color=COLOR_FG)
        axes[1].set_ylabel("MSE", color=GRUVBOX.get("fg", "#ebdbb2"))
        for i, val in enumerate([c_mse, n_mse]):
            axes[1].text(i, val + max(c_mse, n_mse) * 0.03, f"{val:.5f}", ha='center', color=COLOR_FG, fontweight='bold', fontsize=10)
        axes[1].grid(axis='y', linestyle=':', alpha=0.45, color=GRUVBOX.get("bg2", "#504945"))

        # 3. Log-Cosh Loss
        axes[2].bar(methods, [c_lc, n_lc], yerr=[c_lc_std, n_lc_std], capsize=5, color=colors, alpha=0.88, edgecolor=GRUVBOX.get("bg3", "#665c54"), error_kw=dict(ecolor=GRUVBOX.get("fg2", "#d5c4a1"), lw=1.2))
        axes[2].set_title("Log-Cosh Loss (↓)\nLower is Better", fontsize=11, fontweight='bold', color=COLOR_FG)
        axes[2].set_ylabel("Log-Cosh Loss", color=GRUVBOX.get("fg", "#ebdbb2"))
        for i, val in enumerate([c_lc, n_lc]):
            axes[2].text(i, val + max(c_lc, n_lc) * 0.03, f"{val:.5f}", ha='center', color=COLOR_FG, fontweight='bold', fontsize=10)
        axes[2].grid(axis='y', linestyle=':', alpha=0.45, color=GRUVBOX.get("bg2", "#504945"))

        # 4. Processing Speed (Throughput)
        axes[3].bar(methods, [conv_throughput, nn_throughput], color=colors, alpha=0.88, edgecolor=GRUVBOX.get("bg3", "#665c54"))
        axes[3].set_title(f"Throughput (↑)\n{speedup:.1f}x Speedup", fontsize=11, fontweight='bold', color=COLOR_FG)
        axes[3].set_ylabel("Spectra / Second", color=GRUVBOX.get("fg", "#ebdbb2"))
        for i, val in enumerate([conv_throughput, nn_throughput]):
            axes[3].text(i, val + nn_throughput * 0.03, f"{val:,.0f} spec/s", ha='center', color=COLOR_FG, fontweight='bold', fontsize=10)
        axes[3].grid(axis='y', linestyle=':', alpha=0.45, color=GRUVBOX.get("bg2", "#504945"))

        plt.suptitle(f"PolyGaussNet3 vs Conventional Preprocessing Benchmark (N = {total_samples:,} Spectra)", fontsize=13, fontweight='bold', color=COLOR_FG, y=1.02)
        plt.tight_layout()

        plot3_path = "testing3_metrics_benchmark.png"
        plt.savefig(plot3_path, dpi=180, bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close()
        print(f"✓ Saved quantitative benchmark bar chart to: '{plot3_path}'\n")

    # -----------------------------------------------------------------
    # 4.7 Optional Interactive Slideshow Viewers (Gruvbox)
    # -----------------------------------------------------------------
    if args.interactive:
        print("Launching interactive SpectrumSlideshow viewer in Gruvbox style (use [◀ / ▶] arrow keys, ESC to exit)...")
        slideshow = SpectrumSlideshow(
            wavenumbers=wavenumbers_np[0],
            pure_mat=pure_np,
            raw_mat=raw_np,
            conv_mat=conv_clean_np,
            nn_mat=poly_clean_np,
            num_samples=min(50, total_samples)
        )
        slideshow.show()

    if args.residual_slideshow:
        print("Launching interactive ResidualSlideshow viewer in Gruvbox style (use [◀ / ▶] arrow keys, ESC to exit)...")
        residual_slideshow = ResidualSlideshow(
            wavenumbers=wavenumbers_np[0],
            pure_mat=pure_np,
            raw_mat=raw_np,
            bc_true_mat=bc_true_np,
            poly_base_mat=poly_base_np,
            poly_bc_mat=poly_bc_np,
            poly_clean_mat=poly_clean_np,
            conv_clean_mat=conv_clean_np,
            num_samples=min(50, total_samples)
        )
        residual_slideshow.show()


# =====================================================================
# 5. INTERACTIVE SLIDESHOW VIEWER CLASS (GRUVBOX DARK THEME)
# =====================================================================
class SpectrumSlideshow:
    """
    Interactive Matplotlib viewer for inspecting Raman spectra step-by-step
    styled in the Gruvbox Dark aesthetic.
    Shows 4 stitched subplots:
      - Ground Truth Pure Spectrum
      - Raw Unprocessed Composite Spectrum
      - Conventional Preprocessed Spectrum
      - PolyGaussNet3 Inferred Spectrum
    """
    def __init__(self, wavenumbers, pure_mat, raw_mat, conv_mat, nn_mat, num_samples=30):
        apply_gruvbox_styling()
        self.wavenumbers = wavenumbers
        self.num_samples = min(num_samples, len(pure_mat))
        self.indices = np.arange(self.num_samples)

        self.pure = pure_mat[self.indices]
        self.raw = raw_mat[self.indices]
        self.conv = conv_mat[self.indices]
        self.nn = nn_mat[self.indices]

        self.current_idx = 0
        self.fig, self.axes = plt.subplots(2, 2, figsize=(14, 8), dpi=120, sharex=True, facecolor=GRUVBOX.get("bg0", "#282828"))
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

    def show(self):
        self.update_plot()
        plt.show()

    def update_plot(self):
        idx = self.current_idx
        sample_num = self.indices[idx]

        for ax in self.axes.flat:
            ax.clear()
            ax.set_facecolor(GRUVBOX.get("bg0_hard", "#1d2021"))

        ax_pure = self.axes[0, 0]
        ax_raw = self.axes[0, 1]
        ax_conv = self.axes[1, 0]
        ax_nn = self.axes[1, 1]

        # 1. Pure Ground Truth
        ax_pure.plot(self.wavenumbers, self.pure[idx], color=GRUVBOX.get("blue", "#83a598"), linewidth=1.0, label="Pure Spectrum")
        ax_pure.set_title("Ground Truth (Pure)", fontsize=11, fontweight="bold", color=GRUVBOX.get("fg0", "#fbf1c7"))
        ax_pure.set_ylabel("Intensity", color=GRUVBOX.get("fg", "#ebdbb2"))
        ax_pure.legend(loc="upper right", fontsize=9, facecolor=GRUVBOX.get("bg0_soft", "#32302f"), edgecolor=GRUVBOX.get("bg3", "#665c54"))
        ax_pure.grid(True, linestyle=':', alpha=0.45, color=GRUVBOX.get("bg2", "#504945"))

        # 2. Raw Unprocessed
        ax_raw.plot(self.wavenumbers, self.raw[idx], color=GRUVBOX.get("gray", "#928374"), linewidth=0.8, label="Raw Composite")
        ax_raw.set_title("Raw Unprocessed (Peaks + Baseline + Noise + Cosmic)", fontsize=11, fontweight="bold", color=GRUVBOX.get("fg0", "#fbf1c7"))
        ax_raw.legend(loc="upper right", fontsize=9, facecolor=GRUVBOX.get("bg0_soft", "#32302f"), edgecolor=GRUVBOX.get("bg3", "#665c54"))
        ax_raw.grid(True, linestyle=':', alpha=0.45, color=GRUVBOX.get("bg2", "#504945"))

        # 3. Conventional Pipeline
        ax_conv.plot(self.wavenumbers, self.conv[idx], color=GRUVBOX.get("orange", "#fe8019"), linewidth=1.0, label="Conventional Pipeline")
        ax_conv.set_title("Conventional Preprocessed (Despike + Wavelet + Baseline)", fontsize=11, fontweight="bold", color=GRUVBOX.get("fg0", "#fbf1c7"))
        ax_conv.set_xlabel("Wavenumber Bin (cm⁻¹)", color=GRUVBOX.get("fg", "#ebdbb2"))
        ax_conv.set_ylabel("Intensity", color=GRUVBOX.get("fg", "#ebdbb2"))
        ax_conv.legend(loc="upper right", fontsize=9, facecolor=GRUVBOX.get("bg0_soft", "#32302f"), edgecolor=GRUVBOX.get("bg3", "#665c54"))
        ax_conv.grid(True, linestyle=':', alpha=0.45, color=GRUVBOX.get("bg2", "#504945"))

        # 4. PolyGaussNet3
        ax_nn.plot(self.wavenumbers, self.nn[idx], color=GRUVBOX.get("green", "#b8bb26"), linewidth=1.1, label="PolyGaussNet3")
        ax_nn.set_title("PolyGaussNet3 Inferred (Polynomial Subtraction + GenGaussian Filter)", fontsize=11, fontweight="bold", color=GRUVBOX.get("fg0", "#fbf1c7"))
        ax_nn.set_xlabel("Wavenumber Bin (cm⁻¹)", color=GRUVBOX.get("fg", "#ebdbb2"))
        ax_nn.legend(loc="upper right", fontsize=9, facecolor=GRUVBOX.get("bg0_soft", "#32302f"), edgecolor=GRUVBOX.get("bg3", "#665c54"))
        ax_nn.grid(True, linestyle=':', alpha=0.45, color=GRUVBOX.get("bg2", "#504945"))

        self.fig.suptitle(
            f"Spectrum Sample {idx + 1} / {self.num_samples} (Test Set #{sample_num + 1}) | Navigate: [◀ / ▶] Arrow Keys | ESC to Exit",
            fontsize=12,
            fontweight="bold",
            color=GRUVBOX.get("fg0", "#fbf1c7"),
            y=0.98
        )
        self.fig.tight_layout(rect=[0, 0.03, 1, 0.96])
        self.fig.canvas.draw_idle()

    def on_key(self, event):
        if event.key in ['right', ' ', 'down']:
            self.current_idx = (self.current_idx + 1) % self.num_samples
            self.update_plot()
        elif event.key in ['left', 'up']:
            self.current_idx = (self.current_idx - 1) % self.num_samples
            self.update_plot()
        elif event.key in ['escape', 'q']:
            plt.close(self.fig)


# =====================================================================
# 6. INTERACTIVE RESIDUAL & MULTI-STAGE SLIDESHOW VIEWER
# =====================================================================
class ResidualSlideshow:
    """
    Interactive Matplotlib viewer for inspecting 4-stage decomposition and error residuals
    step-by-step styled in the Gruvbox Dark aesthetic.
    Shows 4 stacked subplots:
      - Stage 1: Raw Input Spectrum vs Inferred Polynomial Baseline
      - Stage 2: Latent Baseline-Corrected Signal vs Ground Truth BC
      - Stage 3: Denoised Pure Output vs Ground Truth Pure Spectrum
      - Stage 4: Reconstruction Error Residuals (PolyGaussNet3 vs Conventional)
    """
    def __init__(self, wavenumbers, pure_mat, raw_mat, bc_true_mat, poly_base_mat, 
                 poly_bc_mat, poly_clean_mat, conv_clean_mat, num_samples=30):
        apply_gruvbox_styling()
        self.wavenumbers = wavenumbers
        self.num_samples = min(num_samples, len(pure_mat))
        self.indices = np.arange(self.num_samples)

        self.pure = pure_mat[self.indices]
        self.raw = raw_mat[self.indices]
        self.bc_true = bc_true_mat[self.indices]
        self.poly_base = poly_base_mat[self.indices]
        self.poly_bc = poly_bc_mat[self.indices]
        self.poly_clean = poly_clean_mat[self.indices]
        self.conv_clean = conv_clean_mat[self.indices]

        self.COLOR_RAW = GRUVBOX.get("gray", "#928374")
        self.COLOR_CONV = GRUVBOX.get("orange", "#fe8019")
        self.COLOR_POLY = GRUVBOX.get("green", "#b8bb26")
        self.COLOR_PURE = GRUVBOX.get("blue", "#83a598")
        self.COLOR_BASE = GRUVBOX.get("yellow", "#fabd2f")
        self.COLOR_LATENT = GRUVBOX.get("purple", "#d3869b")
        self.COLOR_BG = GRUVBOX.get("bg0_hard", "#1d2021")
        self.COLOR_FG = GRUVBOX.get("fg0", "#fbf1c7")

        self.current_idx = 0
        self.fig, self.axes = plt.subplots(4, 1, figsize=(14, 11), sharex=True, dpi=120, facecolor=GRUVBOX.get("bg0", "#282828"))
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

    def show(self):
        self.update_plot()
        plt.show()

    def update_plot(self):
        idx = self.current_idx
        sample_num = self.indices[idx]
        wn = self.wavenumbers

        for ax in self.axes:
            ax.clear()
            ax.set_facecolor(self.COLOR_BG)

        # Stage 1: Raw vs Inferred Baseline
        self.axes[0].plot(wn, self.raw[idx], color=self.COLOR_RAW, label='Raw Input Spectrum', alpha=0.6, linewidth=0.7)
        self.axes[0].plot(wn, self.poly_base[idx], color=self.COLOR_BASE, label='PolyGaussNet Inferred Polynomial Baseline (Order 15)', linewidth=1.1)
        self.axes[0].set_title("Stage 1: Polynomial Baseline Subtraction", fontsize=11, fontweight='bold', color=self.COLOR_FG)
        self.axes[0].set_ylabel("Intensity", fontsize=9, color=GRUVBOX.get("fg", "#ebdbb2"))
        self.axes[0].legend(loc="upper right", fontsize=9, facecolor=GRUVBOX.get("bg0_soft", "#32302f"), edgecolor=GRUVBOX.get("bg3", "#665c54"))
        self.axes[0].grid(True, linestyle=':', alpha=0.45, color=GRUVBOX.get("bg2", "#504945"))

        # Stage 2: Baseline-Corrected (Latent) vs True BC
        self.axes[1].plot(wn, self.poly_bc[idx], color=self.COLOR_LATENT, label='PolyGaussNet Stage 1 Output (Raw - Baseline)', alpha=0.85, linewidth=0.85)
        self.axes[1].plot(wn, self.bc_true[idx], color=GRUVBOX.get("fg4", "#a89984"), linestyle=':', label='Ground Truth Baseline-Corrected (Pure + Noise + Cosmic)', alpha=0.7, linewidth=0.8)
        self.axes[1].set_title("Stage 2: Latent Baseline-Corrected Signal", fontsize=11, fontweight='bold', color=self.COLOR_FG)
        self.axes[1].set_ylabel("Intensity", fontsize=9, color=GRUVBOX.get("fg", "#ebdbb2"))
        self.axes[1].legend(loc="upper right", fontsize=9, facecolor=GRUVBOX.get("bg0_soft", "#32302f"), edgecolor=GRUVBOX.get("bg3", "#665c54"))
        self.axes[1].grid(True, linestyle=':', alpha=0.45, color=GRUVBOX.get("bg2", "#504945"))

        # Stage 3: Final Denoised Output vs Pure
        self.axes[2].plot(wn, self.conv_clean[idx], color=self.COLOR_CONV, label='Conventional Pipeline Output', alpha=0.8, linewidth=0.85)
        self.axes[2].plot(wn, self.poly_clean[idx], color=self.COLOR_POLY, label='PolyGaussNet3 Final Inferred Clean Output', linewidth=1.0)
        self.axes[2].plot(wn, self.pure[idx], color=self.COLOR_PURE, linestyle='--', label='Ground Truth Pure Spectrum', linewidth=0.9)
        self.axes[2].set_title("Stage 3: Denoised & Purified Spectrum Comparison", fontsize=11, fontweight='bold', color=self.COLOR_FG)
        self.axes[2].set_ylabel("Intensity", fontsize=9, color=GRUVBOX.get("fg", "#ebdbb2"))
        self.axes[2].legend(loc="upper right", fontsize=9, facecolor=GRUVBOX.get("bg0_soft", "#32302f"), edgecolor=GRUVBOX.get("bg3", "#665c54"))
        self.axes[2].grid(True, linestyle=':', alpha=0.45, color=GRUVBOX.get("bg2", "#504945"))

        # Stage 4: Error Residuals vs Ground Truth
        poly_resid = self.poly_clean[idx] - self.pure[idx]
        conv_resid = self.conv_clean[idx] - self.pure[idx]
        self.axes[3].axhline(0, color=GRUVBOX.get("bg3", "#665c54"), linestyle='-', linewidth=0.7, alpha=0.8)
        self.axes[3].plot(wn, conv_resid, color=self.COLOR_CONV, label=f'Conventional Residuals (MAE: {np.mean(np.abs(conv_resid)):.5f})', alpha=0.75, linewidth=0.75)
        self.axes[3].plot(wn, poly_resid, color=self.COLOR_POLY, label=f'PolyGaussNet3 Residuals (MAE: {np.mean(np.abs(poly_resid)):.5f})', alpha=0.9, linewidth=0.9)
        self.axes[3].set_title("Stage 4: Reconstruction Error Residuals (Prediction − Ground Truth)", fontsize=11, fontweight='bold', color=self.COLOR_FG)
        self.axes[3].set_xlabel("Raman Shift / Wavenumber Bin (cm⁻¹)", fontsize=10, color=GRUVBOX.get("fg", "#ebdbb2"))
        self.axes[3].set_ylabel("Error Residual", fontsize=9, color=GRUVBOX.get("fg", "#ebdbb2"))
        self.axes[3].legend(loc="upper right", fontsize=9, facecolor=GRUVBOX.get("bg0_soft", "#32302f"), edgecolor=GRUVBOX.get("bg3", "#665c54"))
        self.axes[3].grid(True, linestyle=':', alpha=0.45, color=GRUVBOX.get("bg2", "#504945"))

        self.fig.suptitle(
            f"Residual Breakdown: Sample {idx + 1} / {self.num_samples} (Test Set #{sample_num + 1}) | Navigate: [◀ / ▶] Arrow Keys | ESC to Exit",
            fontsize=12,
            fontweight="bold",
            color=self.COLOR_FG,
            y=0.99
        )
        self.fig.tight_layout(rect=[0, 0.02, 1, 0.98])
        self.fig.canvas.draw_idle()

    def on_key(self, event):
        if event.key in ['right', ' ', 'down']:
            self.current_idx = (self.current_idx + 1) % self.num_samples
            self.update_plot()
        elif event.key in ['left', 'up']:
            self.current_idx = (self.current_idx - 1) % self.num_samples
            self.update_plot()
        elif event.key in ['escape', 'q']:
            plt.close(self.fig)


if __name__ == "__main__":
    main()

