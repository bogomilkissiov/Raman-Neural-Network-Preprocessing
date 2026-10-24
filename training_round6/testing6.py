"""
TESTING & BENCHMARKING SCRIPT - ROUND 6 (PolyGaussNet3 vs. Conventional Pipeline)
---------------------------------------------------------------------------------
Head-to-head quantitative and visual evaluation featuring DUAL VALIDATION MODES:
  1. STANDARD RAW VALIDATION: Evaluates unshifted, raw model outputs directly vs.
     ground truth pure spectra.
  2. BASELINE-SHIFTED VALIDATION: Applies the exact same baseline alignment shift
     to BOTH PolyGaussNet3 and Conventional Preprocessing to isolate pure spectral
     shape, peak height, and peak width reconstruction fidelity.

MODELS COMPARED:
  1. PolyGaussNet3 (Degree 7 Baseline + Kernel 31 Adaptive Gaussian Filter + Bounded Amplitude)
  2. Conventional Preprocessing Pipeline (test_files/conventional/pre.py: Despike + Wavelet + Baseline)

EVALUATES (Across both Raw and Shifted Modes):
  1. Cosine Similarity vs. Ground Truth Pure Spectra (↑ Higher is better)
  2. Mean Squared Error (MSE) vs. Ground Truth Pure Spectra (↓ Lower is better)
  3. Numerically Stable Log-Cosh Loss vs. Ground Truth Pure Spectra (↓ Lower is better)
  4. Mean Absolute Error (MAE) & Root Mean Squared Error (RMSE) (↓ Lower is better)
  5. Peak Summit MAE (top 5% highest peaks) (↓ Lower is better)
  6. Baseline Overshoot Violation Rate (% of bins where B_pred > X_raw)
  7. Execution Time, Processing Throughput (spectra/s), and Latency (ms/spectrum)
  8. Speedup Factor of PolyGaussNet3 relative to Conventional Preprocessing

GENERATES:
  - `testing6_report.txt`: Dual-mode text benchmark report (Raw vs. Shifted).
  - `testing6_comparison_overlay.png`: Visual multi-spectrum comparison overlay (Gruvbox Dark Theme).
  - `testing6_metrics_benchmark.png`: Grouped bar chart comparing Raw vs. Shifted metrics.

USAGE:
  # 1. Full head-to-head benchmark across testing_data6 (Raw + Shifted):
  python training_round6/testing6.py

  # 2. Fast neural-network-only benchmark (skipping conventional CPU pipeline):
  python training_round6/testing6.py --no-conv

  # 3. Choose alignment mode ('mean' error shift vs. 'min_zero' floor shift):
  python training_round6/testing6.py --align-mode mean

  # 4. Run raw validation only:
  python training_round6/testing6.py --no-shift
"""

import os
import sys
import time
import glob
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

# Configure paths so imports resolve cleanly whether running from project root or inside training_round6
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
CONVENTIONAL_DIR = os.path.join(PROJECT_ROOT, "test_files", "conventional")

for p in [PROJECT_ROOT, SCRIPT_DIR, CONVENTIONAL_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from spectra_class import spectra
import pre
from polygaussnet3 import PolyGaussNet

# =====================================================================
# 1. CONFIGURATION & DEFAULTS
# =====================================================================
DEFAULT_MODEL_PATH = "polygaussnet6.pth"
DEFAULT_CHECKPOINT_PATH = "training_checkpoint6.pth"
DEFAULT_DATA_DIR = "testing_data6"
DEFAULT_FALLBACK_DIR = "training_data6"
DEFAULT_REPORT_PATH = "testing6_report.txt"
DEFAULT_PLOT_OVERLAY = "testing6_comparison_overlay.png"
DEFAULT_PLOT_METRICS = "testing6_metrics_benchmark.png"

DEFAULT_BATCH_SIZE = 512
DEFAULT_N_JOBS = min(os.cpu_count() or 4, 8)
BIN_SIZES = [800, 1016, 1200, 1400, 1600]

POLY_ORDER = 7
FILTER_KERNEL_SIZE = 31
MIN_AMPLITUDE = 0.85
MAX_AMPLITUDE = 1.15

# Gruvbox Dark Palette
GRUVBOX = {
    "bg0_hard": "#1d2021",
    "bg0":      "#282828",
    "bg1":      "#3c3836",
    "fg0":      "#fbf1c7",
    "fg1":      "#ebdbb2",
    "gray":     "#928374",
    "red":      "#fb4934",
    "green":    "#b8bb26",
    "yellow":   "#fabd2f",
    "blue":     "#83a598",
    "purple":   "#d3869b",
    "aqua":     "#8ec07c",
    "orange":   "#fe8019",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Head-to-head evaluation: PolyGaussNet3 vs Conventional Pipeline with Raw and Baseline-Shifted Modes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR,
                        help="Directory containing test dataset folders (defaults to testing_data6, falls back to training_data6)")
    parser.add_argument("--model-path", "--model", type=str, default=DEFAULT_MODEL_PATH,
                        dest="model_path", help="Path to trained PolyGaussNet3 weights (.pth)")
    parser.add_argument("--checkpoint-path", type=str, default=DEFAULT_CHECKPOINT_PATH,
                        help="Path to training checkpoint (used for epoch metadata fallback)")
    parser.add_argument("--bin-sizes", type=int, nargs="+", default=BIN_SIZES,
                        help="Filter evaluation to specific bin resolutions")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help="Mini-batch size for PyTorch neural network inference")
    parser.add_argument("--n-jobs", type=int, default=DEFAULT_N_JOBS,
                        help="Number of parallel CPU worker processes for conventional pipeline")
    parser.add_argument("--no-conv", action="store_true", default=False,
                        help="Skip running conventional preprocessing pipeline")
    parser.add_argument("--no-shift", action="store_true", default=False,
                        help="Skip baseline-shifted evaluation and report only standard raw metrics")
    parser.add_argument("--align-mode", type=str, default="mean", choices=["mean", "min_zero"],
                        help="Exact same baseline shifting method applied to both models ('mean': optimal DC error subtraction, 'min_zero': minimum value shifted to 0)")
    parser.add_argument("--no-plots", action="store_true", default=False,
                        help="Skip generating graphical comparison plots")
    parser.add_argument("--max-samples-per-chunk", type=int, default=None,
                        help="Optionally limit number of test spectra evaluated per chunk")
    parser.add_argument("--report-path", type=str, default=DEFAULT_REPORT_PATH,
                        help="Path to save formatted text benchmark report")
    parser.add_argument("--poly-order", type=int, default=POLY_ORDER,
                        help="Order of polynomial baseline (Degree 7)")
    parser.add_argument("--filter-kernel-size", type=int, default=FILTER_KERNEL_SIZE,
                        help="Gaussian filter window size (K=31)")
    return parser.parse_args()


# =====================================================================
# 2. BASELINE SHIFTING & SPECTRAL ALIGNMENT
# =====================================================================
def align_baseline(y_true: np.ndarray, y_pred: np.ndarray, mode: str = "mean") -> np.ndarray:
    """
    Shifts the predicted spectrum vertically to evaluate pure spectral shape, peak heights,
    and peak widths independent of any constant vertical baseline offset / pedestal error.
    Applied in the EXACT SAME mathematical manner to both Conventional and PolyGaussNet.

    Parameters:
    -----------
    y_true : np.ndarray, shape (N, L)
        Ground truth pure Raman spectra.
    y_pred : np.ndarray, shape (N, L)
        Predicted / preprocessed clean spectra.
    mode : str, default "mean"
        'mean': Subtracts the mean error per spectrum: offset = mean(y_pred - y_true, axis=-1).
                This is the mathematically optimal constant DC shift that minimizes MSE,
                directly isolating peak shape and denoising quality.
        'min_zero': Shifts each spectrum upwards so its minimum intensity is at 0.0.

    Returns:
    --------
    y_shifted : np.ndarray, shape (N, L)
    """
    if mode == "mean":
        offset = np.mean(y_pred - y_true, axis=-1, keepdims=True)
        return y_pred - offset
    elif mode == "min_zero":
        min_vals = np.min(y_pred, axis=-1, keepdims=True)
        return y_pred - min_vals
    else:
        raise ValueError(f"Unknown alignment mode: '{mode}'. Must be 'mean' or 'min_zero'.")


# =====================================================================
# 3. NUMERICAL ACCURACY & ERROR METRIC FUNCTIONS
# =====================================================================
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Computes rigorous spectral evaluation metrics between ground truth pure spectra and predictions.
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
    diff = y_pred - y_true
    mse = np.mean(diff ** 2, axis=1)

    # 3. Log-Cosh Loss per spectrum (numerically stable)
    abs_diff = np.abs(diff)
    log_cosh = abs_diff + np.log1p(np.exp(-2.0 * abs_diff)) - np.log(2.0)
    log_cosh = np.mean(log_cosh, axis=1)

    # 4. MAE & RMSE per spectrum
    mae = np.mean(abs_diff, axis=1)
    rmse = np.sqrt(mse)

    # 5. Peak Summit MAE (top 5% intensity peaks in ground truth pure spectra)
    peak_mae = np.zeros(y_true.shape[0], dtype=np.float64)
    for i in range(y_true.shape[0]):
        thresh = np.percentile(y_true[i], 95)
        mask = y_true[i] >= thresh
        if np.any(mask):
            peak_mae[i] = np.mean(abs_diff[i, mask])
        else:
            peak_mae[i] = mae[i]

    return {
        "Cosine Similarity": (float(np.mean(cosine_sim)), float(np.std(cosine_sim))),
        "MSE": (float(np.mean(mse)), float(np.std(mse))),
        "Log-Cosh": (float(np.mean(log_cosh)), float(np.std(log_cosh))),
        "MAE": (float(np.mean(mae)), float(np.std(mae))),
        "RMSE": (float(np.mean(rmse)), float(np.std(rmse))),
        "Peak Summit MAE": (float(np.mean(peak_mae)), float(np.std(peak_mae))),
        "_arrays": {
            "cosine_sim": cosine_sim,
            "mse": mse,
            "log_cosh": log_cosh,
            "mae": mae,
            "rmse": rmse,
            "peak_mae": peak_mae
        }
    }


def aggregate_metrics_from_arrays(arrays_list: list[dict]) -> dict:
    """Concatenates metric arrays across multiple files and calculates global mean and std."""
    if not arrays_list:
        return {}

    cos_all = np.concatenate([a["cosine_sim"] for a in arrays_list])
    mse_all = np.concatenate([a["mse"] for a in arrays_list])
    lc_all = np.concatenate([a["log_cosh"] for a in arrays_list])
    mae_all = np.concatenate([a["mae"] for a in arrays_list])
    rmse_all = np.concatenate([a["rmse"] for a in arrays_list])
    peak_all = np.concatenate([a["peak_mae"] for a in arrays_list])

    return {
        "Cosine Similarity": (float(np.mean(cos_all)), float(np.std(cos_all))),
        "MSE": (float(np.mean(mse_all)), float(np.std(mse_all))),
        "Log-Cosh": (float(np.mean(lc_all)), float(np.std(lc_all))),
        "MAE": (float(np.mean(mae_all)), float(np.std(mae_all))),
        "RMSE": (float(np.mean(rmse_all)), float(np.std(rmse_all))),
        "Peak Summit MAE": (float(np.mean(peak_all)), float(np.std(peak_all))),
        "_arrays": {
            "cosine_sim": cos_all,
            "mse": mse_all,
            "log_cosh": lc_all,
            "mae": mae_all,
            "rmse": rmse_all,
            "peak_mae": peak_all
        }
    }


# =====================================================================
# 4. CONVENTIONAL PIPELINE MULTIPROCESSING
# =====================================================================
def _process_chunk_worker(wavenumbers_chunk: np.ndarray, raw_chunk: np.ndarray) -> np.ndarray:
    """Worker task running classical despike + wavelet + baseline removal."""
    conv_data = spectra.from_matrices(wavenumbers_chunk, raw_chunk)
    pre.preprocess_pipeline(conv_data, normalize=False, shift=False)
    return conv_data.intensity_matrix


def run_conventional_pipeline(raw_np: np.ndarray, wavenumbers_np: np.ndarray, n_jobs: int = 4) -> np.ndarray:
    """Parallel wrapper executing conventional pipeline across available CPU cores."""
    n_samples = len(raw_np)
    if n_jobs <= 1 or n_samples < 32:
        conv_data = spectra.from_matrices(wavenumbers_np, np.copy(raw_np))
        pre.preprocess_pipeline(conv_data, normalize=False, shift=False)
        return conv_data.intensity_matrix

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


# =====================================================================
# 5. PATH RESOLUTION & DATA DISCOVERY
# =====================================================================
def resolve_path(path_str: str) -> str:
    """Finds directory or file relative to script dir or project root."""
    if os.path.isabs(path_str) and os.path.exists(path_str):
        return path_str
    if os.path.exists(path_str):
        return os.path.abspath(path_str)
    script_rel = os.path.join(SCRIPT_DIR, path_str)
    if os.path.exists(script_rel):
        return script_rel
    root_rel = os.path.join(PROJECT_ROOT, path_str)
    if os.path.exists(root_rel):
        return root_rel
    return path_str


def discover_test_files(target_dir: str, fallback_dir: str, bin_sizes: list[int]) -> tuple[dict, str]:
    """
    Finds test .npz files for each bin size.
    First checks target_dir (e.g. testing_data6). If not found, falls back to fallback_dir (training_data6).
    """
    resolved_target = resolve_path(target_dir)
    active_dir = resolved_target

    files_by_bin = {}
    if os.path.exists(resolved_target):
        for b in bin_sizes:
            bin_path = os.path.join(resolved_target, str(b))
            if os.path.exists(bin_path):
                chunks = sorted(glob.glob(os.path.join(bin_path, "*.npz")))
                if chunks:
                    files_by_bin[b] = chunks[0]  # Dedicated test chunk

    # Fallback to training_data6 if target_dir has no files
    if not files_by_bin:
        resolved_fallback = resolve_path(fallback_dir)
        if os.path.exists(resolved_fallback):
            for b in bin_sizes:
                bin_path = os.path.join(resolved_fallback, str(b))
                if os.path.exists(bin_path):
                    chunks = sorted(glob.glob(os.path.join(bin_path, "*.npz")))
                    if chunks:
                        files_by_bin[b] = chunks[-1]  # Use last chunk as held-out test
            if files_by_bin:
                active_dir = resolved_fallback

    return files_by_bin, active_dir


# =====================================================================
# 6. MAIN BENCHMARK EXECUTION
# =====================================================================
def main():
    args = parse_args()

    output_lines = []
    def log(msg: str = ""):
        print(msg)
        output_lines.append(msg)

    log("\n" + "=" * 94)
    log("  PolyGaussNet3 vs. Conventional Pipeline Head-to-Head Benchmark (Round 6)  ")
    log("  DUAL EVALUATION: Standard Raw Validation & Baseline-Shifted Shape Validation")
    log("=" * 94)

    # 1. Hardware setup
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    log(f"Hardware Acceleration Device: {device}")
    log(f"Baseline Shift Method:        mode='{args.align_mode}' (Applied identically to Conventional and PGN3)")

    # 2. Discover dataset files
    test_files_by_bin, active_data_dir = discover_test_files(args.data_dir, DEFAULT_FALLBACK_DIR, args.bin_sizes)
    if not test_files_by_bin:
        raise FileNotFoundError(
            f"No dataset files found in '{args.data_dir}' or fallback '{DEFAULT_FALLBACK_DIR}'. "
            "Please generate dataset first via: python training_round6/create_dataset6.py"
        )

    log(f"Active Dataset Location:      {active_data_dir}")
    log(f"Discovered Test Bin Sets:     {sorted(list(test_files_by_bin.keys()))} bins")
    for b, fp in sorted(test_files_by_bin.items()):
        log(f"  • {b:4d} bins -> {os.path.basename(os.path.dirname(fp))}/{os.path.basename(fp)}")

    # 3. Load PolyGaussNet3 Model
    log("\n" + "-" * 94)
    log("1. Loading PolyGaussNet3 Model Weights...")
    log("-" * 94)

    model_candidates = [
        args.model_path,
        resolve_path(args.model_path),
        os.path.join(SCRIPT_DIR, "polygaussnet6.pth"),
        os.path.join(SCRIPT_DIR, "training_checkpoint6.pth"),
        resolve_path(args.checkpoint_path)
    ]
    model_path = next((p for p in model_candidates if os.path.exists(p)), None)
    if model_path is None:
        raise FileNotFoundError(f"PolyGaussNet3 weights not found. Checked: {model_candidates}")

    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    trained_epoch = None
    best_loss = None

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        trained_epoch = ckpt.get("epoch", None)
        best_loss = ckpt.get("best_loss", ckpt.get("loss", None))
    else:
        state_dict = ckpt
        chk_file = resolve_path(args.checkpoint_path)
        if os.path.exists(chk_file):
            try:
                chk_meta = torch.load(chk_file, map_location="cpu", weights_only=False)
                if isinstance(chk_meta, dict):
                    trained_epoch = chk_meta.get("epoch", None)
                    best_loss = chk_meta.get("best_loss", chk_meta.get("loss", None))
            except Exception:
                pass

    model = PolyGaussNet(
        poly_order=args.poly_order,
        filter_kernel_size=args.filter_kernel_size,
        min_amplitude=MIN_AMPLITUDE,
        max_amplitude=MAX_AMPLITUDE
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    epoch_info_str = f" [Checkpoint Epoch: {trained_epoch}, Loss: {best_loss:.6f}]" if trained_epoch is not None else ""
    log(f"✓ Loaded PolyGaussNet3 from '{model_path}'{epoch_info_str}")
    log(f"  Configuration: Degree {args.poly_order} Polynomial | Kernel {args.filter_kernel_size} | Amp [{MIN_AMPLITUDE}, {MAX_AMPLITUDE}]")

    # 4. Evaluation Loop
    log("\n" + "-" * 94)
    log("2. Running Head-to-Head Evaluation Across Multi-Resolution Datasets...")
    if args.no_conv:
        log("   (Skipping Conventional Pipeline as --no-conv was requested)")
    else:
        log(f"   (Conventional Pipeline CPU Workers: {args.n_jobs})")
    log("-" * 94)

    total_evaluated_spectra = 0
    total_conv_time = 0.0
    total_nn_time = 0.0

    # Metric array accumulators for global aggregation
    all_conv_raw_metric_arrays = []
    all_conv_shifted_metric_arrays = []
    all_nn_raw_metric_arrays = []
    all_nn_shifted_metric_arrays = []

    res_conv_raw = {}
    res_conv_shifted = {}
    res_nn_raw = {}
    res_nn_shifted = {}
    res_samples = {}
    sample_overlays = []  # For visual plotting

    overshoot_count = 0
    total_bins_evaluated = 0

    sorted_bins = sorted(list(test_files_by_bin.keys()))
    total_files = len(sorted_bins)

    for file_idx, b in enumerate(sorted_bins, start=1):
        fpath = test_files_by_bin[b]

        with np.load(fpath) as data:
            raw_mat = data["full_matrix"]
            pure_mat = data["pure_matrix"]

        if args.max_samples_per_chunk is not None and args.max_samples_per_chunk < len(raw_mat):
            raw_mat = raw_mat[:args.max_samples_per_chunk]
            pure_mat = pure_mat[:args.max_samples_per_chunk]

        n_samples, seq_len = raw_mat.shape
        total_evaluated_spectra += n_samples
        res_samples[b] = n_samples

        wavenumbers_mat = np.tile(np.arange(seq_len, dtype=np.float32), (n_samples, 1))

        # --- A. Conventional Pipeline ---
        conv_clean_mat = None
        conv_clean_shifted = None
        m_conv_raw = None
        m_conv_shifted = None
        conv_chunk_time = 0.0

        if not args.no_conv:
            t0_c = time.perf_counter()
            conv_clean_mat = run_conventional_pipeline(raw_mat, wavenumbers_mat, n_jobs=args.n_jobs)
            conv_chunk_time = time.perf_counter() - t0_c
            total_conv_time += conv_chunk_time

            # 1. Standard Raw Evaluation
            m_conv_raw = compute_metrics(pure_mat, conv_clean_mat)
            all_conv_raw_metric_arrays.append(m_conv_raw["_arrays"])
            res_conv_raw[b] = m_conv_raw["_arrays"]

            # 2. Baseline-Shifted Evaluation
            if not args.no_shift:
                conv_clean_shifted = align_baseline(pure_mat, conv_clean_mat, mode=args.align_mode)
                m_conv_shifted = compute_metrics(pure_mat, conv_clean_shifted)
                all_conv_shifted_metric_arrays.append(m_conv_shifted["_arrays"])
                res_conv_shifted[b] = m_conv_shifted["_arrays"]

        # --- B. PolyGaussNet3 Inference ---
        raw_tensor = torch.from_numpy(raw_mat).float().unsqueeze(1)
        nn_clean_list = []
        nn_base_list = []

        t0_nn = time.perf_counter()
        with torch.no_grad():
            for b_i in range(0, n_samples, args.batch_size):
                bx = raw_tensor[b_i : b_i + args.batch_size].to(device)
                pred_cl, pred_b, _, _ = model(bx)
                nn_clean_list.append(pred_cl.squeeze(1).cpu())
                nn_base_list.append(pred_b.squeeze(1).cpu())

            if device.type == "mps":
                torch.mps.synchronize()
            elif device.type == "cuda":
                torch.cuda.synchronize()

        nn_chunk_time = time.perf_counter() - t0_nn
        total_nn_time += nn_chunk_time

        nn_clean_mat = torch.cat(nn_clean_list, dim=0).numpy()
        nn_base_mat = torch.cat(nn_base_list, dim=0).numpy()

        # Track baseline overshoot
        overshoot_count += (nn_base_mat > raw_mat).sum()
        total_bins_evaluated += nn_base_mat.size

        # 1. Standard Raw Evaluation
        m_nn_raw = compute_metrics(pure_mat, nn_clean_mat)
        all_nn_raw_metric_arrays.append(m_nn_raw["_arrays"])
        res_nn_raw[b] = m_nn_raw["_arrays"]

        # 2. Baseline-Shifted Evaluation (exact same shift function)
        nn_clean_shifted = None
        m_nn_shifted = None
        if not args.no_shift:
            nn_clean_shifted = align_baseline(pure_mat, nn_clean_mat, mode=args.align_mode)
            m_nn_shifted = compute_metrics(pure_mat, nn_clean_shifted)
            all_nn_shifted_metric_arrays.append(m_nn_shifted["_arrays"])
            res_nn_shifted[b] = m_nn_shifted["_arrays"]

        # Save first spectrum from each bin size for plotting
        sample_overlays.append({
            "bin_size": b,
            "raw": raw_mat[0],
            "pure": pure_mat[0],
            "conv_raw": conv_clean_mat[0] if conv_clean_mat is not None else None,
            "conv_shifted": conv_clean_shifted[0] if conv_clean_shifted is not None else None,
            "nn_raw": nn_clean_mat[0],
            "nn_shifted": nn_clean_shifted[0] if nn_clean_shifted is not None else None,
            "base": nn_base_mat[0],
            "nn_mse_raw": m_nn_raw["MSE"][0],
            "nn_mse_shift": m_nn_shifted["MSE"][0] if m_nn_shifted else m_nn_raw["MSE"][0],
            "conv_mse_shift": m_conv_shifted["MSE"][0] if m_conv_shifted else None
        })

        # Progress readout
        nn_tp = n_samples / nn_chunk_time if nn_chunk_time > 0 else 0
        nn_mse_r = m_nn_raw["MSE"][0]
        nn_mse_s = m_nn_shifted["MSE"][0] if m_nn_shifted else nn_mse_r

        if not args.no_conv and m_conv_raw is not None:
            c_tp = n_samples / conv_chunk_time if conv_chunk_time > 0 else 0
            c_mse_r = m_conv_raw["MSE"][0]
            c_mse_s = m_conv_shifted["MSE"][0] if m_conv_shifted else c_mse_r
            spd = (conv_chunk_time / nn_chunk_time) if nn_chunk_time > 0 else 0.0

            if not args.no_shift:
                adv = "PGN3 lower" if nn_mse_s < c_mse_s else "Conv lower"
                log(
                    f"[{file_idx:2d}/{total_files}] L={b:4d} bins (N={n_samples:,}) | "
                    f"RAW MSE: Conv={c_mse_r:.5f}, PGN3={nn_mse_r:.5f} | "
                    f"SHIFTED MSE: Conv={c_mse_s:.5f}, PGN3={nn_mse_s:.5f} ({adv}, {spd:4.1f}x speedup)"
                )
            else:
                log(
                    f"[{file_idx:2d}/{total_files}] L={b:4d} bins (N={n_samples:,}) | "
                    f"RAW MSE: Conv={c_mse_r:.5f}, PGN3={nn_mse_r:.5f} (PGN3 {nn_tp:5.0f} s/s, {spd:4.1f}x)"
                )
        else:
            if not args.no_shift:
                log(
                    f"[{file_idx:2d}/{total_files}] L={b:4d} bins (N={n_samples:,}) | "
                    f"RAW MSE: {nn_mse_r:.5f} | SHIFTED MSE: {nn_mse_s:.5f} ({nn_tp:5.0f} s/s)"
                )
            else:
                log(
                    f"[{file_idx:2d}/{total_files}] L={b:4d} bins (N={n_samples:,}) | "
                    f"RAW MSE: {nn_mse_r:.5f} ({nn_tp:5.0f} s/s)"
                )

    # 5. Aggregate Metrics
    overall_nn_raw = aggregate_metrics_from_arrays(all_nn_raw_metric_arrays)
    overall_conv_raw = aggregate_metrics_from_arrays(all_conv_raw_metric_arrays) if not args.no_conv else None

    overall_nn_shifted = aggregate_metrics_from_arrays(all_nn_shifted_metric_arrays) if not args.no_shift else None
    overall_conv_shifted = aggregate_metrics_from_arrays(all_conv_shifted_metric_arrays) if (not args.no_conv and not args.no_shift) else None

    conv_throughput = total_evaluated_spectra / total_conv_time if total_conv_time > 0 else 0.0
    conv_latency_ms = (total_conv_time / total_evaluated_spectra) * 1000 if total_evaluated_spectra > 0 else 0.0

    nn_throughput = total_evaluated_spectra / total_nn_time if total_nn_time > 0 else 0.0
    nn_latency_ms = (total_nn_time / total_evaluated_spectra) * 1000 if total_evaluated_spectra > 0 else 0.0
    speedup = (total_conv_time / total_nn_time) if (total_conv_time > 0 and total_nn_time > 0) else 1.0
    overshoot_rate = (overshoot_count / total_bins_evaluated * 100.0) if total_bins_evaluated > 0 else 0.0

    # 6. Format Grand Benchmark Report Tables
    def fmt_cell(val_std):
        if val_std is None:
            return "N/A"
        val, std = val_std
        return f"{val:9.5f} (±{std:.4f})"

    log("\n" + "=" * 94)
    log(f"{'HEAD-TO-HEAD BENCHMARK REPORT (N = ' + f'{total_evaluated_spectra:,}' + ' Spectra)':^94}")
    log("=" * 94)

    # --- PART A: RAW VALIDATION ---
    log("\n" + "-" * 94)
    log("  PART A: STANDARD RAW VALIDATION (UNSHIFTED PREDICTIONS)")
    log("-" * 94)
    log(f"{'Metric / Parameter':<30} | {'Conventional (pre.py)':<24} | {'PolyGaussNet3':<22}")
    log("-" * 94)

    c_cos_r = fmt_cell(overall_conv_raw["Cosine Similarity"] if overall_conv_raw else None)
    n_cos_r = fmt_cell(overall_nn_raw["Cosine Similarity"])
    log(f"{'Cosine Similarity (↑)':<30} | {c_cos_r:<24} | {n_cos_r:<22}")

    c_mse_r = fmt_cell(overall_conv_raw["MSE"] if overall_conv_raw else None)
    n_mse_r = fmt_cell(overall_nn_raw["MSE"])
    log(f"{'Mean Squared Error (MSE ↓)':<30} | {c_mse_r:<24} | {n_mse_r:<22}")

    c_lc_r = fmt_cell(overall_conv_raw["Log-Cosh"] if overall_conv_raw else None)
    n_lc_r = fmt_cell(overall_nn_raw["Log-Cosh"])
    log(f"{'Log-Cosh Loss (↓)':<30} | {c_lc_r:<24} | {n_lc_r:<22}")

    c_mae_r = fmt_cell(overall_conv_raw["MAE"] if overall_conv_raw else None)
    n_mae_r = fmt_cell(overall_nn_raw["MAE"])
    log(f"{'Mean Absolute Error (MAE ↓)':<30} | {c_mae_r:<24} | {n_mae_r:<22}")

    c_rmse_r = fmt_cell(overall_conv_raw["RMSE"] if overall_conv_raw else None)
    n_rmse_r = fmt_cell(overall_nn_raw["RMSE"])
    log(f"{'Root Mean Sq. Error (RMSE ↓)':<30} | {c_rmse_r:<24} | {n_rmse_r:<22}")

    c_peak_r = fmt_cell(overall_conv_raw["Peak Summit MAE"] if overall_conv_raw else None)
    n_peak_r = fmt_cell(overall_nn_raw["Peak Summit MAE"])
    log(f"{'Peak Summit MAE (↓)':<30} | {c_peak_r:<24} | {n_peak_r:<22}")

    log(f"{'Baseline Overshoot Rate':<30} | {'N/A':<24} | {f'{overshoot_rate:.3f}% of bins':<22}")

    # --- PART B: BASELINE-SHIFTED VALIDATION ---
    if not args.no_shift and overall_nn_shifted is not None:
        log("\n" + "-" * 94)
        log("  PART B: BASELINE-SHIFTED VALIDATION (PEAK SHAPE & CONTOUR FIDELITY)")
        log(f"  [Both methods shifted identically via {args.align_mode} alignment]")
        log("-" * 94)
        log(f"{'Metric / Parameter':<30} | {'Conventional (Shifted)':<24} | {'PolyGaussNet3 (Shifted)':<22} | {'Advantage':<12}")
        log("-" * 94)

        c_cos_s = fmt_cell(overall_conv_shifted["Cosine Similarity"] if overall_conv_shifted else None)
        n_cos_s = fmt_cell(overall_nn_shifted["Cosine Similarity"])
        adv_cos = "PGN3 (+)" if overall_conv_shifted and overall_nn_shifted["Cosine Similarity"][0] > overall_conv_shifted["Cosine Similarity"][0] else "Conv (+)"
        log(f"{'Cosine Similarity (↑)':<30} | {c_cos_s:<24} | {n_cos_s:<22} | {adv_cos:<12}")

        c_mse_s = fmt_cell(overall_conv_shifted["MSE"] if overall_conv_shifted else None)
        n_mse_s = fmt_cell(overall_nn_shifted["MSE"])
        if overall_conv_shifted:
            mse_diff = (overall_conv_shifted["MSE"][0] - overall_nn_shifted["MSE"][0]) / overall_conv_shifted["MSE"][0] * 100
            adv_mse = f"PGN3 (-{mse_diff:.1f}%)" if mse_diff > 0 else f"Conv (-{-mse_diff:.1f}%)"
        else:
            adv_mse = "N/A"
        log(f"{'Mean Squared Error (MSE ↓)':<30} | {c_mse_s:<24} | {n_mse_s:<22} | {adv_mse:<12}")

        c_lc_s = fmt_cell(overall_conv_shifted["Log-Cosh"] if overall_conv_shifted else None)
        n_lc_s = fmt_cell(overall_nn_shifted["Log-Cosh"])
        adv_lc = "PGN3 (Better)" if overall_conv_shifted and overall_nn_shifted["Log-Cosh"][0] < overall_conv_shifted["Log-Cosh"][0] else "Conv (Better)"
        log(f"{'Log-Cosh Loss (↓)':<30} | {c_lc_s:<24} | {n_lc_s:<22} | {adv_lc:<12}")

        c_mae_s = fmt_cell(overall_conv_shifted["MAE"] if overall_conv_shifted else None)
        n_mae_s = fmt_cell(overall_nn_shifted["MAE"])
        adv_mae = "PGN3 (Better)" if overall_conv_shifted and overall_nn_shifted["MAE"][0] < overall_conv_shifted["MAE"][0] else "Conv (Better)"
        log(f"{'Mean Absolute Error (MAE ↓)':<30} | {c_mae_s:<24} | {n_mae_s:<22} | {adv_mae:<12}")

        c_rmse_s = fmt_cell(overall_conv_shifted["RMSE"] if overall_conv_shifted else None)
        n_rmse_s = fmt_cell(overall_nn_shifted["RMSE"])
        adv_rmse = "PGN3 (Better)" if overall_conv_shifted and overall_nn_shifted["RMSE"][0] < overall_conv_shifted["RMSE"][0] else "Conv (Better)"
        log(f"{'Root Mean Sq. Error (RMSE ↓)':<30} | {c_rmse_s:<24} | {n_rmse_s:<22} | {adv_rmse:<12}")

        c_peak_s = fmt_cell(overall_conv_shifted["Peak Summit MAE"] if overall_conv_shifted else None)
        n_peak_s = fmt_cell(overall_nn_shifted["Peak Summit MAE"])
        adv_peak = "PGN3 (Better)" if overall_conv_shifted and overall_nn_shifted["Peak Summit MAE"][0] < overall_conv_shifted["Peak Summit MAE"][0] else "Conv (Better)"
        log(f"{'Peak Summit MAE (↓)':<30} | {c_peak_s:<24} | {n_peak_s:<22} | {adv_peak:<12}")

    # --- PART C: PERFORMANCE & THROUGHPUT ---
    log("\n" + "-" * 94)
    log("  PART C: COMPUTATIONAL SPEED & THROUGHPUT")
    log("-" * 94)
    c_dur_str = f"{total_conv_time:9.3f} s" if not args.no_conv else "N/A"
    n_dur_str = f"{total_nn_time:9.3f} s"
    log(f"{'Total Execution Time':<30} | {c_dur_str:<24} | {n_dur_str:<22}")

    c_tp_str = f"{conv_throughput:9.1f} spec/s" if not args.no_conv else "N/A"
    n_tp_str = f"{nn_throughput:9.1f} spec/s"
    log(f"{'Processing Throughput':<30} | {c_tp_str:<24} | {n_tp_str:<22}")

    c_lat_str = f"{conv_latency_ms:9.2f} ms" if not args.no_conv else "N/A"
    n_lat_str = f"{nn_latency_ms:9.4f} ms"
    log(f"{'Latency per Spectrum':<30} | {c_lat_str:<24} | {n_lat_str:<22}")

    c_spd_str = "1.0x (Baseline)" if not args.no_conv else "N/A"
    n_spd_str = f"{speedup:.1f}x Faster" if not args.no_conv else "N/A"
    log(f"{'Speedup Factor':<30} | {c_spd_str:<24} | {n_spd_str:<22}")
    log("=" * 94)

    # 7. Resolution Breakdown Table
    log("\n" + "=" * 116)
    log(f"{'SPECTRAL BIN RESOLUTION BREAKDOWN (800 to 1600 Bins)':^116}")
    log("=" * 116)
    if not args.no_shift:
        log(f"{'Bins (L)':<9} | {'Spectra':<8} | {'Conv Raw MSE':<13} | {'PGN3 Raw MSE':<13} | {'Conv Shift MSE':<15} | {'PGN3 Shift MSE':<15} | {'Shifted Advantage':<18}")
        log("-" * 116)

        for b in sorted_bins:
            n_cnt = res_samples[b]
            nn_raw_mse = np.mean(res_nn_raw[b]["mse"])
            nn_shift_mse = np.mean(res_nn_shifted[b]["mse"])

            if not args.no_conv and b in res_conv_raw:
                c_raw_mse = np.mean(res_conv_raw[b]["mse"])
                c_shift_mse = np.mean(res_conv_shifted[b]["mse"])
                pct = (c_shift_mse - nn_shift_mse) / c_shift_mse * 100
                adv_str = f"PGN3 {pct:+.1f}% lower" if pct > 0 else f"Conv {-pct:+.1f}% lower"
                log(f"{b:<9} | {n_cnt:<8} | {c_raw_mse:<13.6f} | {nn_raw_mse:<13.6f} | {c_shift_mse:<15.6f} | {nn_shift_mse:<15.6f} | {adv_str:<18}")
            else:
                log(f"{b:<9} | {n_cnt:<8} | {'N/A':<13} | {nn_raw_mse:<13.6f} | {'N/A':<15} | {nn_shift_mse:<15.6f} | {'N/A':<18}")
    else:
        log(f"{'Bins (L)':<9} | {'Spectra':<8} | {'Conv Cos':<10} | {'PGN3 Cos':<10} | {'Conv MSE':<11} | {'PGN3 MSE':<11} | {'Conv LC':<10} | {'PGN3 LC':<10}")
        log("-" * 116)
        for b in sorted_bins:
            n_cnt = res_samples[b]
            p_cos = np.mean(res_nn_raw[b]["cosine_sim"])
            p_mse = np.mean(res_nn_raw[b]["mse"])
            p_lc = np.mean(res_nn_raw[b]["log_cosh"])

            if not args.no_conv and b in res_conv_raw:
                c_cos = np.mean(res_conv_raw[b]["cosine_sim"])
                c_mse = np.mean(res_conv_raw[b]["mse"])
                c_lc = np.mean(res_conv_raw[b]["log_cosh"])
                log(f"{b:<9} | {n_cnt:<8} | {c_cos:<10.5f} | {p_cos:<10.5f} | {c_mse:<11.6f} | {p_mse:<11.6f} | {c_lc:<10.6f} | {p_lc:<10.6f}")
            else:
                log(f"{b:<9} | {n_cnt:<8} | {'N/A':<10} | {p_cos:<10.5f} | {'N/A':<11} | {p_mse:<11.6f} | {'N/A':<10} | {p_lc:<10.6f}")

    log("=" * 116 + "\n")

    # 8. Save Text Report
    report_save_path = resolve_path(args.report_path) if os.path.isabs(args.report_path) else os.path.join(SCRIPT_DIR, args.report_path)
    with open(report_save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines) + "\n")
    log(f"Report saved to: '{report_save_path}'")

    # 9. Visual Plots (Gruvbox Dark Theme)
    if not args.no_plots:
        log("\nGenerating comparison plots (Gruvbox Dark Theme)...")
        plt.rcParams.update({
            "figure.facecolor": GRUVBOX["bg0_hard"],
            "axes.facecolor": GRUVBOX["bg0"],
            "axes.edgecolor": GRUVBOX["fg1"],
            "axes.labelcolor": GRUVBOX["fg0"],
            "xtick.color": GRUVBOX["fg1"],
            "ytick.color": GRUVBOX["fg1"],
            "grid.color": GRUVBOX["bg1"],
            "text.color": GRUVBOX["fg0"],
            "font.family": "sans-serif"
        })

        # Plot 1: Comparison Overlay
        fig, axes = plt.subplots(len(sample_overlays), 1, figsize=(15, 3.4 * len(sample_overlays)), sharex=False, dpi=160)
        if len(sample_overlays) == 1:
            axes = [axes]

        for idx, item in enumerate(sample_overlays):
            ax = axes[idx]
            wn = np.arange(item["bin_size"])

            ax.plot(wn, item["raw"], label="Raw Noisy Input", color=GRUVBOX["gray"], alpha=0.50, linewidth=0.85)
            ax.plot(wn, item["base"], label="Predicted Baseline B(x)", color=GRUVBOX["yellow"], linestyle="--", linewidth=1.3)
            ax.plot(wn, item["pure"], label="Ground Truth Pure", color=GRUVBOX["blue"], alpha=0.85, linewidth=1.5)

            if item["conv_raw"] is not None:
                ax.plot(wn, item["conv_raw"], label="Conventional (Raw)", color=GRUVBOX["orange"], alpha=0.55, linestyle=":", linewidth=1.0)
            if item["conv_shifted"] is not None:
                ax.plot(wn, item["conv_shifted"], label="Conventional (Shifted)", color=GRUVBOX["orange"], alpha=0.90, linewidth=1.3)

            ax.plot(wn, item["nn_raw"], label="PolyGaussNet3 (Raw)", color=GRUVBOX["aqua"], alpha=0.50, linestyle=":", linewidth=1.0)
            if item["nn_shifted"] is not None:
                ax.plot(wn, item["nn_shifted"], label="PolyGaussNet3 (Shifted)", color=GRUVBOX["green"], linewidth=1.6)

            title_str = f"Resolution L = {item['bin_size']} bins | PGN3 Raw MSE: {item['nn_mse_raw']:.5f} -> Shifted MSE: {item['nn_mse_shift']:.5f}"
            if item["conv_mse_shift"] is not None:
                title_str += f" vs Conv Shifted MSE: {item['conv_mse_shift']:.5f}"
            ax.set_title(title_str, fontsize=11, fontweight="bold", pad=6)
            ax.set_ylabel("Intensity", fontsize=10)
            ax.grid(True, linestyle=":", alpha=0.6)
            if idx == 0:
                ax.legend(loc="upper right", framealpha=0.9, fontsize=8.5, ncol=2)

        axes[-1].set_xlabel("Spectral Bin Index", fontsize=11)
        plt.tight_layout()

        plot1_path = os.path.join(SCRIPT_DIR, DEFAULT_PLOT_OVERLAY)
        plt.savefig(plot1_path, dpi=160)
        plt.close()
        log(f"✓ Saved comparison overlay plot: '{plot1_path}'")

        # Plot 2: Metrics Benchmark Bar Chart (Raw vs Shifted)
        if overall_conv_raw is not None and overall_conv_shifted is not None:
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(17, 4.4), dpi=160)

            x = np.arange(2)  # [Conventional, PolyGaussNet3]
            width = 0.35
            methods = ["Conventional", "PolyGaussNet3"]

            # Panel 1: MSE (Raw vs Shifted)
            c_raw_mse = overall_conv_raw["MSE"][0]
            n_raw_mse = overall_nn_raw["MSE"][0]
            c_sft_mse = overall_conv_shifted["MSE"][0]
            n_sft_mse = overall_nn_shifted["MSE"][0]

            ax1.bar(x - width/2, [c_raw_mse, n_raw_mse], width, label="Raw (Unshifted)", color=GRUVBOX["gray"], alpha=0.85)
            ax1.bar(x + width/2, [c_sft_mse, n_sft_mse], width, label="Baseline-Shifted", color=[GRUVBOX["orange"], GRUVBOX["green"]])
            ax1.set_xticks(x)
            ax1.set_xticklabels(methods, fontweight="bold")
            ax1.set_title("Mean Squared Error (↓)", fontweight="bold")
            for i, (r_val, s_val) in enumerate(zip([c_raw_mse, n_raw_mse], [c_sft_mse, n_sft_mse])):
                ax1.text(i - width/2, r_val * 1.02, f"{r_val:.5f}", ha="center", fontsize=8, color=GRUVBOX["fg1"])
                ax1.text(i + width/2, s_val * 1.02, f"{s_val:.5f}", ha="center", fontsize=8.5, fontweight="bold", color=GRUVBOX["fg0"])
            ax1.legend(loc="upper right", fontsize=8.5)
            ax1.grid(True, linestyle=":", alpha=0.5, axis="y")

            # Panel 2: Cosine Similarity (Raw vs Shifted)
            c_raw_cos = overall_conv_raw["Cosine Similarity"][0]
            n_raw_cos = overall_nn_raw["Cosine Similarity"][0]
            c_sft_cos = overall_conv_shifted["Cosine Similarity"][0]
            n_sft_cos = overall_nn_shifted["Cosine Similarity"][0]

            ax2.bar(x - width/2, [c_raw_cos, n_raw_cos], width, label="Raw", color=GRUVBOX["gray"], alpha=0.85)
            ax2.bar(x + width/2, [c_sft_cos, n_sft_cos], width, label="Shifted", color=[GRUVBOX["orange"], GRUVBOX["green"]])
            ax2.set_xticks(x)
            ax2.set_xticklabels(methods, fontweight="bold")
            ax2.set_title("Cosine Similarity (↑)", fontweight="bold")
            ax2.set_ylim([min(c_raw_cos, n_raw_cos, c_sft_cos, n_sft_cos) * 0.95, 1.0])
            for i, (r_val, s_val) in enumerate(zip([c_raw_cos, n_raw_cos], [c_sft_cos, n_sft_cos])):
                ax2.text(i - width/2, r_val + 0.003, f"{r_val:.4f}", ha="center", fontsize=8, color=GRUVBOX["fg1"])
                ax2.text(i + width/2, s_val + 0.003, f"{s_val:.4f}", ha="center", fontsize=8.5, fontweight="bold", color=GRUVBOX["fg0"])
            ax2.grid(True, linestyle=":", alpha=0.5, axis="y")

            # Panel 3: Peak Summit MAE (Raw vs Shifted)
            c_raw_pk = overall_conv_raw["Peak Summit MAE"][0]
            n_raw_pk = overall_nn_raw["Peak Summit MAE"][0]
            c_sft_pk = overall_conv_shifted["Peak Summit MAE"][0]
            n_sft_pk = overall_nn_shifted["Peak Summit MAE"][0]

            ax3.bar(x - width/2, [c_raw_pk, n_raw_pk], width, label="Raw", color=GRUVBOX["gray"], alpha=0.85)
            ax3.bar(x + width/2, [c_sft_pk, n_sft_pk], width, label="Shifted", color=[GRUVBOX["orange"], GRUVBOX["green"]])
            ax3.set_xticks(x)
            ax3.set_xticklabels(methods, fontweight="bold")
            ax3.set_title("Peak Summit MAE (↓)", fontweight="bold")
            for i, (r_val, s_val) in enumerate(zip([c_raw_pk, n_raw_pk], [c_sft_pk, n_sft_pk])):
                ax3.text(i - width/2, r_val * 1.02, f"{r_val:.4f}", ha="center", fontsize=8, color=GRUVBOX["fg1"])
                ax3.text(i + width/2, s_val * 1.02, f"{s_val:.4f}", ha="center", fontsize=8.5, fontweight="bold", color=GRUVBOX["fg0"])
            ax3.grid(True, linestyle=":", alpha=0.5, axis="y")

            # Panel 4: Processing Throughput
            tp_vals = [conv_throughput, nn_throughput]
            ax4.bar(methods, tp_vals, color=[GRUVBOX["orange"], GRUVBOX["green"]], width=0.55)
            ax4.set_title(f"Throughput ({speedup:.1f}x Speedup ↑)", fontweight="bold")
            for i, v in enumerate(tp_vals):
                ax4.text(i, v * 1.02, f"{v:.0f} s/s", ha="center", fontweight="bold", fontsize=9)
            ax4.grid(True, linestyle=":", alpha=0.5, axis="y")

            plt.suptitle(
                f"Round 6 Head-to-Head Benchmark: Raw vs. Baseline-Shifted Validation (N = {total_evaluated_spectra:,} Spectra)",
                fontsize=13, fontweight="bold", y=1.02
            )
            plt.tight_layout()

            plot2_path = os.path.join(SCRIPT_DIR, DEFAULT_PLOT_METRICS)
            plt.savefig(plot2_path, dpi=160)
            plt.close()
            log(f"✓ Saved metrics benchmark chart: '{plot2_path}'")

    log("\nBenchmark evaluation completed successfully!\n")


if __name__ == "__main__":
    main()
