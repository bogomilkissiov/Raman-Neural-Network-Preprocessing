"""
TESTING & BENCHMARKING SCRIPT - ROUND 5 (PolyGaussNet3+ vs. Conventional Pipeline)
---------------------------------------------------------------------------------
Head-to-head evaluation:
  1. PolyGaussNet3+ (Round 5 Architecture: Input-Size Agnostic, Degree 7 Baseline,
     Full-Resolution Dilated Residual CNN predicting per-bin [sigma, bounded amplitude],
     and Differentiable Adaptive Gaussian Filter Denoising K=63)
  2. Conventional Preprocessing Pipeline (test_files/conventional/pre.py: Despike + Wavelet + Baseline)

EVALUATES ACROSS MULTI-RESOLUTION TEST SETS (testing_data5):
1. Cosine Similarity vs. Ground Truth Pure Spectra (↑ Higher is better)
2. Mean Squared Error (MSE) vs. Ground Truth Pure Spectra (↓ Lower is better)
3. Numerically Stable Log-Cosh Loss vs. Ground Truth Pure Spectra (↓ Lower is better)
4. Mean Absolute Error (MAE) & Root Mean Squared Error (RMSE)
5. Execution Time, Processing Throughput (spectra/s), and Latency (ms/spectrum)
6. Speedup Factor of PolyGaussNet3+ relative to Conventional Preprocessing
7. Granular breakdowns across optical stretching modes (1x, 1.5x, 2x) and bin lengths (800 to 3200 bins)

USAGE:
  # 1. Run full evaluation across all multi-resolution test spectra:
  python testing5.py

  # 2. Run fast intermittent validation on a subset (e.g. 1024 spectra total or 128 per chunk):
  python testing5.py --max-samples 1024
  python testing5.py --max-samples-per-chunk 128

  # 3. Evaluate specific optical mode (e.g. 1x only or 2x only):
  python testing5.py --modes 1x

  # 4. Neural-network-only inference (skip conventional CPU preprocessing):
  python testing5.py --no-conv
"""

import os
import sys
import time
import glob
import argparse
import importlib
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
import torch.nn.functional as F

# Configure paths so imports resolve cleanly whether running from project root or inside training_round5
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
CONVENTIONAL_DIR = os.path.join(PROJECT_ROOT, "test_files", "conventional")

for p in [PROJECT_ROOT, SCRIPT_DIR, CONVENTIONAL_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Imports from project
from spectra_class import spectra
import pre

# Dynamically import PolyGaussNet from polygaussnet3+ module (since '+' is non-standard identifier)
try:
    polygaussnet3_plus = importlib.import_module("polygaussnet3+")
    PolyGaussNet = polygaussnet3_plus.PolyGaussNet
except ImportError as e:
    raise ImportError(f"Could not import PolyGaussNet from 'polygaussnet3+.py': {e}")


# =====================================================================
# 1. CONFIGURATION & CLI ARGUMENT PARSER
# =====================================================================
DEFAULT_DATA_DIR = "testing_data5"
DEFAULT_MODEL_PATH = "polygaussnet5.pth"
DEFAULT_CHECKPOINT_PATH = "training_checkpoint5.pth"
DEFAULT_BATCH_SIZE = 512
DEFAULT_N_JOBS = min(os.cpu_count() or 4, 8)
DEFAULT_MODES = ["1x", "1.5x", "2x"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Head-to-head evaluation: PolyGaussNet3+ vs Conventional Preprocessing Pipeline (Round 5)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR,
                        help="Directory containing multi-resolution test dataset folders (1x, 1.5x, 2x)")
    parser.add_argument("--model-path", "--model", type=str, default=DEFAULT_MODEL_PATH,
                        dest="model_path", help="Path to trained PolyGaussNet3+ model weights (.pth)")
    parser.add_argument("--checkpoint-path", type=str, default=DEFAULT_CHECKPOINT_PATH,
                        help="Path to training checkpoint (used for epoch metadata or fallback weights)")
    parser.add_argument("--modes", nargs="+", default=DEFAULT_MODES,
                        help="Optical stretching modes to evaluate (1x, 1.5x, 2x)")
    parser.add_argument("--bin-sizes", type=int, nargs="+", default=None,
                        help="Filter evaluation to specific bin resolutions (e.g. 800 1016 1200)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum total test spectra to evaluate across all test files")
    parser.add_argument("--max-samples-per-chunk", type=int, default=None,
                        help="Maximum number of test spectra to evaluate per chunk file")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help="Mini-batch size for PyTorch neural network inference")
    parser.add_argument("--n-jobs", type=int, default=DEFAULT_N_JOBS,
                        help="Number of CPU worker processes for conventional pipeline")
    parser.add_argument("--no-conv", action="store_true", default=False,
                        help="Skip running the conventional preprocessing pipeline")
    parser.add_argument("--poly-order", type=int, default=7,
                        help="Polynomial order for PolyGaussNet3+ baseline estimator (Degree 7)")
    parser.add_argument("--filter-kernel-size", type=int, default=63,
                        help="Odd kernel window size for adaptive Gaussian filter (K=63)")
    parser.add_argument("--min-sigma", type=float, default=0.2,
                        help="Minimum sigma bound for adaptive Gaussian filter")
    parser.add_argument("--max-sigma", type=float, default=10.0,
                        help="Maximum sigma bound for adaptive Gaussian filter")
    parser.add_argument("--min-amplitude", type=float, default=0.85,
                        help="Minimum bounded amplitude modulation")
    parser.add_argument("--max-amplitude", type=float, default=1.15,
                        help="Maximum bounded amplitude modulation")
    parser.add_argument("--save-report", type=str, default="testing5_report.txt",
                        help="Filename to save the formatted text benchmark report (set empty to disable)")
    return parser.parse_args()


# =====================================================================
# 2. NUMERICAL METRIC FUNCTIONS
# =====================================================================
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
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


def aggregate_metrics_from_arrays(arrays_list: list[dict]) -> dict:
    """Combines multiple chunk metric arrays and computes global statistics."""
    if not arrays_list:
        return None

    cos_all = np.concatenate([a["cosine_sim"] for a in arrays_list])
    mse_all = np.concatenate([a["mse"] for a in arrays_list])
    lc_all = np.concatenate([a["log_cosh"] for a in arrays_list])
    mae_all = np.concatenate([a["mae"] for a in arrays_list])
    rmse_all = np.concatenate([a["rmse"] for a in arrays_list])

    return {
        "Cosine Similarity": (float(np.mean(cos_all)), float(np.std(cos_all))),
        "MSE": (float(np.mean(mse_all)), float(np.std(mse_all))),
        "Log-Cosh": (float(np.mean(lc_all)), float(np.std(lc_all))),
        "MAE": (float(np.mean(mae_all)), float(np.std(mae_all))),
        "RMSE": (float(np.mean(rmse_all)), float(np.std(rmse_all))),
        "_arrays": {
            "cosine_sim": cos_all,
            "mse": mse_all,
            "log_cosh": lc_all,
            "mae": mae_all,
            "rmse": rmse_all
        }
    }


# =====================================================================
# 3. CONVENTIONAL PIPELINE WORKER FUNCTION (MULTIPROCESSING)
# =====================================================================
def _process_chunk_worker(wavenumbers_chunk: np.ndarray, raw_chunk: np.ndarray) -> np.ndarray:
    """Worker function for running conventional preprocessing on a chunk of spectra."""
    conv_data = spectra.from_matrices(wavenumbers_chunk, raw_chunk)
    pre.preprocess_pipeline(conv_data, normalize=False, shift=False)
    return conv_data.intensity_matrix


def run_conventional_pipeline(raw_np: np.ndarray, wavenumbers_np: np.ndarray, n_jobs: int = 4) -> np.ndarray:
    """
    Runs conventional pipeline (Despike -> BayesShrink Wavelet -> Baseline Removal)
    across raw spectra in parallel across CPU cores.
    """
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
# 4. DATASET DISCOVERY HELPERS
# =====================================================================
def resolve_path(path_str: str) -> str:
    """Resolves file or directory path checking script dir and project root."""
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

    round5_rel = os.path.join(PROJECT_ROOT, "training_round5", path_str)
    if os.path.exists(round5_rel):
        return round5_rel

    return path_str


def discover_test_files(data_dir: str, allowed_modes: list[str], allowed_bins: list[int] = None) -> list[dict]:
    """
    Discovers all test .npz files inside testing_data5 organized by mode and bin resolution.
    Returns list of metadata dicts: [{'path': str, 'mode': str, 'bin_size': int, 'name': str}, ...]
    """
    data_dir = resolve_path(data_dir)
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Test data directory '{data_dir}' not found.")

    discovered = []
    # Search for files matching pattern: data_dir/{mode}/{bin_size}/*.npz
    all_npz = sorted(glob.glob(os.path.join(data_dir, "**", "*.npz"), recursive=True))

    for fpath in all_npz:
        rel = os.path.relpath(fpath, data_dir)
        parts = rel.split(os.sep)

        mode = "unknown"
        bin_size = None

        if len(parts) >= 3:
            mode = parts[0]
            try:
                bin_size = int(parts[1])
            except ValueError:
                bin_size = None
        elif len(parts) == 2:
            mode = parts[0]

        if allowed_modes and mode != "unknown" and mode not in allowed_modes:
            continue

        if allowed_bins is not None and bin_size is not None and bin_size not in allowed_bins:
            continue

        discovered.append({
            "path": fpath,
            "mode": mode,
            "bin_size": bin_size,
            "rel_path": rel,
            "filename": os.path.basename(fpath)
        })

    return discovered


# =====================================================================
# 5. MAIN BENCHMARK & EVALUATION ROUTINE
# =====================================================================
def main():
    args = parse_args()

    output_lines = []
    def log(msg: str = ""):
        print(msg)
        output_lines.append(msg)

    log("\n" + "=" * 82)
    log(" PolyGaussNet3+ vs. Conventional Pipeline Head-to-Head Benchmark (Round 5) ")
    log("=" * 82)

    # -----------------------------------------------------------------
    # 5.1 Discover Test Files
    # -----------------------------------------------------------------
    test_files = discover_test_files(args.data_dir, args.modes, args.bin_sizes)
    if not test_files:
        raise FileNotFoundError(
            f"No valid test dataset files found in '{args.data_dir}' for modes={args.modes}."
        )

    # Summarize discovered test sets
    modes_found = sorted(list(set(f["mode"] for f in test_files)))
    log(f"Discovered {len(test_files)} test file(s) across optical modes {modes_found}:")
    for m in modes_found:
        m_files = [f for f in test_files if f["mode"] == m]
        bins_in_mode = sorted(list(set(f["bin_size"] for f in m_files if f["bin_size"] is not None)))
        log(f"  • Mode '{m}': {len(m_files)} file(s), resolutions: {bins_in_mode} bins")

    # Hardware acceleration detection
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    log(f"\nHardware Acceleration Device: {device}")

    # -----------------------------------------------------------------
    # 5.2 Load PolyGaussNet3+ Model
    # -----------------------------------------------------------------
    log("\n" + "-" * 82)
    log("1. Loading PolyGaussNet3+ Model Weights...")
    log("-" * 82)

    model_candidates = [
        args.model_path,
        resolve_path(args.model_path),
        os.path.join(SCRIPT_DIR, "polygaussnet5.pth"),
        os.path.join(SCRIPT_DIR, "training_checkpoint5.pth"),
        resolve_path(args.checkpoint_path)
    ]
    model_path = next((p for p in model_candidates if os.path.exists(p)), None)
    if model_path is None:
        raise FileNotFoundError(f"PolyGaussNet3+ model weights file not found. Checked: {model_candidates}")

    ckpt = torch.load(model_path, map_location=device, weights_only=False)

    # Determine state dict & metadata
    poly_order = args.poly_order
    filter_kernel_size = args.filter_kernel_size
    min_sigma = args.min_sigma
    max_sigma = args.max_sigma
    min_amplitude = args.min_amplitude
    max_amplitude = args.max_amplitude
    trained_epoch = None
    best_loss = None

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        poly_order = ckpt.get("poly_order", poly_order)
        filter_kernel_size = ckpt.get("filter_kernel_size", filter_kernel_size)
        min_sigma = ckpt.get("min_sigma", min_sigma)
        max_sigma = ckpt.get("max_sigma", max_sigma)
        min_amplitude = ckpt.get("min_amplitude", min_amplitude)
        max_amplitude = ckpt.get("max_amplitude", max_amplitude)
        trained_epoch = ckpt.get("epoch", None)
        best_loss = ckpt.get("best_loss", ckpt.get("loss", None))
    else:
        state_dict = ckpt
        # Also check checkpoint file for training epoch metadata
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
        poly_order=poly_order,
        filter_kernel_size=filter_kernel_size,
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        min_amplitude=min_amplitude,
        max_amplitude=max_amplitude
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    epoch_info_str = f" [Checkpoint Epoch: {trained_epoch + 1}, Loss: {best_loss:.6f}]" if trained_epoch is not None else ""
    log(f"✓ Loaded PolyGaussNet3+ from '{model_path}'{epoch_info_str}")
    log(f"  Configuration: Degree {poly_order} Polynomial | Kernel {filter_kernel_size} | "
        f"Sigma [{min_sigma}, {max_sigma}] | Amplitude [{min_amplitude}, {max_amplitude}]")

    # Warmup forward pass
    dummy_x = torch.randn(min(32, args.batch_size), 1016, device=device)
    with torch.no_grad():
        _ = model(dummy_x)
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()

    # -----------------------------------------------------------------
    # 5.3 Multi-Resolution Evaluation Loop
    # -----------------------------------------------------------------
    log("\n" + "-" * 82)
    log(f"2. Running Head-to-Head Evaluation Across Multi-Resolution Datasets...")
    if args.no_conv:
        log("   (Skipping Conventional Pipeline as --no-conv was requested)")
    else:
        log(f"   (Conventional Pipeline CPU Workers: {args.n_jobs})")
    log("-" * 82)

    total_files = len(test_files)
    total_evaluated_spectra = 0

    # Timing accumulators
    total_conv_time = 0.0
    total_nn_time = 0.0

    # Metric array accumulators:
    # All spectra
    all_nn_metric_arrays = []
    all_conv_metric_arrays = []

    # Grouped by mode: mode -> list[arrays]
    mode_nn_arrays = {m: [] for m in modes_found}
    mode_conv_arrays = {m: [] for m in modes_found}
    mode_samples = {m: 0 for m in modes_found}

    # Grouped by resolution: bin_size -> list[arrays]
    res_nn_arrays = {}
    res_conv_arrays = {}
    res_samples = {}
    res_modes = {}

    # Sample budget calculation if max_samples is provided
    samples_per_chunk_target = None
    if args.max_samples_per_chunk is not None:
        samples_per_chunk_target = args.max_samples_per_chunk
    elif args.max_samples is not None:
        samples_per_chunk_target = max(16, int(np.ceil(args.max_samples / total_files)))

    for file_idx, f_info in enumerate(test_files, start=1):
        fpath = f_info["path"]
        mode = f_info["mode"]

        # Load file data
        with np.load(fpath) as data:
            raw_mat = data["full_matrix"]
            pure_mat = data["pure_matrix"]
            bc_key = "pure_noise_cosmic_matrix" if "pure_noise_cosmic_matrix" in data else "pure_matrix"
            bc_mat = data[bc_key]

        # Slice sample budget
        n_file_samples = len(raw_mat)
        if samples_per_chunk_target is not None and samples_per_chunk_target < n_file_samples:
            raw_mat = raw_mat[:samples_per_chunk_target]
            pure_mat = pure_mat[:samples_per_chunk_target]
            bc_mat = bc_mat[:samples_per_chunk_target]

        # Check total max samples budget
        if args.max_samples is not None and (total_evaluated_spectra + len(raw_mat)) > args.max_samples:
            remaining = args.max_samples - total_evaluated_spectra
            if remaining <= 0:
                break
            raw_mat = raw_mat[:remaining]
            pure_mat = pure_mat[:remaining]
            bc_mat = bc_mat[:remaining]

        n_samples, seq_len = raw_mat.shape
        bin_size = seq_len
        f_info["bin_size"] = bin_size

        total_evaluated_spectra += n_samples
        mode_samples[mode] = mode_samples.get(mode, 0) + n_samples
        res_samples[bin_size] = res_samples.get(bin_size, 0) + n_samples
        res_modes[bin_size] = mode

        # Coordinate grid for conventional pipeline
        wavenumbers_mat = np.tile(np.arange(seq_len, dtype=np.float32), (n_samples, 1))

        # --- A. Conventional Pipeline ---
        conv_clean_mat = None
        conv_chunk_time = 0.0
        m_conv = None

        if not args.no_conv:
            t0_c = time.perf_counter()
            conv_clean_mat = run_conventional_pipeline(raw_mat, wavenumbers_mat, n_jobs=args.n_jobs)
            conv_chunk_time = time.perf_counter() - t0_c
            total_conv_time += conv_chunk_time
            m_conv = compute_metrics(pure_mat, conv_clean_mat)

            all_conv_metric_arrays.append(m_conv["_arrays"])
            mode_conv_arrays[mode].append(m_conv["_arrays"])
            if bin_size not in res_conv_arrays:
                res_conv_arrays[bin_size] = []
            res_conv_arrays[bin_size].append(m_conv["_arrays"])

        # --- B. PolyGaussNet3+ Inference ---
        raw_tensor = torch.from_numpy(raw_mat).float()
        nn_clean_list = []

        t0_nn = time.perf_counter()
        with torch.no_grad():
            for b_i in range(0, n_samples, args.batch_size):
                batch_in = raw_tensor[b_i : b_i + args.batch_size].to(device)
                clean_pred, _, _, _ = model(batch_in)
                nn_clean_list.append(clean_pred.cpu())

            if device.type == "mps":
                torch.mps.synchronize()
            elif device.type == "cuda":
                torch.cuda.synchronize()

        nn_chunk_time = time.perf_counter() - t0_nn
        total_nn_time += nn_chunk_time
        nn_clean_mat = torch.cat(nn_clean_list, dim=0).numpy()

        m_nn = compute_metrics(pure_mat, nn_clean_mat)
        all_nn_metric_arrays.append(m_nn["_arrays"])
        mode_nn_arrays[mode].append(m_nn["_arrays"])
        if bin_size not in res_nn_arrays:
            res_nn_arrays[bin_size] = []
        res_nn_arrays[bin_size].append(m_nn["_arrays"])

        # Per-chunk progress log
        nn_tp = n_samples / nn_chunk_time if nn_chunk_time > 0 else 0
        nn_cos = m_nn["Cosine Similarity"][0]
        nn_mse = m_nn["MSE"][0]
        nn_lc = m_nn["Log-Cosh"][0]

        if not args.no_conv and m_conv is not None:
            c_tp = n_samples / conv_chunk_time if conv_chunk_time > 0 else 0
            c_cos = m_conv["Cosine Similarity"][0]
            c_mse = m_conv["MSE"][0]
            c_lc = m_conv["Log-Cosh"][0]
            spd = (conv_chunk_time / nn_chunk_time) if nn_chunk_time > 0 else 0.0

            log(
                f"[{file_idx:2d}/{total_files}] Mode: {mode:<4} | L={bin_size:<4} bins | N={n_samples:,} | "
                f"Conv: Cos={c_cos:.4f}, MSE={c_mse:.5f}, LC={c_lc:.5f} ({c_tp:5.0f} s/s) | "
                f"PGN3+: Cos={nn_cos:.4f}, MSE={nn_mse:.5f}, LC={nn_lc:.5f} ({nn_tp:6.0f} s/s, {spd:4.1f}x)"
            )
        else:
            log(
                f"[{file_idx:2d}/{total_files}] Mode: {mode:<4} | L={bin_size:<4} bins | N={n_samples:,} | "
                f"PGN3+: Cos={nn_cos:.4f}, MSE={nn_mse:.5f}, LC={nn_lc:.5f} ({nn_tp:6.0f} s/s)"
            )

    # -----------------------------------------------------------------
    # 5.4 Compute Aggregated Metrics
    # -----------------------------------------------------------------
    overall_nn = aggregate_metrics_from_arrays(all_nn_metric_arrays)
    overall_conv = aggregate_metrics_from_arrays(all_conv_metric_arrays) if not args.no_conv else None

    conv_throughput = total_evaluated_spectra / total_conv_time if total_conv_time > 0 else 0.0
    conv_latency_ms = (total_conv_time / total_evaluated_spectra) * 1000 if total_evaluated_spectra > 0 else 0.0

    nn_throughput = total_evaluated_spectra / total_nn_time if total_nn_time > 0 else 0.0
    nn_latency_ms = (total_nn_time / total_evaluated_spectra) * 1000 if total_evaluated_spectra > 0 else 0.0
    speedup = (total_conv_time / total_nn_time) if (total_conv_time > 0 and total_nn_time > 0) else 1.0

    # -----------------------------------------------------------------
    # 5.5 Print Grand Benchmark Report Table (Matching testing4.py)
    # -----------------------------------------------------------------
    def fmt_cell(val_std):
        if val_std is None:
            return "N/A"
        val, std = val_std
        return f"{val:9.5f} (±{std:.4f})"

    log("\n" + "=" * 82)
    log(f"{'HEAD-TO-HEAD BENCHMARK REPORT (N = ' + f'{total_evaluated_spectra:,}' + ' Spectra)':^82}")
    log("=" * 82)
    log(f"{'Metric / Parameter':<30} | {'Conventional (pre.py)':<24} | {'PolyGaussNet3+':<22}")
    log("-" * 82)

    # 1. Cosine Similarity
    c_cos_str = fmt_cell(overall_conv["Cosine Similarity"] if overall_conv else None)
    n_cos_str = fmt_cell(overall_nn["Cosine Similarity"])
    log(f"{'Cosine Similarity (↑)':<30} | {c_cos_str:<24} | {n_cos_str:<22}")

    # 2. MSE
    c_mse_str = fmt_cell(overall_conv["MSE"] if overall_conv else None)
    n_mse_str = fmt_cell(overall_nn["MSE"])
    log(f"{'Mean Squared Error (MSE ↓)':<30} | {c_mse_str:<24} | {n_mse_str:<22}")

    # 3. Log-Cosh
    c_lc_str = fmt_cell(overall_conv["Log-Cosh"] if overall_conv else None)
    n_lc_str = fmt_cell(overall_nn["Log-Cosh"])
    log(f"{'Log-Cosh Loss (↓)':<30} | {c_lc_str:<24} | {n_lc_str:<22}")

    # 4. MAE
    c_mae_str = fmt_cell(overall_conv["MAE"] if overall_conv else None)
    n_mae_str = fmt_cell(overall_nn["MAE"])
    log(f"{'Mean Absolute Error (MAE ↓)':<30} | {c_mae_str:<24} | {n_mae_str:<22}")

    # 5. RMSE
    c_rmse_str = fmt_cell(overall_conv["RMSE"] if overall_conv else None)
    n_rmse_str = fmt_cell(overall_nn["RMSE"])
    log(f"{'Root Mean Sq. Error (RMSE ↓)':<30} | {c_rmse_str:<24} | {n_rmse_str:<22}")

    log("-" * 82)
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
    log("=" * 82)

    # -----------------------------------------------------------------
    # 5.6 Breakdown by Optical Stretching Mode (1.0x vs 1.5x vs 2.0x)
    # -----------------------------------------------------------------
    log("\n" + "=" * 94)
    log(f"{'OPTICAL RESOLUTION MODE BREAKDOWN (1.0x vs. 1.5x vs. 2.0x)':^94}")
    log("=" * 94)
    log(f"{'Mode':<6} | {'Spectra':<8} | {'Conv Cos':<10} | {'PGN3+ Cos':<10} | {'Conv MSE':<11} | {'PGN3+ MSE':<11} | {'Conv LogCosh':<12} | {'PGN3+ LogCosh':<12}")
    log("-" * 94)

    for m in modes_found:
        if mode_samples.get(m, 0) == 0:
            continue
        n_m = mode_samples[m]
        m_nn_res = aggregate_metrics_from_arrays(mode_nn_arrays[m])
        m_conv_res = aggregate_metrics_from_arrays(mode_conv_arrays[m]) if not args.no_conv else None

        c_cos = f"{m_conv_res['Cosine Similarity'][0]:.5f}" if m_conv_res else "N/A"
        n_cos = f"{m_nn_res['Cosine Similarity'][0]:.5f}" if m_nn_res else "N/A"

        c_mse = f"{m_conv_res['MSE'][0]:.6f}" if m_conv_res else "N/A"
        n_mse = f"{m_nn_res['MSE'][0]:.6f}" if m_nn_res else "N/A"

        c_lc = f"{m_conv_res['Log-Cosh'][0]:.6f}" if m_conv_res else "N/A"
        n_lc = f"{m_nn_res['Log-Cosh'][0]:.6f}" if m_nn_res else "N/A"

        log(f"{m:<6} | {n_m:<8,d} | {c_cos:<10} | {n_cos:<10} | {c_mse:<11} | {n_mse:<11} | {c_lc:<12} | {n_lc:<12}")

    log("=" * 94)

    # -----------------------------------------------------------------
    # 5.7 Breakdown by Individual Spectral Bin Resolution
    # -----------------------------------------------------------------
    sorted_bins = sorted(res_samples.keys())
    log("\n" + "=" * 94)
    log(f"{'SPECTRAL BIN RESOLUTION BREAKDOWN (800 to 3200 Bins)':^94}")
    log("=" * 94)
    log(f"{'Bins (L)':<8} | {'Mode':<6} | {'Spectra':<8} | {'Conv Cos':<10} | {'PGN3+ Cos':<10} | {'Conv MSE':<11} | {'PGN3+ MSE':<11} | {'Conv LogCosh':<12} | {'PGN3+ LogCosh':<12}")
    log("-" * 94)

    for b in sorted_bins:
        n_b = res_samples[b]
        m_tag = res_modes.get(b, "-")
        b_nn_res = aggregate_metrics_from_arrays(res_nn_arrays[b])
        b_conv_res = aggregate_metrics_from_arrays(res_conv_arrays[b]) if not args.no_conv else None

        c_cos = f"{b_conv_res['Cosine Similarity'][0]:.5f}" if b_conv_res else "N/A"
        n_cos = f"{b_nn_res['Cosine Similarity'][0]:.5f}" if b_nn_res else "N/A"

        c_mse = f"{b_conv_res['MSE'][0]:.6f}" if b_conv_res else "N/A"
        n_mse = f"{b_nn_res['MSE'][0]:.6f}" if b_nn_res else "N/A"

        c_lc = f"{b_conv_res['Log-Cosh'][0]:.6f}" if b_conv_res else "N/A"
        n_lc = f"{b_nn_res['Log-Cosh'][0]:.6f}" if b_nn_res else "N/A"

        log(f"{b:<8} | {m_tag:<6} | {n_b:<8,d} | {c_cos:<10} | {n_cos:<10} | {c_mse:<11} | {n_mse:<11} | {c_lc:<12} | {n_lc:<12}")

    log("=" * 94 + "\n")

    # -----------------------------------------------------------------
    # 5.8 Save Text Report (if configured)
    # -----------------------------------------------------------------
    if args.save_report:
        report_dest = args.save_report
        if not os.path.isabs(report_dest):
            report_dest = os.path.join(SCRIPT_DIR, args.save_report)
        try:
            with open(report_dest, "w") as f_rep:
                f_rep.write("\n".join(output_lines) + "\n")
            log(f"✓ Saved benchmark report to: '{report_dest}'\n")
        except Exception as e:
            log(f"[Warning] Could not write report file: {e}")


if __name__ == "__main__":
    main()
