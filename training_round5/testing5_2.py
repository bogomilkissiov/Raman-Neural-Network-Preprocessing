"""
TESTING & BENCHMARKING SCRIPT - 3-WAY COMPARISON (ROUND 5.2)
------------------------------------------------------------
Head-to-head 3-way evaluation on 1x Optical Stretching Mode Dataset (testing_data5/1x):
  1. Conventional Preprocessing Pipeline (test_files/conventional/pre.py: Despike + Wavelet + Baseline)
  2. PolyGaussNet2 (Round 4: Degree 7 Baseline, Adaptive Sigma-only Gaussian Filter K=31, Unit Amplitude)
  3. PolyGaussNet3+ (Round 5: Degree 7 Baseline, Adaptive [Sigma + Bounded Amplitude] Gaussian Filter K=63)

DATASET:
  Evaluates across 1x optical stretching mode test data (800, 1016, 1200, 1400, 1600 bins).
  Note: L=1016 corresponds to the native Round 4 spectral resolution.

EVALUATES:
1. Cosine Similarity vs. Ground Truth Pure Spectra (↑ Higher is better)
2. Mean Squared Error (MSE) vs. Ground Truth Pure Spectra (↓ Lower is better)
3. Numerically Stable Log-Cosh Loss vs. Ground Truth Pure Spectra (↓ Lower is better)
4. Mean Absolute Error (MAE) & Root Mean Squared Error (RMSE)
5. Execution Time, Processing Throughput (spectra/s), and Latency (ms/spectrum)
6. Speedup Factors relative to Conventional Preprocessing
7. Granular breakdown across each 1x bin resolution (800, 1016, 1200, 1400, 1600)

USAGE:
  # 1. Run full 3-way evaluation on 1x mode:
  python testing5_2.py

  # 2. Run fast intermittent evaluation on 64 spectra per resolution:
  python testing5_2.py --max-samples-per-chunk 64

  # 3. Evaluate specifically on native Round 4 resolution (1016 bins):
  python testing5_2.py --bin-sizes 1016
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
ROUND4_DIR = os.path.join(PROJECT_ROOT, "training_round4")

for p in [PROJECT_ROOT, SCRIPT_DIR, CONVENTIONAL_DIR, ROUND4_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

# Imports from project
from spectra_class import spectra
import pre
from polygaussnet2 import PolyGaussNet as PolyGaussNet2

# Dynamically import PolyGaussNet from polygaussnet3+ module
try:
    polygaussnet3_plus = importlib.import_module("polygaussnet3+")
    PolyGaussNet3Plus = polygaussnet3_plus.PolyGaussNet
except ImportError as e:
    raise ImportError(f"Could not import PolyGaussNet from 'polygaussnet3+.py': {e}")


# =====================================================================
# 1. CONFIGURATION & CLI ARGUMENT PARSER
# =====================================================================
DEFAULT_DATA_DIR = "testing_data5"
DEFAULT_MODEL_P2 = os.path.join(ROUND4_DIR, "polygaussnet2.pth")
DEFAULT_MODEL_P3 = os.path.join(SCRIPT_DIR, "polygaussnet5.pth")
DEFAULT_BATCH_SIZE = 512
DEFAULT_N_JOBS = min(os.cpu_count() or 4, 8)


def parse_args():
    parser = argparse.ArgumentParser(
        description="3-Way Benchmark: Conventional vs. PolyGaussNet2 (Round 4) vs. PolyGaussNet3+ (Round 5)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR,
                        help="Directory containing test dataset (evaluates 1x mode subfolder)")
    parser.add_argument("--model-p2", type=str, default=DEFAULT_MODEL_P2,
                        help="Path to trained PolyGaussNet2 model weights (.pth)")
    parser.add_argument("--model-p3", type=str, default=DEFAULT_MODEL_P3,
                        help="Path to trained PolyGaussNet3+ model weights (.pth)")
    parser.add_argument("--bin-sizes", type=int, nargs="+", default=None,
                        help="Filter evaluation to specific bin resolutions (e.g. 1016 or 800 1016)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum total test spectra to evaluate across 1x files")
    parser.add_argument("--max-samples-per-chunk", type=int, default=None,
                        help="Maximum test spectra to evaluate per chunk file")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help="Mini-batch size for PyTorch neural network inference")
    parser.add_argument("--n-jobs", type=int, default=DEFAULT_N_JOBS,
                        help="Number of CPU worker processes for conventional pipeline")
    parser.add_argument("--no-conv", action="store_true", default=False,
                        help="Skip running the conventional preprocessing pipeline")
    parser.add_argument("--save-report", type=str, default="testing5_2_report.txt",
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
# 3. CONVENTIONAL PIPELINE WORKER (MULTIPROCESSING)
# =====================================================================
def _process_chunk_worker(wavenumbers_chunk: np.ndarray, raw_chunk: np.ndarray) -> np.ndarray:
    """Worker function for running conventional preprocessing on a chunk of spectra."""
    conv_data = spectra.from_matrices(wavenumbers_chunk, raw_chunk)
    pre.preprocess_pipeline(conv_data, normalize=False, shift=False)
    return conv_data.intensity_matrix


def run_conventional_pipeline(raw_np: np.ndarray, wavenumbers_np: np.ndarray, n_jobs: int = 4) -> np.ndarray:
    """Runs conventional pipeline across CPU cores."""
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
# 4. PATH RESOLUTION & DATA DISCOVERY
# =====================================================================
def resolve_path(path_str: str) -> str:
    """Resolves file or directory path checking script dir and project root."""
    if os.path.isabs(path_str) and os.path.exists(path_str):
        return path_str
    if os.path.exists(path_str):
        return os.path.abspath(path_str)

    candidates = [
        os.path.join(SCRIPT_DIR, path_str),
        os.path.join(PROJECT_ROOT, path_str),
        os.path.join(ROUND4_DIR, path_str),
        os.path.join(PROJECT_ROOT, "training_round5", path_str)
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)

    return path_str


def discover_1x_files(data_dir: str, allowed_bins: list[int] = None) -> list[dict]:
    """Discovers test .npz files inside testing_data5/1x."""
    base = resolve_path(data_dir)
    mode_1x_path = os.path.join(base, "1x") if not base.endswith("1x") else base
    if not os.path.exists(mode_1x_path):
        mode_1x_path = base

    all_npz = sorted(glob.glob(os.path.join(mode_1x_path, "**", "*.npz"), recursive=True))
    if not all_npz:
        raise FileNotFoundError(f"No .npz files found in '{mode_1x_path}'.")

    discovered = []
    for fpath in all_npz:
        rel = os.path.relpath(fpath, mode_1x_path)
        parts = rel.split(os.sep)
        bin_size = None
        if len(parts) >= 2:
            try:
                bin_size = int(parts[0])
            except ValueError:
                bin_size = None

        if allowed_bins is not None and bin_size is not None and bin_size not in allowed_bins:
            continue

        discovered.append({
            "path": fpath,
            "bin_size": bin_size,
            "rel_path": rel,
            "filename": os.path.basename(fpath)
        })

    # Sort by bin_size
    discovered.sort(key=lambda d: (d["bin_size"] if d["bin_size"] is not None else 0, d["path"]))
    return discovered


# =====================================================================
# 5. MAIN 3-WAY BENCHMARK ROUTINE
# =====================================================================
def main():
    args = parse_args()

    output_lines = []
    def log(msg: str = ""):
        print(msg)
        output_lines.append(msg)

    log("\n" + "=" * 105)
    log(" 3-Way Benchmark: Conventional vs. PolyGaussNet2 (Round 4) vs. PolyGaussNet3+ (Round 5) ")
    log("=" * 105)

    # -----------------------------------------------------------------
    # 5.1 Discover 1x Mode Test Files
    # -----------------------------------------------------------------
    test_files = discover_1x_files(args.data_dir, args.bin_sizes)
    log(f"Evaluating on 1x Optical Stretching Mode Dataset ({len(test_files)} file(s)):")
    for f in test_files:
        bin_str = f"{f['bin_size']} bins" if f['bin_size'] else "Unknown resolution"
        log(f"  • {f['rel_path']} ({bin_str})")

    # Hardware device detection
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    log(f"\nHardware Acceleration Device: {device}")

    # -----------------------------------------------------------------
    # 5.2 Load Models
    # -----------------------------------------------------------------
    log("\n" + "-" * 105)
    log("1. Loading Models...")
    log("-" * 105)

    # Model 1: PolyGaussNet2 (Round 4)
    model_p2_path = resolve_path(args.model_p2)
    if not os.path.exists(model_p2_path):
        raise FileNotFoundError(f"PolyGaussNet2 weights not found at '{model_p2_path}'.")

    ckpt2 = torch.load(model_p2_path, map_location=device, weights_only=False)
    state2 = ckpt2.get("model_state_dict", ckpt2) if isinstance(ckpt2, dict) else ckpt2
    poly_order_p2 = ckpt2.get("poly_order", 7) if isinstance(ckpt2, dict) else 7
    kernel_p2 = ckpt2.get("filter_kernel_size", 31) if isinstance(ckpt2, dict) else 31

    model_p2 = PolyGaussNet2(
        poly_order=poly_order_p2,
        filter_kernel_size=kernel_p2
    ).to(device)
    model_p2.load_state_dict(state2)
    model_p2.eval()
    log(f"✓ Loaded PolyGaussNet2 (Round 4) from '{model_p2_path}' (Degree {poly_order_p2}, Kernel {kernel_p2}, Unit Amp)")

    # Model 2: PolyGaussNet3+ (Round 5)
    model_p3_path = resolve_path(args.model_p3)
    if not os.path.exists(model_p3_path):
        # Fallback to checkpoint
        alt_p3 = resolve_path("training_checkpoint5.pth")
        if os.path.exists(alt_p3):
            model_p3_path = alt_p3
        else:
            raise FileNotFoundError(f"PolyGaussNet3+ weights not found at '{model_p3_path}'.")

    ckpt3 = torch.load(model_p3_path, map_location=device, weights_only=False)
    state3 = ckpt3.get("model_state_dict", ckpt3) if isinstance(ckpt3, dict) else ckpt3
    poly_order_p3 = ckpt3.get("poly_order", 7) if isinstance(ckpt3, dict) else 7
    kernel_p3 = ckpt3.get("filter_kernel_size", 63) if isinstance(ckpt3, dict) else 63
    min_sigma_p3 = ckpt3.get("min_sigma", 0.2) if isinstance(ckpt3, dict) else 0.2
    max_sigma_p3 = ckpt3.get("max_sigma", 10.0) if isinstance(ckpt3, dict) else 10.0
    min_amp_p3 = ckpt3.get("min_amplitude", 0.85) if isinstance(ckpt3, dict) else 0.85
    max_amp_p3 = ckpt3.get("max_amplitude", 1.15) if isinstance(ckpt3, dict) else 1.15
    epoch_p3 = ckpt3.get("epoch", None) if isinstance(ckpt3, dict) else None

    model_p3 = PolyGaussNet3Plus(
        poly_order=poly_order_p3,
        filter_kernel_size=kernel_p3,
        min_sigma=min_sigma_p3,
        max_sigma=max_sigma_p3,
        min_amplitude=min_amp_p3,
        max_amplitude=max_amp_p3
    ).to(device)
    model_p3.load_state_dict(state3)
    model_p3.eval()
    ep_str = f" [Epoch {epoch_p3 + 1}]" if epoch_p3 is not None else ""
    log(f"✓ Loaded PolyGaussNet3+ (Round 5) from '{model_p3_path}'{ep_str} (Degree {poly_order_p3}, Kernel {kernel_p3}, Amp [{min_amp_p3}, {max_amp_p3}])")

    # Warmup both models
    dummy = torch.randn(min(32, args.batch_size), 1016, device=device)
    with torch.no_grad():
        _ = model_p2(dummy)
        _ = model_p3(dummy)
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()

    # -----------------------------------------------------------------
    # 5.3 Evaluation Loop
    # -----------------------------------------------------------------
    log("\n" + "-" * 105)
    log("2. Running Head-to-Head 3-Way Benchmark...")
    if not args.no_conv:
        log(f"   (Conventional Pipeline CPU Workers: {args.n_jobs})")
    else:
        log("   (Skipping Conventional Pipeline as --no-conv was requested)")
    log("-" * 105)

    total_files = len(test_files)
    total_evaluated_spectra = 0

    # Accumulators
    time_conv = 0.0
    time_p2 = 0.0
    time_p3 = 0.0

    arrays_conv = []
    arrays_p2 = []
    arrays_p3 = []

    res_conv = {}
    res_p2 = {}
    res_p3 = {}
    res_counts = {}

    samples_per_chunk_target = None
    if args.max_samples_per_chunk is not None:
        samples_per_chunk_target = args.max_samples_per_chunk
    elif args.max_samples is not None:
        samples_per_chunk_target = max(16, int(np.ceil(args.max_samples / total_files)))

    for f_idx, f_info in enumerate(test_files, start=1):
        fpath = f_info["path"]

        with np.load(fpath) as data:
            raw_mat = data["full_matrix"]
            pure_mat = data["pure_matrix"]

        n_samples = len(raw_mat)
        if samples_per_chunk_target is not None and samples_per_chunk_target < n_samples:
            raw_mat = raw_mat[:samples_per_chunk_target]
            pure_mat = pure_mat[:samples_per_chunk_target]

        if args.max_samples is not None and (total_evaluated_spectra + len(raw_mat)) > args.max_samples:
            rem = args.max_samples - total_evaluated_spectra
            if rem <= 0:
                break
            raw_mat = raw_mat[:rem]
            pure_mat = pure_mat[:rem]

        n_samples, seq_len = raw_mat.shape
        bin_size = seq_len
        f_info["bin_size"] = bin_size
        total_evaluated_spectra += n_samples
        res_counts[bin_size] = res_counts.get(bin_size, 0) + n_samples

        wn_mat = np.tile(np.arange(seq_len, dtype=np.float32), (n_samples, 1))

        # --- A. Conventional Pipeline ---
        m_conv = None
        conv_chunk_time = 0.0
        if not args.no_conv:
            t0_c = time.perf_counter()
            conv_clean = run_conventional_pipeline(raw_mat, wn_mat, n_jobs=args.n_jobs)
            conv_chunk_time = time.perf_counter() - t0_c
            time_conv += conv_chunk_time
            m_conv = compute_metrics(pure_mat, conv_clean)
            arrays_conv.append(m_conv["_arrays"])
            if bin_size not in res_conv:
                res_conv[bin_size] = []
            res_conv[bin_size].append(m_conv["_arrays"])

        # --- B. PolyGaussNet2 (Round 4) ---
        raw_tensor = torch.from_numpy(raw_mat).float()
        p2_clean_list = []
        t0_p2 = time.perf_counter()
        with torch.no_grad():
            for b in range(0, n_samples, args.batch_size):
                b_in = raw_tensor[b : b + args.batch_size].to(device)
                pred_c, _, _, _ = model_p2(b_in)
                p2_clean_list.append(pred_c.cpu())

            if device.type == "mps":
                torch.mps.synchronize()
            elif device.type == "cuda":
                torch.cuda.synchronize()
        p2_chunk_time = time.perf_counter() - t0_p2
        time_p2 += p2_chunk_time
        p2_clean_np = torch.cat(p2_clean_list, dim=0).numpy()
        m_p2 = compute_metrics(pure_mat, p2_clean_np)
        arrays_p2.append(m_p2["_arrays"])
        if bin_size not in res_p2:
            res_p2[bin_size] = []
        res_p2[bin_size].append(m_p2["_arrays"])

        # --- C. PolyGaussNet3+ (Round 5) ---
        p3_clean_list = []
        t0_p3 = time.perf_counter()
        with torch.no_grad():
            for b in range(0, n_samples, args.batch_size):
                b_in = raw_tensor[b : b + args.batch_size].to(device)
                pred_c, _, _, _ = model_p3(b_in)
                p3_clean_list.append(pred_c.cpu())

            if device.type == "mps":
                torch.mps.synchronize()
            elif device.type == "cuda":
                torch.cuda.synchronize()
        p3_chunk_time = time.perf_counter() - t0_p3
        time_p3 += p3_chunk_time
        p3_clean_np = torch.cat(p3_clean_list, dim=0).numpy()
        m_p3 = compute_metrics(pure_mat, p3_clean_np)
        arrays_p3.append(m_p3["_arrays"])
        if bin_size not in res_p3:
            res_p3[bin_size] = []
        res_p3[bin_size].append(m_p3["_arrays"])

        # Progress line
        p2_cos = m_p2["Cosine Similarity"][0]
        p2_mse = m_p2["MSE"][0]
        p2_lc = m_p2["Log-Cosh"][0]

        p3_cos = m_p3["Cosine Similarity"][0]
        p3_mse = m_p3["MSE"][0]
        p3_lc = m_p3["Log-Cosh"][0]

        if not args.no_conv and m_conv is not None:
            c_cos = m_conv["Cosine Similarity"][0]
            c_mse = m_conv["MSE"][0]
            c_lc = m_conv["Log-Cosh"][0]
            log(
                f"[{f_idx:2d}/{total_files}] L={bin_size:<4} bins | N={n_samples:,} | "
                f"Conv: Cos={c_cos:.4f}, LC={c_lc:.5f} | "
                f"PGN2: Cos={p2_cos:.4f}, LC={p2_lc:.5f} | "
                f"PGN3+: Cos={p3_cos:.4f}, LC={p3_lc:.5f}"
            )
        else:
            log(
                f"[{f_idx:2d}/{total_files}] L={bin_size:<4} bins | N={n_samples:,} | "
                f"PGN2: Cos={p2_cos:.4f}, LC={p2_lc:.5f} | "
                f"PGN3+: Cos={p3_cos:.4f}, LC={p3_lc:.5f}"
            )

    # -----------------------------------------------------------------
    # 5.4 Compute Aggregated Global Metrics
    # -----------------------------------------------------------------
    tot = total_evaluated_spectra
    overall_conv = aggregate_metrics_from_arrays(arrays_conv) if not args.no_conv else None
    overall_p2 = aggregate_metrics_from_arrays(arrays_p2)
    overall_p3 = aggregate_metrics_from_arrays(arrays_p3)

    conv_tp = tot / time_conv if time_conv > 0 else 0.0
    conv_lat = (time_conv / tot) * 1000 if tot > 0 else 0.0

    p2_tp = tot / time_p2 if time_p2 > 0 else 0.0
    p2_lat = (time_p2 / tot) * 1000 if tot > 0 else 0.0
    p2_spd = (time_conv / time_p2) if (time_conv > 0 and time_p2 > 0) else 1.0

    p3_tp = tot / time_p3 if time_p3 > 0 else 0.0
    p3_lat = (time_p3 / tot) * 1000 if tot > 0 else 0.0
    p3_spd = (time_conv / time_p3) if (time_conv > 0 and time_p3 > 0) else 1.0

    def fmt_cell(val_std):
        if val_std is None:
            return "N/A"
        val, std = val_std
        return f"{val:9.5f} (±{std:.4f})"

    # -----------------------------------------------------------------
    # 5.5 Grand Benchmark Report Table (Matching testing4.py)
    # -----------------------------------------------------------------
    log("\n" + "=" * 105)
    log(f"{'HEAD-TO-HEAD 3-WAY BENCHMARK REPORT (N = ' + f'{tot:,}' + ' Spectra)':^105}")
    log("=" * 105)
    log(f"{'Metric / Parameter':<30} | {'Conventional (pre.py)':<22} | {'PolyGaussNet2 (Rnd 4)':<22} | {'PolyGaussNet3+ (Rnd 5)':<22}")
    log("-" * 105)

    # 1. Cosine Similarity
    c_cos_s = fmt_cell(overall_conv["Cosine Similarity"] if overall_conv else None)
    p2_cos_s = fmt_cell(overall_p2["Cosine Similarity"])
    p3_cos_s = fmt_cell(overall_p3["Cosine Similarity"])
    log(f"{'Cosine Similarity (↑)':<30} | {c_cos_s:<22} | {p2_cos_s:<22} | {p3_cos_s:<22}")

    # 2. MSE
    c_mse_s = fmt_cell(overall_conv["MSE"] if overall_conv else None)
    p2_mse_s = fmt_cell(overall_p2["MSE"])
    p3_mse_s = fmt_cell(overall_p3["MSE"])
    log(f"{'Mean Squared Error (MSE ↓)':<30} | {c_mse_s:<22} | {p2_mse_s:<22} | {p3_mse_s:<22}")

    # 3. Log-Cosh
    c_lc_s = fmt_cell(overall_conv["Log-Cosh"] if overall_conv else None)
    p2_lc_s = fmt_cell(overall_p2["Log-Cosh"])
    p3_lc_s = fmt_cell(overall_p3["Log-Cosh"])
    log(f"{'Log-Cosh Loss (↓)':<30} | {c_lc_s:<22} | {p2_lc_s:<22} | {p3_lc_s:<22}")

    # 4. MAE
    c_mae_s = fmt_cell(overall_conv["MAE"] if overall_conv else None)
    p2_mae_s = fmt_cell(overall_p2["MAE"])
    p3_mae_s = fmt_cell(overall_p3["MAE"])
    log(f"{'Mean Absolute Error (MAE ↓)':<30} | {c_mae_s:<22} | {p2_mae_s:<22} | {p3_mae_s:<22}")

    # 5. RMSE
    c_rmse_s = fmt_cell(overall_conv["RMSE"] if overall_conv else None)
    p2_rmse_s = fmt_cell(overall_p2["RMSE"])
    p3_rmse_s = fmt_cell(overall_p3["RMSE"])
    log(f"{'Root Mean Sq. Error (RMSE ↓)':<30} | {c_rmse_s:<22} | {p2_rmse_s:<22} | {p3_rmse_s:<22}")

    log("-" * 105)
    c_dur_s = f"{time_conv:9.3f} s" if not args.no_conv else "N/A"
    p2_dur_s = f"{time_p2:9.3f} s"
    p3_dur_s = f"{time_p3:9.3f} s"
    log(f"{'Total Execution Time':<30} | {c_dur_s:<22} | {p2_dur_s:<22} | {p3_dur_s:<22}")

    c_tp_s = f"{conv_tp:9.1f} spec/s" if not args.no_conv else "N/A"
    p2_tp_s = f"{p2_tp:9.1f} spec/s"
    p3_tp_s = f"{p3_tp:9.1f} spec/s"
    log(f"{'Processing Throughput':<30} | {c_tp_s:<22} | {p2_tp_s:<22} | {p3_tp_s:<22}")

    c_lat_s = f"{conv_lat:9.2f} ms" if not args.no_conv else "N/A"
    p2_lat_s = f"{p2_lat:9.4f} ms"
    p3_lat_s = f"{p3_lat:9.4f} ms"
    log(f"{'Latency per Spectrum':<30} | {c_lat_s:<22} | {p2_lat_s:<22} | {p3_lat_s:<22}")

    c_spd_s = "1.0x (Baseline)" if not args.no_conv else "N/A"
    p2_spd_s = f"{p2_spd:.1f}x Faster" if not args.no_conv else "N/A"
    p3_spd_s = f"{p3_spd:.1f}x Faster" if not args.no_conv else "N/A"
    log(f"{'Speedup Factor':<30} | {c_spd_s:<22} | {p2_spd_s:<22} | {p3_spd_s:<22}")
    log("=" * 105)

    # -----------------------------------------------------------------
    # 5.6 Breakdown by Spectral Bin Resolution (800 to 1600 Bins)
    # -----------------------------------------------------------------
    sorted_bins = sorted(res_counts.keys())
    log("\n" + "=" * 105)
    log(f"{'RESOLUTION BREAKDOWN: CONVENTIONAL vs. POLYGAUSSNET2 vs. POLYGAUSSNET3+':^105}")
    log("=" * 105)
    log(f"{'Bins (L)':<9} | {'Spectra':<8} | {'Conv Cos':<9} | {'PGN2 Cos':<9} | {'PGN3+ Cos':<9} | {'Conv LC':<9} | {'PGN2 LC':<9} | {'PGN3+ LC':<9} | {'PGN3+ MSE':<9}")
    log("-" * 105)

    for b in sorted_bins:
        n_b = res_counts[b]
        b_c = aggregate_metrics_from_arrays(res_conv[b]) if not args.no_conv else None
        b_p2 = aggregate_metrics_from_arrays(res_p2[b])
        b_p3 = aggregate_metrics_from_arrays(res_p3[b])

        c_cos = f"{b_c['Cosine Similarity'][0]:.4f}" if b_c else "N/A"
        p2_cos = f"{b_p2['Cosine Similarity'][0]:.4f}"
        p3_cos = f"{b_p3['Cosine Similarity'][0]:.4f}"

        c_lc = f"{b_c['Log-Cosh'][0]:.5f}" if b_c else "N/A"
        p2_lc = f"{b_p2['Log-Cosh'][0]:.5f}"
        p3_lc = f"{b_p3['Log-Cosh'][0]:.5f}"

        p3_mse = f"{b_p3['MSE'][0]:.5f}"

        log(f"{b:<9} | {n_b:<8,d} | {c_cos:<9} | {p2_cos:<9} | {p3_cos:<9} | {c_lc:<9} | {p2_lc:<9} | {p3_lc:<9} | {p3_mse:<9}")

    log("=" * 105 + "\n")

    # -----------------------------------------------------------------
    # 5.7 Save Text Report
    # -----------------------------------------------------------------
    if args.save_report:
        rep_path = args.save_report
        if not os.path.isabs(rep_path):
            rep_path = os.path.join(SCRIPT_DIR, args.save_report)
        try:
            with open(rep_path, "w") as f_out:
                f_out.write("\n".join(output_lines) + "\n")
            log(f"✓ Saved 3-way benchmark report to: '{rep_path}'\n")
        except Exception as e:
            log(f"[Warning] Could not write report file: {e}")


if __name__ == "__main__":
    main()
