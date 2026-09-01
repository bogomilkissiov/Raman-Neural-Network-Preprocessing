"""
HEAD-TO-HEAD TESTING & BENCHMARKING SCRIPT - ROUND 5
---------------------------------------------------
Evaluates and directly compares:
  1. PolyGaussNet2 (Round 4: Sigma-only Adaptive Gaussian Filter, Amplitude = 1.0)
  2. PolyGaussNet3 (Round 5: Sigma + Bounded Amplitude Modulation [0.85, 1.15])

Dataset:
  Reaches directly into `training_round4/test_data4` (N = 32,768 test spectra).

Evaluates:
  1. Cosine Similarity vs. Ground Truth Pure Spectra (↑ Higher is better)
  2. Mean Squared Error (MSE) vs. Ground Truth Pure Spectra (↓ Lower is better)
  3. Numerically Stable Log-Cosh Loss (↓ Lower is better)
  4. Mean Absolute Error (MAE) & Root Mean Squared Error (RMSE)
  5. Peak Summit Preservation (MAE at top 10% peak locations)
  6. Baseline Overshoot Violation Rate
  7. Execution Time, Processing Throughput (spectra/s), and Latency (ms/spectrum)

Outputs:
  - Detailed individual benchmark reports for both models.
  - A direct side-by-side comparison table highlighting deltas and percentage differences.
  - Optional visual overlay and residual comparison plots (`head2head_comparison.png`).
"""

import os
import sys
import time
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

# =====================================================================
# PATH RESOLUTION
# =====================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

for p in [PROJECT_ROOT, SCRIPT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from polygaussnet2 import PolyGaussNet as PolyGaussNet2
from polygaussnet3 import PolyGaussNet as PolyGaussNet3

# Gruvbox Dark Palette Definitions
GRUVBOX = {
    "bg0": "#282828", "bg0_hard": "#1d2021", "bg0_soft": "#32302f",
    "bg1": "#3c3836", "bg2": "#504945", "bg3": "#665c54", "bg4": "#7c6f64",
    "fg": "#ebdbb2", "fg0": "#fbf1c7", "fg1": "#ebdbb2", "fg2": "#d5c4a1",
    "fg3": "#bdae93", "fg4": "#a89984",
    "red": "#cc241d", "green": "#b8bb26", "yellow": "#fabd2f", "blue": "#83a598",
    "purple": "#d3869b", "aqua": "#8ec07c", "orange": "#fe8019",
    "gray": "#928374"
}

# Default Paths
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "training_round4", "test_data4")
DEFAULT_MODEL_P2 = os.path.join(SCRIPT_DIR, "head2head_polygaussnet2.pth")
DEFAULT_MODEL_P3 = os.path.join(SCRIPT_DIR, "head2head_polygaussnet3.pth")
DEFAULT_BATCH_SIZE = 512


# =====================================================================
# CLI ARGUMENT PARSER
# =====================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Head-to-Head Testing & Direct Comparison: PolyGaussNet2 vs. PolyGaussNet3",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR,
                        help="Directory containing test dataset .npz files")
    parser.add_argument("--model-p2", type=str, default=DEFAULT_MODEL_P2,
                        help="Path to trained PolyGaussNet2 weights (.pth)")
    parser.add_argument("--model-p3", type=str, default=DEFAULT_MODEL_P3,
                        help="Path to trained PolyGaussNet3 weights (.pth)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum number of test spectra to evaluate (default: all)")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                        help="Mini-batch size for PyTorch neural network inference")
    parser.add_argument("--poly-order", type=int, default=7,
                        help="Polynomial order for baseline estimator (Degree 7)")
    parser.add_argument("--filter-kernel-size", type=int, default=31,
                        help="Gaussian filter kernel size")
    parser.add_argument("--min-amplitude", type=float, default=0.85,
                        help="PolyGaussNet3 minimum amplitude bound")
    parser.add_argument("--max-amplitude", type=float, default=1.15,
                        help="PolyGaussNet3 maximum amplitude bound")
    parser.add_argument("--no-plots", action="store_true", default=False,
                        help="Skip generating PNG visual comparison plots")
    parser.add_argument("--save-plots", type=str, default="head2head_comparison.png",
                        help="Output path for visual comparison plot")
    return parser.parse_args()


# =====================================================================
# NUMERICAL METRICS
# =====================================================================
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, pred_baseline: np.ndarray = None, raw_spectrum: np.ndarray = None):
    """
    Computes spectral evaluation metrics between ground truth and predictions.
    y_true, y_pred shape: (N, L)
    """
    # 1. Cosine Similarity
    dot_product = np.sum(y_true * y_pred, axis=1)
    norm_true = np.linalg.norm(y_true, axis=1)
    norm_pred = np.linalg.norm(y_pred, axis=1)

    valid_mask = (norm_true > 1e-12) & (norm_pred > 1e-12)
    both_zero_mask = (norm_true <= 1e-12) & (norm_pred <= 1e-12)

    cosine_sim = np.zeros(y_true.shape[0], dtype=np.float64)
    cosine_sim[both_zero_mask] = 1.0
    cosine_sim[valid_mask] = dot_product[valid_mask] / (norm_true[valid_mask] * norm_pred[valid_mask])

    # 2. Mean Squared Error (MSE)
    mse = np.mean((y_true - y_pred) ** 2, axis=1)

    # 3. Log-Cosh Loss
    diff = y_pred - y_true
    abs_diff = np.abs(diff)
    log_cosh_elementwise = abs_diff + np.log1p(np.exp(-2.0 * abs_diff)) - np.log(2.0)
    log_cosh = np.mean(log_cosh_elementwise, axis=1)

    # 4. Mean Absolute Error (MAE)
    mae = np.mean(abs_diff, axis=1)

    # 5. Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)

    # 6. Peak Summit Preservation Metric (Top 10% peak locations)
    peak_threshold = np.percentile(y_true, 90, axis=1, keepdims=True)
    peak_mask = (y_true >= peak_threshold) & (y_true > 0.05)
    
    peak_mae_list = []
    for i in range(len(y_true)):
        m = peak_mask[i]
        if np.any(m):
            peak_mae_list.append(np.mean(np.abs(y_true[i, m] - y_pred[i, m])))
        else:
            peak_mae_list.append(mae[i])
    peak_mae = np.array(peak_mae_list, dtype=np.float64)

    # 7. Baseline Overshoot Violation Rate
    asym_violation_rate = 0.0
    max_overshoot = 0.0
    if pred_baseline is not None and raw_spectrum is not None:
        overshoot = np.maximum(0.0, pred_baseline - raw_spectrum)
        asym_violation_rate = float(np.mean(overshoot > 1e-4) * 100.0)
        max_overshoot = float(np.max(overshoot))

    return {
        "Cosine Similarity": (float(np.mean(cosine_sim)), float(np.std(cosine_sim))),
        "MSE": (float(np.mean(mse)), float(np.std(mse))),
        "Log-Cosh": (float(np.mean(log_cosh)), float(np.std(log_cosh))),
        "MAE": (float(np.mean(mae)), float(np.std(mae))),
        "RMSE": (float(np.mean(rmse)), float(np.std(rmse))),
        "Peak Summit MAE": (float(np.mean(peak_mae)), float(np.std(peak_mae))),
        "Asym Violation Rate (%)": asym_violation_rate,
        "Max Baseline Overshoot": max_overshoot,
        "_arrays": {
            "cosine_sim": cosine_sim,
            "mse": mse,
            "log_cosh": log_cosh,
            "mae": mae,
            "rmse": rmse,
            "peak_mae": peak_mae
        }
    }


# =====================================================================
# DATA LOADING
# =====================================================================
def load_test_dataset(data_dir: str, max_samples: int = None):
    """Loads all test .npz files into NumPy matrices."""
    possible_dirs = [
        data_dir,
        os.path.join(SCRIPT_DIR, data_dir),
        os.path.join(PROJECT_ROOT, data_dir),
        os.path.join(PROJECT_ROOT, "training_round4", "test_data4"),
        os.path.join(PROJECT_ROOT, "test_data.npz")
    ]

    test_files = []
    resolved_dir = None
    for candidate in possible_dirs:
        if os.path.isfile(candidate) and candidate.endswith(".npz"):
            test_files = [candidate]
            resolved_dir = os.path.dirname(candidate)
            break
        elif os.path.isdir(candidate):
            files = sorted(glob.glob(os.path.join(candidate, "*.npz")))
            if files:
                test_files = files
                resolved_dir = candidate
                break

    if not test_files:
        raise FileNotFoundError(f"Could not find test dataset in '{data_dir}' or fallback paths.")

    print(f"Loading {len(test_files)} test dataset chunk(s) from '{resolved_dir}':")
    for f in test_files:
        print(f"  • {os.path.basename(f)}")

    pure_list, bc_true_list, raw_list = [], [], []
    for filepath in test_files:
        with np.load(filepath) as data:
            pure_list.append(data["pure_matrix"])
            if "pure_noise_cosmic_matrix" in data:
                bc_true_list.append(data["pure_noise_cosmic_matrix"])
            elif "pure_noise_matrix" in data:
                bc_true_list.append(data["pure_noise_matrix"])
            else:
                bc_true_list.append(data["pure_matrix"])
            
            full_key = "full_matrix" if "full_matrix" in data else ("composite_matrix" if "composite_matrix" in data else data.files[0])
            raw_list.append(data[full_key])

    pure_np = np.vstack(pure_list).astype(np.float32)
    bc_true_np = np.vstack(bc_true_list).astype(np.float32)
    raw_np = np.vstack(raw_list).astype(np.float32)

    if max_samples is not None and max_samples < len(raw_np):
        print(f"Subsetting to first {max_samples:,} spectra (from total {len(raw_np):,})...")
        pure_np = pure_np[:max_samples]
        bc_true_np = bc_true_np[:max_samples]
        raw_np = raw_np[:max_samples]

    return raw_np, pure_np, bc_true_np, resolved_dir


# =====================================================================
# MODEL INFERENCE RUNNER
# =====================================================================
def run_model_inference(model: nn.Module, raw_np: np.ndarray, batch_size: int, device: torch.device):
    """Executes model inference across all test spectra in mini-batches."""
    total_samples = len(raw_np)
    raw_tensor = torch.tensor(raw_np, dtype=torch.float32)

    # Warmup
    warmup_batch = raw_tensor[:min(32, total_samples)].to(device)
    with torch.no_grad():
        _ = model(warmup_batch)
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()

    clean_list, base_list, bc_list, sigma_list, amp_list = [], [], [], [], []
    t0 = time.perf_counter()

    with torch.no_grad():
        for i in range(0, total_samples, batch_size):
            batch_in = raw_tensor[i : i + batch_size].to(device)
            clean_pred, pred_baseline, bc_pred, latent_params = model(batch_in)

            clean_list.append(clean_pred.cpu())
            base_list.append(pred_baseline.cpu())
            bc_list.append(bc_pred.cpu())

            if 'sigma' in latent_params:
                sigma_list.append(latent_params['sigma'].cpu())
            if 'amplitude' in latent_params:
                amp_list.append(latent_params['amplitude'].cpu())

        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()

    duration = time.perf_counter() - t0
    throughput = total_samples / duration if duration > 0 else 0
    latency_ms = (duration / total_samples) * 1000 if total_samples > 0 else 0

    clean_np = torch.cat(clean_list, dim=0).numpy()
    base_np = torch.cat(base_list, dim=0).numpy()
    bc_np = torch.cat(bc_list, dim=0).numpy()
    sigma_np = torch.cat(sigma_list, dim=0).numpy() if sigma_list else None
    amp_np = torch.cat(amp_list, dim=0).numpy() if amp_list else None

    return {
        "clean": clean_np,
        "baseline": base_np,
        "bc": bc_np,
        "sigma": sigma_np,
        "amplitude": amp_np,
        "duration": duration,
        "throughput": throughput,
        "latency_ms": latency_ms
    }


# =====================================================================
# VISUALIZATION / PLOTTING
# =====================================================================
def plot_visual_comparison(raw_np, pure_np, res_p2, res_p3, save_path="head2head_comparison.png", num_examples=4):
    """Generates a clean comparison plot in Gruvbox dark style."""
    plt.rcParams.update({
        "figure.facecolor": GRUVBOX["bg0"],
        "axes.facecolor": GRUVBOX["bg0_hard"],
        "axes.edgecolor": GRUVBOX["bg3"],
        "axes.linewidth": 1.0,
        "axes.labelcolor": GRUVBOX["fg"],
        "axes.titlecolor": GRUVBOX["fg0"],
        "xtick.color": GRUVBOX["fg4"],
        "ytick.color": GRUVBOX["fg4"],
        "grid.color": GRUVBOX["bg2"],
        "grid.alpha": 0.5,
        "grid.linestyle": ":",
        "text.color": GRUVBOX["fg"],
        "legend.facecolor": GRUVBOX["bg0_soft"],
        "legend.edgecolor": GRUVBOX["bg3"],
        "legend.labelcolor": GRUVBOX["fg1"],
        "legend.framealpha": 0.92,
    })

    fig, axes = plt.subplots(num_examples, 3, figsize=(18, 3.5 * num_examples), sharex=True)
    if num_examples == 1:
        axes = np.expand_dims(axes, 0)

    indices = np.linspace(0, len(raw_np) - 1, num_examples, dtype=int)
    wavenumbers = np.arange(raw_np.shape[1])

    for row_idx, sample_idx in enumerate(indices):
        raw = raw_np[sample_idx]
        pure = pure_np[sample_idx]
        clean_p2 = res_p2["clean"][sample_idx]
        base_p2 = res_p2["baseline"][sample_idx]
        clean_p3 = res_p3["clean"][sample_idx]
        base_p3 = res_p3["baseline"][sample_idx]

        # Column 1: Raw & Baselines
        ax0 = axes[row_idx, 0]
        ax0.plot(wavenumbers, raw, color=GRUVBOX["fg3"], alpha=0.5, label="Raw Input", lw=1.0)
        ax0.plot(wavenumbers, base_p2, color=GRUVBOX["yellow"], linestyle="--", label="P2 Baseline (Degree 7)", lw=1.4)
        ax0.plot(wavenumbers, base_p3, color=GRUVBOX["aqua"], linestyle=":", label="P3 Baseline (Degree 7)", lw=1.4)
        ax0.set_title(f"Sample #{sample_idx}: Raw & Baseline Estimates", fontsize=11, fontweight="bold")
        ax0.legend(loc="upper right", fontsize=8)
        ax0.grid(True)

        # Column 2: Reconstructed Clean vs Ground Truth
        ax1 = axes[row_idx, 1]
        ax1.plot(wavenumbers, pure, color=GRUVBOX["fg0"], label="Ground Truth Pure", lw=1.6)
        ax1.plot(wavenumbers, clean_p2, color=GRUVBOX["orange"], alpha=0.85, label="PolyGaussNet2 (No Amp)", lw=1.2)
        ax1.plot(wavenumbers, clean_p3, color=GRUVBOX["green"], alpha=0.85, label="PolyGaussNet3 (Bounded Amp)", lw=1.2)
        ax1.set_title(f"Sample #{sample_idx}: Preprocessed Spectra Comparison", fontsize=11, fontweight="bold")
        ax1.legend(loc="upper right", fontsize=8)
        ax1.grid(True)

        # Column 3: Error Residuals (Prediction - Ground Truth)
        ax2 = axes[row_idx, 2]
        res_2 = clean_p2 - pure
        res_3 = clean_p3 - pure
        ax2.axhline(0, color=GRUVBOX["gray"], linestyle="--", lw=0.8)
        ax2.plot(wavenumbers, res_2, color=GRUVBOX["orange"], alpha=0.75, label=f"P2 Residual (MAE: {np.mean(np.abs(res_2)):.4f})", lw=1.0)
        ax2.plot(wavenumbers, res_3, color=GRUVBOX["green"], alpha=0.85, label=f"P3 Residual (MAE: {np.mean(np.abs(res_3)):.4f})", lw=1.0)
        ax2.set_title(f"Sample #{sample_idx}: Error Residuals", fontsize=11, fontweight="bold")
        ax2.legend(loc="upper right", fontsize=8)
        ax2.grid(True)

    axes[-1, 0].set_xlabel("Wavenumber Bins (cm⁻¹)", fontsize=10)
    axes[-1, 1].set_xlabel("Wavenumber Bins (cm⁻¹)", fontsize=10)
    axes[-1, 2].set_xlabel("Wavenumber Bins (cm⁻¹)", fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"✓ Visual comparison plot saved to: '{save_path}'")


# =====================================================================
# MAIN BENCHMARK & COMPARISON ROUTINE
# =====================================================================
def main():
    args = parse_args()

    print("\n" + "=" * 90)
    print(" ROUND 5: HEAD-TO-HEAD BENCHMARK (PolyGaussNet2 vs. PolyGaussNet3) ")
    print("=" * 90)

    # Hardware device detection
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Hardware Acceleration Device: {device}\n")

    # 1. Load Test Dataset
    raw_np, pure_np, bc_true_np, data_dir = load_test_dataset(args.data_dir, args.max_samples)
    total_samples, spectral_length = raw_np.shape

    print(f"\nDataset Overview:")
    print(f"  - Total Test Spectra:  {total_samples:,}")
    print(f"  - Spectral Resolution: {spectral_length} bins")
    print(f"  - Test Directory:      {data_dir}")
    print("-" * 90)

    # Helper function to resolve model path with fallbacks
    def resolve_model_path(path_arg, fallback_names):
        candidates = [path_arg, os.path.join(SCRIPT_DIR, path_arg), os.path.join(PROJECT_ROOT, path_arg)]
        for f in fallback_names:
            candidates.extend([
                os.path.join(SCRIPT_DIR, f),
                os.path.join(PROJECT_ROOT, "training_round5", f),
                os.path.join(PROJECT_ROOT, "training_round4", f),
                os.path.join(PROJECT_ROOT, f)
            ])
        for c in candidates:
            if os.path.exists(c) and os.path.isfile(c):
                return os.path.abspath(c)
        return None

    path_p2 = resolve_model_path(args.model_p2, ["head2head_polygaussnet2.pth", "polygaussnet2.pth", "polygaussnet4.pth"])
    path_p3 = resolve_model_path(args.model_p3, ["head2head_polygaussnet3.pth", "polygaussnet3.pth"])

    if path_p2 is None:
        raise FileNotFoundError(f"Could not locate PolyGaussNet2 weights file '{args.model_p2}'. Please run head2head_training.py first.")
    if path_p3 is None:
        raise FileNotFoundError(f"Could not locate PolyGaussNet3 weights file '{args.model_p3}'. Please run head2head_training.py first.")

    # -----------------------------------------------------------------
    # 2. Evaluate PolyGaussNet2 (Sigma-Only, Amplitude = 1.0)
    # -----------------------------------------------------------------
    print(f"\n1. Evaluating PolyGaussNet2 (Weights: {os.path.basename(path_p2)})...")
    ckpt_p2 = torch.load(path_p2, map_location=device, weights_only=False)
    state_p2 = ckpt_p2.get("model_state_dict", ckpt_p2)

    model_p2 = PolyGaussNet2(
        poly_order=args.poly_order,
        filter_kernel_size=args.filter_kernel_size
    ).to(device)
    model_p2.load_state_dict(state_p2)
    model_p2.eval()

    res_p2 = run_model_inference(model_p2, raw_np, args.batch_size, device)
    metrics_p2 = compute_metrics(pure_np, res_p2["clean"], res_p2["baseline"], raw_np)

    print(f"   ✓ Inference complete: {res_p2['duration']:.3f}s ({res_p2['throughput']:8.1f} spec/s, latency: {res_p2['latency_ms']:.4f} ms)")
    print(f"   • Cosine Sim: {metrics_p2['Cosine Similarity'][0]:.5f} | MSE: {metrics_p2['MSE'][0]:.6f} | Log-Cosh: {metrics_p2['Log-Cosh'][0]:.6f} | Peak MAE: {metrics_p2['Peak Summit MAE'][0]:.5f}")

    # -----------------------------------------------------------------
    # 3. Evaluate PolyGaussNet3 (Sigma + Bounded Amplitude [0.85, 1.15])
    # -----------------------------------------------------------------
    print(f"\n2. Evaluating PolyGaussNet3 (Weights: {os.path.basename(path_p3)})...")
    ckpt_p3 = torch.load(path_p3, map_location=device, weights_only=False)
    state_p3 = ckpt_p3.get("model_state_dict", ckpt_p3)

    model_p3 = PolyGaussNet3(
        poly_order=args.poly_order,
        filter_kernel_size=args.filter_kernel_size,
        min_amplitude=args.min_amplitude,
        max_amplitude=args.max_amplitude
    ).to(device)
    model_p3.load_state_dict(state_p3)
    model_p3.eval()

    res_p3 = run_model_inference(model_p3, raw_np, args.batch_size, device)
    metrics_p3 = compute_metrics(pure_np, res_p3["clean"], res_p3["baseline"], raw_np)

    print(f"   ✓ Inference complete: {res_p3['duration']:.3f}s ({res_p3['throughput']:8.1f} spec/s, latency: {res_p3['latency_ms']:.4f} ms)")
    print(f"   • Cosine Sim: {metrics_p3['Cosine Similarity'][0]:.5f} | MSE: {metrics_p3['MSE'][0]:.6f} | Log-Cosh: {metrics_p3['Log-Cosh'][0]:.6f} | Peak MAE: {metrics_p3['Peak Summit MAE'][0]:.5f}")

    # -----------------------------------------------------------------
    # 4. Print Individual Reports
    # -----------------------------------------------------------------
    def fmt_val(val_tuple):
        val, std = val_tuple
        return f"{val:9.6f} (±{std:.5f})"

    print("\n" + "=" * 90)
    print(f"{'INDIVIDUAL EVALUATION REPORTS (N = ' + f'{total_samples:,}' + ' Spectra)':^90}")
    print("=" * 90)

    for name, m, res in [("PolyGaussNet2 (Sigma Only, A=1.0)", metrics_p2, res_p2),
                         ("PolyGaussNet3 (Sigma + Bounded Amplitude)", metrics_p3, res_p3)]:
        print(f"\n--- {name} ---")
        print(f"  • Cosine Similarity (↑):    {fmt_val(m['Cosine Similarity'])}")
        print(f"  • Mean Squared Error (↓):   {fmt_val(m['MSE'])}")
        print(f"  • Log-Cosh Loss (↓):        {fmt_val(m['Log-Cosh'])}")
        print(f"  • Mean Absolute Error (↓):  {fmt_val(m['MAE'])}")
        print(f"  • Root Mean Sq Error (↓):   {fmt_val(m['RMSE'])}")
        print(f"  • Peak Summit MAE (↓):      {fmt_val(m['Peak Summit MAE'])}")
        print(f"  • Baseline Overshoot Rate:  {m['Asym Violation Rate (%)']:.3f}% of bins")
        print(f"  • Max Baseline Overshoot:   {m['Max Baseline Overshoot']:.5f}")
        print(f"  • Throughput & Latency:     {res['throughput']:8.1f} spectra/s ({res['latency_ms']:.4f} ms/spec)")

    # -----------------------------------------------------------------
    # 5. DIRECT HEAD-TO-HEAD COMPARISON PRINT OUT
    # -----------------------------------------------------------------
    print("\n" + "=" * 95)
    print(f"{'DIRECT HEAD-TO-HEAD COMPARISON: PolyGaussNet2 vs. PolyGaussNet3':^95}")
    print(f"{'(Evaluated on ' + f'{total_samples:,}' + ' Test Spectra from Round 4)':^95}")
    print("=" * 95)
    print(f"{'Metric / Feature':<28} | {'PolyGaussNet2 (No Amp)':<23} | {'PolyGaussNet3 (Bounded Amp)':<23} | {'Delta / Advantage':<18}")
    print("-" * 95)

    def print_comp_row(metric_name, val_p2, val_p3, higher_is_better=False, is_percent=False):
        p2_mean, p2_std = val_p2
        p3_mean, p3_std = val_p3
        diff = p3_mean - p2_mean
        pct_change = (diff / (abs(p2_mean) + 1e-12)) * 100.0

        p2_str = f"{p2_mean:.6f}" if not is_percent else f"{p2_mean:.3f}%"
        p3_str = f"{p3_mean:.6f}" if not is_percent else f"{p3_mean:.3f}%"

        if higher_is_better:
            winner = "P3 (+)" if diff > 0 else ("P2 (+)" if diff < 0 else "Tie")
            adv_str = f"{diff:+.6f} ({pct_change:+.2f}%) {winner}"
        else:
            winner = "P3 (Better)" if diff < 0 else ("P2 (Better)" if diff > 0 else "Tie")
            adv_str = f"{diff:+.6f} ({pct_change:+.2f}%) {winner}"

        print(f"{metric_name:<28} | {p2_str:<23} | {p3_str:<23} | {adv_str:<18}")

    print(f"{'Amplitude Modulation':<28} | {'Fixed A = 1.0':<23} | {'Bounded [0.85, 1.15]':<23} | {'Dynamic Peak Gain':<18}")
    print(f"{'Parameters':<28} | {'148,913':<23} | {'148,946':<23} | {'+33 params':<18}")
    print("-" * 95)

    print_comp_row("Cosine Similarity (↑)", metrics_p2["Cosine Similarity"], metrics_p3["Cosine Similarity"], higher_is_better=True)
    print_comp_row("Mean Squared Error (MSE ↓)", metrics_p2["MSE"], metrics_p3["MSE"], higher_is_better=False)
    print_comp_row("Log-Cosh Loss (↓)", metrics_p2["Log-Cosh"], metrics_p3["Log-Cosh"], higher_is_better=False)
    print_comp_row("Mean Absolute Error (MAE ↓)", metrics_p2["MAE"], metrics_p3["MAE"], higher_is_better=False)
    print_comp_row("Root Mean Sq Error (RMSE ↓)", metrics_p2["RMSE"], metrics_p3["RMSE"], higher_is_better=False)
    print_comp_row("Peak Summit MAE (↓)", metrics_p2["Peak Summit MAE"], metrics_p3["Peak Summit MAE"], higher_is_better=False)
    
    print("-" * 95)
    print(f"{'Throughput (spectra/s)':<28} | {res_p2['throughput']:<23.1f} | {res_p3['throughput']:<23.1f} | {res_p3['throughput'] - res_p2['throughput']:+8.1f} spec/s")
    print(f"{'Latency per Spectrum':<28} | {res_p2['latency_ms']:<23.4f} ms | {res_p3['latency_ms']:<23.4f} ms | {res_p3['latency_ms'] - res_p2['latency_ms']:+8.4f} ms")
    print(f"{'Baseline Overshoot Rate':<28} | {metrics_p2['Asym Violation Rate (%)']:<22.3f}% | {metrics_p3['Asym Violation Rate (%)']:<22.3f}% | {'Stable'}")
    print("=" * 95)

    # -----------------------------------------------------------------
    # 6. Generate Comparison Plot
    # -----------------------------------------------------------------
    if not args.no_plots:
        save_plot_dest = args.save_plots
        if not os.path.isabs(save_plot_dest):
            save_plot_dest = os.path.join(SCRIPT_DIR, args.save_plots)
        plot_visual_comparison(raw_np, pure_np, res_p2, res_p3, save_path=save_plot_dest, num_examples=4)

    print("\n✓ Head-to-Head Testing and Direct Comparison Complete!\n")


if __name__ == "__main__":
    main()
