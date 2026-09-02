"""
ASYMMETRIC PENALTY COMPARISON TESTING & BENCHMARKING SCRIPT - ROUND 5
---------------------------------------------------------------------
Evaluates and directly compares:
  1. PolyGaussNet3 trained with Normal Asymmetric Scaling (lambda_asym = 15.0)
  2. PolyGaussNet3 trained with High Asymmetric Scaling   (lambda_asym = 60.0)

Dataset:
  Reaches directly into `training_round4/test_data4` (N = 32,768 test spectra).

Evaluates:
  1. Pearson Correlation Coefficient (ρ ↑ 100% Shift & Scale Invariant)
  2. Zero-Shifted MSE (MSE_zero ↓ Removes constant baseline pedestal offset)
  3. Zero-Shifted MAE (MAE_zero ↓)
  4. Raw Cosine Similarity (↑ Higher is better)
  5. Raw Mean Squared Error (MSE ↓)
  6. Numerically Stable Log-Cosh Loss (↓ Lower is better)
  7. Raw Mean Absolute Error (MAE) & Root Mean Squared Error (RMSE)
  8. Peak Summit Preservation (MAE at top 10% peak locations)
  9. Baseline Overshoot Violation Rate (%) & Max Overshoot
  10. Average Baseline Valley Floor / Pedestal Offset (min(y_pred))
  11. Execution Time, Processing Throughput (spectra/s), and Latency (ms/spectrum)

Outputs:
  - Detailed individual evaluation reports for both asymmetric penalty scales.
  - A direct side-by-side comparison table highlighting deltas and percentage differences.
  - Visual comparison plot saved to `training_round5/asymcomp_comparison.png`.
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

from polygaussnet3 import PolyGaussNet
from training_round5.head2head_testing import load_test_dataset, run_model_inference, GRUVBOX

# Default Paths
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "training_round4", "test_data4")
DEFAULT_MODEL_NORMAL = os.path.join(SCRIPT_DIR, "asymcomp_normal_lambda15.pth")
DEFAULT_MODEL_HIGH = os.path.join(SCRIPT_DIR, "asymcomp_high_lambda60.pth")
DEFAULT_BATCH_SIZE = 512


# =====================================================================
# CLI ARGUMENT PARSER
# =====================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Asymmetric Loss Comparison Testing: Normal (λ=15) vs. High (λ=60) on PolyGaussNet3",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR,
                        help="Directory containing test dataset .npz files")
    parser.add_argument("--model-normal", type=str, default=DEFAULT_MODEL_NORMAL,
                        help="Path to trained Normal Asym (λ=15) weights (.pth)")
    parser.add_argument("--model-high", type=str, default=DEFAULT_MODEL_HIGH,
                        help="Path to trained High Asym (λ=60) weights (.pth)")
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
    parser.add_argument("--save-plots", type=str, default="asymcomp_comparison.png",
                        help="Output path for visual comparison plot")
    return parser.parse_args()


# =====================================================================
# EXTENDED METRICS COMPUTATION (INCLUDES PEARSON & ZERO-SHIFTED MSE)
# =====================================================================
def compute_extended_metrics(y_true_np: np.ndarray, y_pred_np: np.ndarray, baseline_pred_np: np.ndarray, raw_np: np.ndarray):
    """
    Computes a comprehensive suite of numerical and chemometric quality metrics:
      - Pearson Correlation (ρ): Centered, shift & scale invariant shape fidelity
      - Zero-Shifted MSE: MSE after subtracting min(y_pred) to remove any baseline pedestal
      - Zero-Shifted MAE: MAE after min-zeroing
      - Cosine Similarity: Standard uncentered cosine similarity
      - Raw MSE, Log-Cosh, MAE, RMSE
      - Peak Summit MAE: MAE restricted to top 10% highest ground truth peaks
      - Baseline Overshoot Violations: Percentage of bins where baseline > raw
      - Baseline Floor / Pedestal: Mean minimum value of predicted spectra
    """
    N, L = y_true_np.shape

    # 1. Pearson Correlation Coefficient (100% shift and scale invariant)
    # rho = cov(x, y) / (std(x) * std(y))
    y_true_mean = np.mean(y_true_np, axis=1, keepdims=True)
    y_pred_mean = np.mean(y_pred_np, axis=1, keepdims=True)
    true_diff = y_true_np - y_true_mean
    pred_diff = y_pred_np - y_pred_mean
    numerator = np.sum(true_diff * pred_diff, axis=1)
    denominator = np.sqrt(np.sum(true_diff ** 2, axis=1) * np.sum(pred_diff ** 2, axis=1) + 1e-12)
    pearson_per_sample = numerator / denominator

    # 2. Zero-Shifted Spectra (y_pred - min(y_pred))
    pred_min = np.min(y_pred_np, axis=1, keepdims=True)
    y_pred_zero_shifted = y_pred_np - pred_min

    # Zero-Shifted MSE & MAE
    zero_mse_per_sample = np.mean((y_true_np - y_pred_zero_shifted) ** 2, axis=1)
    zero_mae_per_sample = np.mean(np.abs(y_true_np - y_pred_zero_shifted), axis=1)

    # 3. Raw Cosine Similarity
    dot_product = np.sum(y_true_np * y_pred_np, axis=1)
    norm_true = np.linalg.norm(y_true_np, axis=1)
    norm_pred = np.linalg.norm(y_pred_np, axis=1)
    cosine_per_sample = dot_product / (norm_true * norm_pred + 1e-12)

    # 4. Raw MSE & MAE & RMSE
    mse_per_sample = np.mean((y_true_np - y_pred_np) ** 2, axis=1)
    mae_per_sample = np.mean(np.abs(y_true_np - y_pred_np), axis=1)
    rmse_per_sample = np.sqrt(mse_per_sample)

    # 5. Numerically Stable Log-Cosh Loss
    diff = y_pred_np - y_true_np
    logcosh_per_sample = np.mean(
        np.abs(diff) + np.log1p(np.exp(-2.0 * np.abs(diff))) - np.log(2.0),
        axis=1
    )

    # 6. Peak Summit MAE (MAE on top 10% highest ground truth peaks per spectrum)
    peak_mae_list = []
    for i in range(N):
        threshold = np.percentile(y_true_np[i], 90.0)
        peak_mask = y_true_np[i] >= threshold
        if np.any(peak_mask):
            peak_mae_list.append(np.mean(np.abs(diff[i][peak_mask])))
        else:
            peak_mae_list.append(mae_per_sample[i])
    peak_mae_arr = np.array(peak_mae_list)

    # 7. Asymmetric Baseline Violations (Baseline > Raw)
    asym_excess = np.maximum(0.0, baseline_pred_np - raw_np)
    violation_bins = np.sum(asym_excess > 0.0)
    total_bins = N * L
    asym_violation_rate = (violation_bins / total_bins) * 100.0
    max_asym_excess = np.max(asym_excess)

    # 8. Average Minimum Floor (Pedestal offset)
    pedestal_per_sample = pred_min.squeeze(1)

    return {
        "Pearson Correlation (ρ)": (float(np.mean(pearson_per_sample)), float(np.std(pearson_per_sample))),
        "Zero-Shifted MSE": (float(np.mean(zero_mse_per_sample)), float(np.std(zero_mse_per_sample))),
        "Zero-Shifted MAE": (float(np.mean(zero_mae_per_sample)), float(np.std(zero_mae_per_sample))),
        "Cosine Similarity": (float(np.mean(cosine_per_sample)), float(np.std(cosine_per_sample))),
        "Raw MSE": (float(np.mean(mse_per_sample)), float(np.std(mse_per_sample))),
        "Log-Cosh": (float(np.mean(logcosh_per_sample)), float(np.std(logcosh_per_sample))),
        "Raw MAE": (float(np.mean(mae_per_sample)), float(np.std(mae_per_sample))),
        "RMSE": (float(np.mean(rmse_per_sample)), float(np.std(rmse_per_sample))),
        "Peak Summit MAE": (float(np.mean(peak_mae_arr)), float(np.std(peak_mae_arr))),
        "Pedestal Floor Offset": (float(np.mean(pedestal_per_sample)), float(np.std(pedestal_per_sample))),
        "Asym Violation Rate (%)": float(asym_violation_rate),
        "Max Baseline Overshoot": float(max_asym_excess)
    }


# =====================================================================
# VISUALIZATION / PLOTTING
# =====================================================================
def plot_asym_comparison(raw_np, pure_np, res_normal, res_high, save_path="asymcomp_comparison.png", num_examples=4):
    """Generates a clean visual comparison plot in Gruvbox dark style."""
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
        clean_norm = res_normal["clean"][sample_idx]
        base_norm = res_normal["baseline"][sample_idx]
        clean_high = res_high["clean"][sample_idx]
        base_high = res_high["baseline"][sample_idx]

        # Column 1: Raw & Baselines
        ax0 = axes[row_idx, 0]
        ax0.plot(wavenumbers, raw, color=GRUVBOX["fg3"], alpha=0.5, label="Raw Input", lw=1.0)
        ax0.plot(wavenumbers, base_norm, color=GRUVBOX["yellow"], linestyle="--", label="Normal Baseline (λ=15)", lw=1.4)
        ax0.plot(wavenumbers, base_high, color=GRUVBOX["aqua"], linestyle=":", label="High Baseline (λ=60)", lw=1.4)
        ax0.set_title(f"Sample #{sample_idx}: Raw & Baseline Estimates", fontsize=11, fontweight="bold")
        ax0.legend(loc="upper right", fontsize=8)
        ax0.grid(True)

        # Column 2: Reconstructed Clean vs Ground Truth
        ax1 = axes[row_idx, 1]
        ax1.plot(wavenumbers, pure, color=GRUVBOX["fg0"], label="Ground Truth Pure", lw=1.6)
        ax1.plot(wavenumbers, clean_norm, color=GRUVBOX["orange"], alpha=0.85, label="Normal Penalty (λ=15)", lw=1.2)
        ax1.plot(wavenumbers, clean_high, color=GRUVBOX["green"], alpha=0.85, label="High Penalty (λ=60)", lw=1.2)
        ax1.set_title(f"Sample #{sample_idx}: Preprocessed Spectra Comparison", fontsize=11, fontweight="bold")
        ax1.legend(loc="upper right", fontsize=8)
        ax1.grid(True)

        # Column 3: Error Residuals (Prediction - Ground Truth)
        ax2 = axes[row_idx, 2]
        res_n = clean_norm - pure
        res_h = clean_high - pure
        ax2.axhline(0, color=GRUVBOX["gray"], linestyle="--", lw=0.8)
        ax2.plot(wavenumbers, res_n, color=GRUVBOX["orange"], alpha=0.75, label=f"Normal Residual (MAE: {np.mean(np.abs(res_n)):.4f})", lw=1.0)
        ax2.plot(wavenumbers, res_h, color=GRUVBOX["green"], alpha=0.85, label=f"High Residual (MAE: {np.mean(np.abs(res_h)):.4f})", lw=1.0)
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

    print("\n" + "=" * 94)
    print(" ROUND 5: ASYMMETRIC LOSS SCALE HEAD-TO-HEAD BENCHMARK (PolyGaussNet3) ")
    print(" Normal Asymmetric Penalty (λ=15.0) vs. High Asymmetric Penalty (λ=60.0) ")
    print("=" * 94)

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
    print("-" * 94)

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

    path_normal = resolve_model_path(args.model_normal, ["asymcomp_normal_lambda15.pth", "losscomp_dual_supervised.pth", "head2head_polygaussnet3.pth", "polygaussnet3.pth"])
    path_high = resolve_model_path(args.model_high, ["asymcomp_high_lambda60.pth"])

    if path_normal is None:
        raise FileNotFoundError(f"Could not locate Normal Asym model weights '{args.model_normal}'. Please run asymcomparison_training.py first.")
    if path_high is None:
        raise FileNotFoundError(f"Could not locate High Asym model weights '{args.model_high}'. Please run asymcomparison_training.py first.")

    # -----------------------------------------------------------------
    # 2. Evaluate Normal Asymmetric Penalty Model (λ=15.0)
    # -----------------------------------------------------------------
    print(f"\n1. Evaluating Normal Asymmetric Model (λ=15.0, Weights: {os.path.basename(path_normal)})...")
    ckpt_norm = torch.load(path_normal, map_location=device, weights_only=False)
    state_norm = ckpt_norm.get("model_state_dict", ckpt_norm)

    model_norm = PolyGaussNet(
        poly_order=args.poly_order,
        filter_kernel_size=args.filter_kernel_size,
        min_amplitude=args.min_amplitude,
        max_amplitude=args.max_amplitude
    ).to(device)
    model_norm.load_state_dict(state_norm)
    model_norm.eval()

    res_norm = run_model_inference(model_norm, raw_np, args.batch_size, device)
    metrics_norm = compute_extended_metrics(pure_np, res_norm["clean"], res_norm["baseline"], raw_np)

    print(f"   ✓ Inference complete: {res_norm['duration']:.3f}s ({res_norm['throughput']:8.1f} spec/s, latency: {res_norm['latency_ms']:.4f} ms)")
    print(f"   • Pearson ρ: {metrics_norm['Pearson Correlation (ρ)'][0]:.5f} | Zero-MSE: {metrics_norm['Zero-Shifted MSE'][0]:.6f} | Raw MSE: {metrics_norm['Raw MSE'][0]:.6f} | Peak MAE: {metrics_norm['Peak Summit MAE'][0]:.5f}")

    # -----------------------------------------------------------------
    # 3. Evaluate High Asymmetric Penalty Model (λ=60.0)
    # -----------------------------------------------------------------
    print(f"\n2. Evaluating High Asymmetric Model (λ=60.0, Weights: {os.path.basename(path_high)})...")
    ckpt_high = torch.load(path_high, map_location=device, weights_only=False)
    state_high = ckpt_high.get("model_state_dict", ckpt_high)

    model_high = PolyGaussNet(
        poly_order=args.poly_order,
        filter_kernel_size=args.filter_kernel_size,
        min_amplitude=args.min_amplitude,
        max_amplitude=args.max_amplitude
    ).to(device)
    model_high.load_state_dict(state_high)
    model_high.eval()

    res_high = run_model_inference(model_high, raw_np, args.batch_size, device)
    metrics_high = compute_extended_metrics(pure_np, res_high["clean"], res_high["baseline"], raw_np)

    print(f"   ✓ Inference complete: {res_high['duration']:.3f}s ({res_high['throughput']:8.1f} spec/s, latency: {res_high['latency_ms']:.4f} ms)")
    print(f"   • Pearson ρ: {metrics_high['Pearson Correlation (ρ)'][0]:.5f} | Zero-MSE: {metrics_high['Zero-Shifted MSE'][0]:.6f} | Raw MSE: {metrics_high['Raw MSE'][0]:.6f} | Peak MAE: {metrics_high['Peak Summit MAE'][0]:.5f}")

    # -----------------------------------------------------------------
    # 4. Print Individual Reports
    # -----------------------------------------------------------------
    def fmt_val(val_tuple):
        val, std = val_tuple
        return f"{val:9.6f} (±{std:.5f})"

    print("\n" + "=" * 94)
    print(f"{'INDIVIDUAL EVALUATION REPORTS (N = ' + f'{total_samples:,}' + ' Spectra)':^94}")
    print("=" * 94)

    for name, m, res in [("Normal Asymmetric Scaling: Dual Supervised + 15.0 * Asym", metrics_norm, res_norm),
                         ("High Asymmetric Scaling: Dual Supervised + 60.0 * Asym (4x)", metrics_high, res_high)]:
        print(f"\n--- {name} ---")
        print(f"  • Pearson Correlation (ρ ↑):  {fmt_val(m['Pearson Correlation (ρ)'])}")
        print(f"  • Zero-Shifted MSE (↓):       {fmt_val(m['Zero-Shifted MSE'])}")
        print(f"  • Zero-Shifted MAE (↓):       {fmt_val(m['Zero-Shifted MAE'])}")
        print(f"  • Raw Cosine Similarity (↑):  {fmt_val(m['Cosine Similarity'])}")
        print(f"  • Raw Mean Squared Error (↓): {fmt_val(m['Raw MSE'])}")
        print(f"  • Log-Cosh Loss (↓):          {fmt_val(m['Log-Cosh'])}")
        print(f"  • Raw Mean Absolute Error (↓):{fmt_val(m['Raw MAE'])}")
        print(f"  • Root Mean Sq Error (↓):     {fmt_val(m['RMSE'])}")
        print(f"  • Peak Summit MAE (↓):        {fmt_val(m['Peak Summit MAE'])}")
        print(f"  • Pedestal Floor Offset:      {fmt_val(m['Pedestal Floor Offset'])}")
        print(f"  • Baseline Overshoot Rate:    {m['Asym Violation Rate (%)']:.3f}% of bins")
        print(f"  • Max Baseline Overshoot:     {m['Max Baseline Overshoot']:.5f}")
        print(f"  • Throughput & Latency:       {res['throughput']:8.1f} spectra/s ({res['latency_ms']:.4f} ms/spec)")

    # -----------------------------------------------------------------
    # 5. DIRECT HEAD-TO-HEAD COMPARISON PRINT OUT
    # -----------------------------------------------------------------
    print("\n" + "=" * 98)
    print(f"{'DIRECT ASYMMETRIC PENALTY COMPARISON: Normal (λ=15) vs. High (λ=60)':^98}")
    print(f"{'(Both on PolyGaussNet3 evaluated across ' + f'{total_samples:,}' + ' Test Spectra)':^98}")
    print("=" * 98)
    print(f"{'Metric / Feature':<30} | {'Normal (λ=15.0)':<21} | {'High (λ=60.0)':<21} | {'Delta / Advantage':<19}")
    print("-" * 98)

    def print_comp_row(metric_name, val_norm, val_high, higher_is_better=False, is_percent=False):
        n_mean, n_std = val_norm
        h_mean, h_std = val_high
        diff = h_mean - n_mean
        pct_change = (diff / (abs(n_mean) + 1e-12)) * 100.0

        n_str = f"{n_mean:.6f}" if not is_percent else f"{n_mean:.3f}%"
        h_str = f"{h_mean:.6f}" if not is_percent else f"{h_mean:.3f}%"

        if higher_is_better:
            winner = "High (+)" if diff > 0 else ("Normal (+)" if diff < 0 else "Tie")
            adv_str = f"{diff:+.6f} ({pct_change:+.2f}%) {winner}"
        else:
            winner = "High (Better)" if diff < 0 else ("Normal (Better)" if diff > 0 else "Tie")
            adv_str = f"{diff:+.6f} ({pct_change:+.2f}%) {winner}"

        print(f"{metric_name:<30} | {n_str:<21} | {h_str:<21} | {adv_str:<19}")

    print(f"{'Asymmetric Scaling (λ)':<30} | {'15.0':<21} | {'60.0 (4x)':<21} | {'4x Stronger':<19}")
    print(f"{'Loss Formulation':<30} | {'Dual Supervised':<21} | {'Dual Supervised':<21} | {'Identical':<19}")
    print(f"{'Model Architecture':<30} | {'PolyGaussNet3':<21} | {'PolyGaussNet3':<21} | {'Identical (148.9k)':<19}")
    print("-" * 98)

    # Invariant & Primary Metrics
    print_comp_row("Pearson Correlation (ρ ↑)", metrics_norm["Pearson Correlation (ρ)"], metrics_high["Pearson Correlation (ρ)"], higher_is_better=True)
    print_comp_row("Zero-Shifted MSE (↓)", metrics_norm["Zero-Shifted MSE"], metrics_high["Zero-Shifted MSE"], higher_is_better=False)
    print_comp_row("Zero-Shifted MAE (↓)", metrics_norm["Zero-Shifted MAE"], metrics_high["Zero-Shifted MAE"], higher_is_better=False)
    print_comp_row("Peak Summit MAE (↓)", metrics_norm["Peak Summit MAE"], metrics_high["Peak Summit MAE"], higher_is_better=False)
    
    print("-" * 98)
    # Raw Global Metrics
    print_comp_row("Raw Cosine Similarity (↑)", metrics_norm["Cosine Similarity"], metrics_high["Cosine Similarity"], higher_is_better=True)
    print_comp_row("Raw MSE (↓)", metrics_norm["Raw MSE"], metrics_high["Raw MSE"], higher_is_better=False)
    print_comp_row("Log-Cosh Loss (↓)", metrics_norm["Log-Cosh"], metrics_high["Log-Cosh"], higher_is_better=False)
    print_comp_row("Raw MAE (↓)", metrics_norm["Raw MAE"], metrics_high["Raw MAE"], higher_is_better=False)
    print_comp_row("Pedestal Floor Offset (min)", metrics_norm["Pedestal Floor Offset"], metrics_high["Pedestal Floor Offset"], higher_is_better=False)

    print("-" * 98)
    # Overshoot & Operational
    print(f"{'Baseline Overshoot Rate':<30} | {metrics_norm['Asym Violation Rate (%)']:<20.3f}% | {metrics_high['Asym Violation Rate (%)']:<20.3f}% | {'Direct Comp'}")
    print(f"{'Max Baseline Overshoot':<30} | {metrics_norm['Max Baseline Overshoot']:<21.5f} | {metrics_high['Max Baseline Overshoot']:<21.5f} | {metrics_high['Max Baseline Overshoot'] - metrics_norm['Max Baseline Overshoot']:+.5f}")
    print(f"{'Throughput (spectra/s)':<30} | {res_norm['throughput']:<21.1f} | {res_high['throughput']:<21.1f} | {res_high['throughput'] - res_norm['throughput']:+8.1f} spec/s")
    print(f"{'Latency per Spectrum':<30} | {res_norm['latency_ms']:<21.4f} ms | {res_high['latency_ms']:<21.4f} ms | {res_high['latency_ms'] - res_norm['latency_ms']:+8.4f} ms")
    print("=" * 98)

    # -----------------------------------------------------------------
    # 6. Generate Comparison Plot
    # -----------------------------------------------------------------
    if not args.no_plots:
        save_plot_dest = args.save_plots
        if not os.path.isabs(save_plot_dest):
            save_plot_dest = os.path.join(SCRIPT_DIR, args.save_plots)
        plot_asym_comparison(raw_np, pure_np, res_norm, res_high, save_path=save_plot_dest, num_examples=4)

    print("\n✓ Asymmetric Loss Scale Testing & Benchmark Complete!\n")


if __name__ == "__main__":
    main()
