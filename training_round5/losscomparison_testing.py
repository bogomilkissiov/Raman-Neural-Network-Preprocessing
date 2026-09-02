"""
LOSS COMPARISON TESTING & BENCHMARKING SCRIPT - ROUND 5
-------------------------------------------------------
Evaluates and directly compares:
  1. PolyGaussNet3 trained with Dual Supervised + Asymmetric Loss
       (L_total = LogCosh(BC) + LogCosh(Clean) + 15 * Asym)
  2. PolyGaussNet3 trained with Single End-to-End + Asymmetric Loss
       (L_total = LogCosh(Clean) + 15 * Asym)

Dataset:
  Reaches directly into `training_round4/test_data4` (N = 32,768 test spectra).

Evaluates:
  1. Cosine Similarity vs. Ground Truth Pure Spectra (↑ Higher is better)
  2. Mean Squared Error (MSE) vs. Ground Truth Pure Spectra (↓ Lower is better)
  3. Numerically Stable Log-Cosh Loss (↓ Lower is better)
  4. Mean Absolute Error (MAE) & Root Mean Squared Error (RMSE)
  5. Peak Summit Preservation (MAE at top 10% peak locations)
  6. Baseline Overshoot Violation Rate (%) & Max Overshoot
  7. Execution Time, Processing Throughput (spectra/s), and Latency (ms/spectrum)

Outputs:
  - Detailed individual evaluation reports for both loss regimes.
  - A direct side-by-side comparison table highlighting deltas and percentage differences.
  - Visual comparison plot saved to `training_round5/losscomp_comparison.png`.
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
from training_round5.head2head_testing import load_test_dataset, run_model_inference, compute_metrics, GRUVBOX

# Default Paths
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "training_round4", "test_data4")
DEFAULT_MODEL_DUAL = os.path.join(SCRIPT_DIR, "losscomp_dual_supervised.pth")
DEFAULT_MODEL_E2E = os.path.join(SCRIPT_DIR, "losscomp_end_to_end.pth")
DEFAULT_BATCH_SIZE = 512


# =====================================================================
# CLI ARGUMENT PARSER
# =====================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Loss Comparison Testing & Direct Benchmark: Dual Supervised vs. Single End-to-End (PolyGaussNet3)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR,
                        help="Directory containing test dataset .npz files")
    parser.add_argument("--model-dual", type=str, default=DEFAULT_MODEL_DUAL,
                        help="Path to trained Dual Supervised weights (.pth)")
    parser.add_argument("--model-e2e", type=str, default=DEFAULT_MODEL_E2E,
                        help="Path to trained End-to-End weights (.pth)")
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
    parser.add_argument("--save-plots", type=str, default="losscomp_comparison.png",
                        help="Output path for visual comparison plot")
    return parser.parse_args()


# =====================================================================
# VISUALIZATION / PLOTTING
# =====================================================================
def plot_loss_comparison(raw_np, pure_np, res_dual, res_e2e, save_path="losscomp_comparison.png", num_examples=4):
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
        clean_dual = res_dual["clean"][sample_idx]
        base_dual = res_dual["baseline"][sample_idx]
        clean_e2e = res_e2e["clean"][sample_idx]
        base_e2e = res_e2e["baseline"][sample_idx]

        # Column 1: Raw & Baselines
        ax0 = axes[row_idx, 0]
        ax0.plot(wavenumbers, raw, color=GRUVBOX["fg3"], alpha=0.5, label="Raw Input", lw=1.0)
        ax0.plot(wavenumbers, base_dual, color=GRUVBOX["yellow"], linestyle="--", label="Dual Supervised Baseline", lw=1.4)
        ax0.plot(wavenumbers, base_e2e, color=GRUVBOX["aqua"], linestyle=":", label="End-to-End Baseline", lw=1.4)
        ax0.set_title(f"Sample #{sample_idx}: Raw & Baseline Estimates", fontsize=11, fontweight="bold")
        ax0.legend(loc="upper right", fontsize=8)
        ax0.grid(True)

        # Column 2: Reconstructed Clean vs Ground Truth
        ax1 = axes[row_idx, 1]
        ax1.plot(wavenumbers, pure, color=GRUVBOX["fg0"], label="Ground Truth Pure", lw=1.6)
        ax1.plot(wavenumbers, clean_dual, color=GRUVBOX["orange"], alpha=0.85, label="Dual Supervised (BC+Clean)", lw=1.2)
        ax1.plot(wavenumbers, clean_e2e, color=GRUVBOX["green"], alpha=0.85, label="Single End-to-End (Clean)", lw=1.2)
        ax1.set_title(f"Sample #{sample_idx}: Preprocessed Spectra Comparison", fontsize=11, fontweight="bold")
        ax1.legend(loc="upper right", fontsize=8)
        ax1.grid(True)

        # Column 3: Error Residuals (Prediction - Ground Truth)
        ax2 = axes[row_idx, 2]
        res_d = clean_dual - pure
        res_e = clean_e2e - pure
        ax2.axhline(0, color=GRUVBOX["gray"], linestyle="--", lw=0.8)
        ax2.plot(wavenumbers, res_d, color=GRUVBOX["orange"], alpha=0.75, label=f"Dual Residual (MAE: {np.mean(np.abs(res_d)):.4f})", lw=1.0)
        ax2.plot(wavenumbers, res_e, color=GRUVBOX["green"], alpha=0.85, label=f"End-to-End Residual (MAE: {np.mean(np.abs(res_e)):.4f})", lw=1.0)
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

    print("\n" + "=" * 92)
    print(" ROUND 5: LOSS FUNCTION HEAD-TO-HEAD BENCHMARK (PolyGaussNet3) ")
    print(" Dual Supervised (LogCosh BC + LogCosh Clean) vs. Single End-to-End (LogCosh Clean) ")
    print("=" * 92)

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
    print("-" * 92)

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

    path_dual = resolve_model_path(args.model_dual, ["losscomp_dual_supervised.pth", "head2head_polygaussnet3.pth", "polygaussnet3.pth"])
    path_e2e = resolve_model_path(args.model_e2e, ["losscomp_end_to_end.pth"])

    if path_dual is None:
        raise FileNotFoundError(f"Could not locate Dual Supervised model weights '{args.model_dual}'. Please run losscomparison_training.py first.")
    if path_e2e is None:
        raise FileNotFoundError(f"Could not locate End-to-End model weights '{args.model_e2e}'. Please run losscomparison_training.py first.")

    # -----------------------------------------------------------------
    # 2. Evaluate Dual Supervised PolyGaussNet3
    # -----------------------------------------------------------------
    print(f"\n1. Evaluating Dual Supervised Model (Weights: {os.path.basename(path_dual)})...")
    ckpt_dual = torch.load(path_dual, map_location=device, weights_only=False)
    state_dual = ckpt_dual.get("model_state_dict", ckpt_dual)

    model_dual = PolyGaussNet(
        poly_order=args.poly_order,
        filter_kernel_size=args.filter_kernel_size,
        min_amplitude=args.min_amplitude,
        max_amplitude=args.max_amplitude
    ).to(device)
    model_dual.load_state_dict(state_dual)
    model_dual.eval()

    res_dual = run_model_inference(model_dual, raw_np, args.batch_size, device)
    metrics_dual = compute_metrics(pure_np, res_dual["clean"], res_dual["baseline"], raw_np)

    print(f"   ✓ Inference complete: {res_dual['duration']:.3f}s ({res_dual['throughput']:8.1f} spec/s, latency: {res_dual['latency_ms']:.4f} ms)")
    print(f"   • Cosine Sim: {metrics_dual['Cosine Similarity'][0]:.5f} | MSE: {metrics_dual['MSE'][0]:.6f} | Log-Cosh: {metrics_dual['Log-Cosh'][0]:.6f} | Peak MAE: {metrics_dual['Peak Summit MAE'][0]:.5f}")

    # -----------------------------------------------------------------
    # 3. Evaluate Single End-to-End PolyGaussNet3
    # -----------------------------------------------------------------
    print(f"\n2. Evaluating Single End-to-End Model (Weights: {os.path.basename(path_e2e)})...")
    ckpt_e2e = torch.load(path_e2e, map_location=device, weights_only=False)
    state_e2e = ckpt_e2e.get("model_state_dict", ckpt_e2e)

    model_e2e = PolyGaussNet(
        poly_order=args.poly_order,
        filter_kernel_size=args.filter_kernel_size,
        min_amplitude=args.min_amplitude,
        max_amplitude=args.max_amplitude
    ).to(device)
    model_e2e.load_state_dict(state_e2e)
    model_e2e.eval()

    res_e2e = run_model_inference(model_e2e, raw_np, args.batch_size, device)
    metrics_e2e = compute_metrics(pure_np, res_e2e["clean"], res_e2e["baseline"], raw_np)

    print(f"   ✓ Inference complete: {res_e2e['duration']:.3f}s ({res_e2e['throughput']:8.1f} spec/s, latency: {res_e2e['latency_ms']:.4f} ms)")
    print(f"   • Cosine Sim: {metrics_e2e['Cosine Similarity'][0]:.5f} | MSE: {metrics_e2e['MSE'][0]:.6f} | Log-Cosh: {metrics_e2e['Log-Cosh'][0]:.6f} | Peak MAE: {metrics_e2e['Peak Summit MAE'][0]:.5f}")

    # -----------------------------------------------------------------
    # 4. Print Individual Reports
    # -----------------------------------------------------------------
    def fmt_val(val_tuple):
        val, std = val_tuple
        return f"{val:9.6f} (±{std:.5f})"

    print("\n" + "=" * 92)
    print(f"{'INDIVIDUAL EVALUATION REPORTS (N = ' + f'{total_samples:,}' + ' Spectra)':^92}")
    print("=" * 92)

    for name, m, res in [("Dual Supervised Loss: LogCosh(BC) + LogCosh(Clean) + Asym", metrics_dual, res_dual),
                         ("Single End-to-End Loss: LogCosh(Clean) + Asym", metrics_e2e, res_e2e)]:
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
    print("\n" + "=" * 96)
    print(f"{'DIRECT LOSS FUNCTION COMPARISON: Dual Supervised vs. Single End-to-End':^96}")
    print(f"{'(Both on PolyGaussNet3 evaluated across ' + f'{total_samples:,}' + ' Test Spectra)':^96}")
    print("=" * 96)
    print(f"{'Metric / Feature':<28} | {'Dual Supervised':<22} | {'Single End-to-End':<22} | {'Delta / Advantage':<19}")
    print("-" * 96)

    def print_comp_row(metric_name, val_dual, val_e2e, higher_is_better=False, is_percent=False):
        d_mean, d_std = val_dual
        e_mean, e_std = val_e2e
        diff = e_mean - d_mean
        pct_change = (diff / (abs(d_mean) + 1e-12)) * 100.0

        d_str = f"{d_mean:.6f}" if not is_percent else f"{d_mean:.3f}%"
        e_str = f"{e_mean:.6f}" if not is_percent else f"{e_mean:.3f}%"

        if higher_is_better:
            winner = "E2E (+)" if diff > 0 else ("Dual (+)" if diff < 0 else "Tie")
            adv_str = f"{diff:+.6f} ({pct_change:+.2f}%) {winner}"
        else:
            winner = "E2E (Better)" if diff < 0 else ("Dual (Better)" if diff > 0 else "Tie")
            adv_str = f"{diff:+.6f} ({pct_change:+.2f}%) {winner}"

        print(f"{metric_name:<28} | {d_str:<22} | {e_str:<22} | {adv_str:<19}")

    print(f"{'Loss Formulation':<28} | {'LogCosh(BC+Clean)+Asym':<22} | {'LogCosh(Clean)+Asym':<22} | {'End-to-End':<19}")
    print(f"{'Model Architecture':<28} | {'PolyGaussNet3':<22} | {'PolyGaussNet3':<22} | {'Identical (148.9k)':<19}")
    print(f"{'Asymmetric Penalty (λ)':<28} | {'15.0':<22} | {'15.0':<22} | {'Identical':<19}")
    print("-" * 96)

    print_comp_row("Cosine Similarity (↑)", metrics_dual["Cosine Similarity"], metrics_e2e["Cosine Similarity"], higher_is_better=True)
    print_comp_row("Mean Squared Error (MSE ↓)", metrics_dual["MSE"], metrics_e2e["MSE"], higher_is_better=False)
    print_comp_row("Log-Cosh Loss (↓)", metrics_dual["Log-Cosh"], metrics_e2e["Log-Cosh"], higher_is_better=False)
    print_comp_row("Mean Absolute Error (MAE ↓)", metrics_dual["MAE"], metrics_e2e["MAE"], higher_is_better=False)
    print_comp_row("Root Mean Sq Error (RMSE ↓)", metrics_dual["RMSE"], metrics_e2e["RMSE"], higher_is_better=False)
    print_comp_row("Peak Summit MAE (↓)", metrics_dual["Peak Summit MAE"], metrics_e2e["Peak Summit MAE"], higher_is_better=False)
    
    print("-" * 96)
    print(f"{'Throughput (spectra/s)':<28} | {res_dual['throughput']:<22.1f} | {res_e2e['throughput']:<22.1f} | {res_e2e['throughput'] - res_dual['throughput']:+8.1f} spec/s")
    print(f"{'Latency per Spectrum':<28} | {res_dual['latency_ms']:<22.4f} ms | {res_e2e['latency_ms']:<22.4f} ms | {res_e2e['latency_ms'] - res_dual['latency_ms']:+8.4f} ms")
    print(f"{'Baseline Overshoot Rate':<28} | {metrics_dual['Asym Violation Rate (%)']:<21.3f}% | {metrics_e2e['Asym Violation Rate (%)']:<21.3f}% | {'Direct Comp'}")
    print("=" * 96)

    # -----------------------------------------------------------------
    # 6. Generate Comparison Plot
    # -----------------------------------------------------------------
    if not args.no_plots:
        save_plot_dest = args.save_plots
        if not os.path.isabs(save_plot_dest):
            save_plot_dest = os.path.join(SCRIPT_DIR, args.save_plots)
        plot_loss_comparison(raw_np, pure_np, res_dual, res_e2e, save_path=save_plot_dest, num_examples=4)

    print("\n✓ Loss Comparison Testing & Direct Benchmark Complete!\n")


if __name__ == "__main__":
    main()
