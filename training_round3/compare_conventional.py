"""
COMPARISON SCRIPT: PolyGaussNet vs. Conventional Preprocessing Pipeline
------------------------------------------------------------------------
Evaluates both methods on the exact same 1,024 spectra from dataset_part_1.npz:
  Method 1: Conventional Pipeline (Despike -> BayesShrink Wavelet -> Baseline Removal)
  Method 2: PolyGaussNet (15th Order Polynomial Baseline + Adaptive GenGaussian Filter)

Metrics:
  - Total Log-Cosh Loss
  - Mean Absolute Error (MAE) vs. Ground Truth Pure Spectra
  - Root Mean Squared Error (RMSE)
  - Processing Throughput (spectra/second)

Generates:
  - 'conventional_vs_polygaussnet_comparison.png' visual overlay plot
"""

import os
import sys
import time
import numpy as np
import torch
import matplotlib.pyplot as plt

# Configure paths so imports resolve whether running from project root or training_round3
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
CONVENTIONAL_DIR = os.path.join(PROJECT_ROOT, "test_files", "conventional")

for p in [PROJECT_ROOT, SCRIPT_DIR, CONVENTIONAL_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

import pre
from spectra_class import spectra
from polygaussnet import PolyGaussNet


def log_cosh_loss_np(y_pred, y_true):
    """Numerically stable Log-Cosh loss in NumPy."""
    x = y_pred - y_true
    # log(cosh(x)) = |x| + log1p(exp(-2|x|)) - log(2)
    return np.mean(np.abs(x) + np.log1p(np.exp(-2.0 * np.abs(x))) - np.log(2.0))


def run_comparison():
    print("=" * 78)
    print(" Benchmarking: Conventional Preprocessing vs. PolyGaussNet ")
    print("=" * 78)

    # 1. Load exact 1,024 test dataset
    data_file = os.path.join(SCRIPT_DIR, "training_spectra3", "dataset_part_1.npz")
    if not os.path.exists(data_file):
        data_file = os.path.join(PROJECT_ROOT, "training_round3", "training_spectra3", "dataset_part_1.npz")
    if not os.path.exists(data_file):
        data_file = "training_spectra3/dataset_part_1.npz"
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Cannot find dataset file 'training_spectra3/dataset_part_1.npz'")

    data = np.load(data_file)
    n_samples = 1024
    raw_np = data["full_matrix"][:n_samples].astype(np.float32)
    bc_true_np = data["pure_noise_cosmic_matrix"][:n_samples].astype(np.float32)
    clean_true_np = data["pure_matrix"][:n_samples].astype(np.float32)
    data.close()

    L = raw_np.shape[1]
    wavenumbers_np = np.tile(np.arange(L, dtype=np.float32), (n_samples, 1))
    print(f"Loaded {n_samples:,} spectra (Spectral length: {L} bins)\n")

    # -----------------------------------------------------------------
    # 2. RUN CONVENTIONAL PIPELINE
    # -----------------------------------------------------------------
    print("1. Running Conventional Preprocessing Pipeline (pre.py)...")
    t0 = time.time()
    
    # Create spectra object using factory method
    conv_data = spectra.from_matrices(wavenumbers_np, np.copy(raw_np))
    
    # Run full conventional pipeline: Despike -> Denoise -> Baseline Removal -> Normalization -> Shift
    pre.preprocess_pipeline(conv_data, normalize=False, shift=False)
    
    conv_clean_np = conv_data.intensity_matrix
    conv_duration = time.time() - t0
    conv_throughput = n_samples / conv_duration if conv_duration > 0 else 0

    conv_logcosh = log_cosh_loss_np(conv_clean_np, clean_true_np)
    conv_mae = np.mean(np.abs(conv_clean_np - clean_true_np))
    conv_rmse = np.sqrt(np.mean((conv_clean_np - clean_true_np) ** 2))

    print(f"   Done in {conv_duration:.2f}s ({conv_throughput:.1f} spectra/s)")
    print(f"   Log-Cosh Loss: {conv_logcosh:.6f} | MAE: {conv_mae:.6f} | RMSE: {conv_rmse:.6f}\n")

    # -----------------------------------------------------------------
    # 3. RUN POLYGAUSSNET
    # -----------------------------------------------------------------
    print("2. Running PolyGaussNet (test_polygaussnet3.pth)...")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = PolyGaussNet().to(device)
    
    # Check for weights file locally or in training_round3
    possible_model_paths = [
        os.path.join(SCRIPT_DIR, "test_polygaussnet3.pth"),
        os.path.join(SCRIPT_DIR, "polygaussnet3.pth"),
        os.path.join(PROJECT_ROOT, "training_round3", "test_polygaussnet3.pth"),
        os.path.join(PROJECT_ROOT, "training_round3", "polygaussnet3.pth"),
        "test_polygaussnet3.pth",
        "polygaussnet3.pth"
    ]
    model_path = next((p for p in possible_model_paths if os.path.exists(p)), None)
    if model_path is None:
        raise FileNotFoundError("Could not find test_polygaussnet3.pth or polygaussnet3.pth")
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    raw_torch = torch.tensor(raw_np, dtype=torch.float32, device=device)

    t0 = time.time()
    with torch.no_grad():
        poly_clean_torch, poly_base_torch, poly_bc_torch, _ = model(raw_torch)
        if device.type == "mps":
            torch.mps.synchronize()
    poly_duration = time.time() - t0
    poly_throughput = n_samples / poly_duration if poly_duration > 0 else 0

    poly_clean_np = poly_clean_torch.cpu().numpy()
    poly_base_np = poly_base_torch.cpu().numpy()

    poly_logcosh = log_cosh_loss_np(poly_clean_np, clean_true_np)
    poly_mae = np.mean(np.abs(poly_clean_np - clean_true_np))
    poly_rmse = np.sqrt(np.mean((poly_clean_np - clean_true_np) ** 2))

    print(f"   Done in {poly_duration:.3f}s ({poly_throughput:.1f} spectra/s)")
    print(f"   Log-Cosh Loss: {poly_logcosh:.6f} | MAE: {poly_mae:.6f} | RMSE: {poly_rmse:.6f}\n")

    # -----------------------------------------------------------------
    # 4. HEAD-TO-HEAD COMPARISON TABLE
    # -----------------------------------------------------------------
    print("=" * 78)
    print(f"{'Metric':<30s} | {'Conventional (pre.py)':<20s} | {'PolyGaussNet':<20s}")
    print("-" * 78)
    print(f"{'Log-Cosh Loss (vs Pure)':<30s} | {conv_logcosh:<20.6f} | {poly_logcosh:<20.6f}")
    print(f"{'Mean Absolute Error (MAE)':<30s} | {conv_mae:<20.6f} | {poly_mae:<20.6f}")
    print(f"{'Root Mean Squared Error (RMSE)':<30s} | {conv_rmse:<20.6f} | {poly_rmse:<20.6f}")
    print(f"{'Processing Speed':<30s} | {f'{conv_throughput:.1f} spec/s':<20s} | {f'{poly_throughput:.1f} spec/s':<20s}")
    speedup = poly_throughput / conv_throughput if conv_throughput > 0 else 1
    print(f"{'Speedup Factor':<30s} | {'1.0x (Baseline)':<20s} | {f'{speedup:.1f}x Faster':<20s}")
    print("=" * 78)

    # -----------------------------------------------------------------
    # 5. VISUAL PLOTTING COMPARISON
    # -----------------------------------------------------------------
    print("\nGenerating visual comparison plot...")
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    for i in range(3):
        ax = axes[i]
        ax.plot(raw_np[i], label='Raw Noisy Input', color='lightgray', linewidth=1.2)
        ax.plot(conv_clean_np[i], label='Conventional Pipeline (Wavelet+Baseline)', color='blue', alpha=0.8, linewidth=1.2)
        ax.plot(poly_clean_np[i], label='PolyGaussNet (Neural Filter)', color='red', alpha=0.9, linewidth=1.4)
        ax.plot(clean_true_np[i], label='Ground Truth Pure', color='black', linestyle='--', alpha=0.75, linewidth=1.2)
        
        ax.set_title(f"Test Spectrum #{i+1}", fontsize=11, fontweight='bold')
        ax.set_ylabel("Intensity", fontsize=10)
        ax.grid(True, alpha=0.3, linestyle=':')
        if i == 0:
            ax.legend(loc='upper right', framealpha=0.9, fontsize=9)

    axes[-1].set_xlabel("Spectral Channel (Wavenumber Bin)", fontsize=10)
    plt.tight_layout()

    plot_path = "conventional_vs_polygaussnet_comparison.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Plot saved to '{plot_path}'.\n")


if __name__ == "__main__":
    run_comparison()
