"""
OVERFIT CHECK SCRIPT - ROUND 5 (PolyGaussNet2 vs. PolyGaussNet3 vs. PolyGaussNet3+)
-----------------------------------------------------------------------------------
Builds an overfitting benchmark dataset consisting of 480 spectra:
  - 3 Optical Stretching Modes: 1x, 1.5x, 2x
  - 5 Bin Sizes per Mode:
      * 1x:   800,  1016, 1200, 1400, 1600
      * 1.5x: 1200, 1524, 1800, 2100, 2400
      * 2x:   1600, 2032, 2400, 2800, 3200
  - 32 spectra sampled per bin size -> 15 groups x 32 spectra = 480 spectra total.

TRAINING REGIME (512 Epochs per Model):
  - Models: PolyGaussNet2 (sigma only), PolyGaussNet3 (bounded amp), PolyGaussNet3+ (expanded RF + bounded amp)
  - Optimizer: AdamW with lr = 1e-3 (1 order of magnitude higher than training 1e-4)
  - Weight Decay: 0.0 (unrestricted capacity to fit sample data)
  - Scheduler: ReduceLROnPlateau(patience=0, factor=0.9, min_lr=1e-6)
  - Loss: Dual-supervised LogCosh(BC) + LogCosh(Clean) + AsymmetricBaselinePenalty (lambda_asym=15.0)
  - Progress updates: ONLY prints when each model's overfit run is completely finished (NO per-epoch spam)
  - Final Report: Formatted terminal table reporting the final losses of all three models.
  - Visualization: 3x3 plot (Rows = Models, Columns = 1 random sample from 1x, 1.5x, 2x)
    showing Raw Spectra, Pure Ground Truth, and Model Predictions.
"""

import os
import sys
import time
import glob
import random
import importlib
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

# Configure paths so imports resolve whether executed from root or inside training_round5
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
for path in [PROJECT_ROOT, SCRIPT_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Import model architectures
from polygaussnet2 import PolyGaussNet as PolyGaussNet2
from polygaussnet3 import PolyGaussNet as PolyGaussNet3

try:
    polygaussnet3_plus_mod = importlib.import_module("polygaussnet3+")
    PolyGaussNet3Plus = polygaussnet3_plus_mod.PolyGaussNet
except ImportError as e:
    raise ImportError(f"Could not dynamically import PolyGaussNet3+: {e}")


# =====================================================================
# 1. CONFIGURATION & HYPERPARAMETERS
# =====================================================================
EPOCHS = 512                          # Epochs per model
LEARNING_RATE = 1e-3                  # 1 order of magnitude higher than training (1e-4 -> 1e-3)
WEIGHT_DECAY = 0.0                    # Zero weight decay for overfitting verification
LR_FACTOR = 0.9                       # ReduceLROnPlateau decay factor
LR_PATIENCE = 0                       # ReduceLROnPlateau patience
MIN_LR = 1e-6                         # Minimum learning rate

# Dataset Configuration
SAMPLES_PER_BIN = 32                  # 32 spectra per bin size
MODES = ["1x", "1.5x", "2x"]
MODE_BIN_SIZES = {
    "1x":   [800, 1016, 1200, 1400, 1600],
    "1.5x": [1200, 1524, 1800, 2100, 2400],
    "2x":   [1600, 2032, 2400, 2800, 3200]
}
DATA_DIR = "training_data5"
OUTPUT_PLOT_PATH = "overfit_check5.png"

# Model Architecture Parameters
POLY_ORDER = 7                        # Polynomial baseline degree
FILTER_KERNEL_SIZE = 63               # Gaussian filter window size
MIN_SIGMA = 0.2
MAX_SIGMA = 10.0
MIN_AMPLITUDE = 0.85
MAX_AMPLITUDE = 1.15

# Loss Weights (Matches training5 and training4)
LAMBDA_BC = 1.0
LAMBDA_CLEAN = 1.0
LAMBDA_ASYM = 15.0


# =====================================================================
# 2. LOSS FUNCTIONS
# =====================================================================
class LogCoshLoss(nn.Module):
    """Stable LogCosh loss for MAE-like outlier-robust supervision."""
    def __init__(self):
        super().__init__()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        x = y_pred - y_true
        return torch.mean(
            torch.abs(x) +
            torch.log1p(torch.exp(-2.0 * torch.abs(x))) -
            torch.log(torch.tensor(2.0, device=x.device, dtype=x.dtype))
        )


class AsymmetricBaselinePenalty(nn.Module):
    """Penalizes baseline predictions exceeding raw input (B_pred > X_raw)."""
    def __init__(self, lambda_asym: float = 15.0):
        super().__init__()
        self.lambda_asym = lambda_asym

    def forward(self, pred_baseline: torch.Tensor, raw_spectrum: torch.Tensor) -> torch.Tensor:
        excess = F.relu(pred_baseline - raw_spectrum)
        with torch.no_grad():
            num_violating_bins = (pred_baseline > raw_spectrum).float().sum(dim=-1, keepdim=True)
        return self.lambda_asym * torch.mean((excess ** 2) * num_violating_bins)


# =====================================================================
# 3. DATASET LOADING & RESOLUTION
# =====================================================================
def resolve_data_path(path_str: str) -> str:
    """Finds data directory whether executed from workspace root or subdirectory."""
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


def build_overfitting_dataset(data_dir: str, samples_per_bin: int = 32):
    """
    Loads exactly `samples_per_bin` (32) spectra from each bin size of each stretching mode.
    Returns:
      dataset_groups: list of dicts with keys:
        - 'mode': str ('1x', '1.5x', '2x')
        - 'bin_size': int
        - 'x': tensor (32, 1, L)
        - 'y_bc': tensor (32, 1, L)
        - 'y_clean': tensor (32, 1, L)
    Total spectra: 3 modes * 5 bin sizes * 32 samples = 480 spectra.
    """
    resolved_dir = resolve_data_path(data_dir)
    dataset_groups = []
    total_spectra = 0

    print(f"Loading overfitting dataset from: {resolved_dir}")

    for mode in MODES:
        bin_sizes = MODE_BIN_SIZES[mode]
        for b in bin_sizes:
            bin_dir = os.path.join(resolved_dir, mode, str(b))
            if not os.path.exists(bin_dir):
                raise FileNotFoundError(f"Missing bin folder: {bin_dir}")

            # Find first chunk file (e.g. spectra_chunk_01.npz)
            chunk_files = sorted(glob.glob(os.path.join(bin_dir, "*.npz")))
            if not chunk_files:
                raise FileNotFoundError(f"No .npz chunk files found in {bin_dir}")

            first_chunk = chunk_files[0]
            data = np.load(first_chunk)

            # Take the first `samples_per_bin` spectra
            x_raw = data["full_matrix"][:samples_per_bin].astype(np.float32)
            y_bc = data["pure_noise_cosmic_matrix"][:samples_per_bin].astype(np.float32)
            y_clean = data["pure_matrix"][:samples_per_bin].astype(np.float32)
            data.close()

            if len(x_raw) < samples_per_bin:
                raise ValueError(f"File {first_chunk} has only {len(x_raw)} spectra, expected >= {samples_per_bin}")

            # Shape: (32, 1, L)
            group = {
                "mode": mode,
                "bin_size": b,
                "x": torch.tensor(x_raw, dtype=torch.float32).unsqueeze(1),
                "y_bc": torch.tensor(y_bc, dtype=torch.float32).unsqueeze(1),
                "y_clean": torch.tensor(y_clean, dtype=torch.float32).unsqueeze(1)
            }
            dataset_groups.append(group)
            total_spectra += len(x_raw)

    print(f"Successfully constructed overfitting dataset: {total_spectra} spectra across {len(dataset_groups)} groups (32 spectra each).\n")
    return dataset_groups


# =====================================================================
# 4. MODEL OVERFIT TRAINING ROUTINE
# =====================================================================
def train_model_overfit(
    model_name: str,
    model: nn.Module,
    dataset_groups: list[dict],
    device: torch.device,
    epochs: int = EPOCHS,
    lr: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    lambda_bc: float = LAMBDA_BC,
    lambda_clean: float = LAMBDA_CLEAN,
    lambda_asym: float = LAMBDA_ASYM
) -> dict:
    """
    Trains a model on the 480 spectra for `epochs` (256) epochs without per-epoch printouts.
    Prints update ONLY upon complete overfit run.
    """
    model = model.to(device)
    criterion = LogCoshLoss().to(device)
    asym_penalty = AsymmetricBaselinePenalty(lambda_asym=lambda_asym).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=LR_FACTOR,
        patience=LR_PATIENCE,
        min_lr=MIN_LR
    )

    total_spectra = sum(g["x"].size(0) for g in dataset_groups)
    start_time = time.time()
    initial_loss = None

    for epoch in range(epochs):
        model.train()
        running_total_loss = 0.0
        running_clean_loss = 0.0
        running_bc_loss = 0.0
        running_asym_loss = 0.0

        # Shuffle order of the 15 bin groups each epoch
        group_indices = list(range(len(dataset_groups)))
        random.shuffle(group_indices)

        for idx in group_indices:
            group = dataset_groups[idx]
            bx = group["x"].to(device)
            by_bc = group["y_bc"].to(device)
            by_clean = group["y_clean"].to(device)
            bs = bx.size(0)

            # Forward pass
            clean_pred, pred_baseline, bc_pred, _ = model(bx)

            # Multi-loss computation
            loss_bc = criterion(bc_pred, by_bc)
            loss_clean = criterion(clean_pred, by_clean)
            loss_asym = asym_penalty(pred_baseline, bx)

            total_loss = (lambda_bc * loss_bc) + (lambda_clean * loss_clean) + loss_asym

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            running_total_loss += total_loss.item() * bs
            running_clean_loss += loss_clean.item() * bs
            running_bc_loss += loss_bc.item() * bs
            running_asym_loss += loss_asym.item() * bs

        avg_loss = running_total_loss / total_spectra
        avg_clean = running_clean_loss / total_spectra
        avg_bc = running_bc_loss / total_spectra
        avg_asym = running_asym_loss / total_spectra

        if initial_loss is None:
            initial_loss = avg_loss

        scheduler.step(avg_loss)

    elapsed_time = time.time() - start_time

    # Print notification ONLY upon model overfit completion as requested
    print(
        f"✓ [{model_name:<14}] Overfit complete ({epochs} epochs in {elapsed_time:.1f}s) | "
        f"Initial: {initial_loss:.6f} -> Final Loss: {avg_loss:.6f} "
        f"(Clean: {avg_clean:.6f}, BC: {avg_bc:.6f}, Asym: {avg_asym:.6f})"
    )

    return {
        "model_name": model_name,
        "model": model,
        "initial_loss": initial_loss,
        "final_loss": avg_loss,
        "final_clean": avg_clean,
        "final_bc": avg_bc,
        "final_asym": avg_asym,
        "elapsed_time": elapsed_time
    }


# =====================================================================
# 5. VISUALIZATION (3x3 GRID PLOT)
# =====================================================================
def plot_overfit_comparison(
    models_dict: dict[str, nn.Module],
    dataset_groups: list[dict],
    device: torch.device,
    save_path: str = OUTPUT_PLOT_PATH
):
    """
    Picks 1 random spectrum from each stretching mode (1x, 1.5x, 2x).
    Constructs a 3x3 plot:
      - Rows: PolyGaussNet2, PolyGaussNet3, PolyGaussNet3+
      - Columns: Mode 1x sample, Mode 1.5x sample, Mode 2x sample
      - Each cell shows: Raw Spectra, Pure Ground Truth, and Model Prediction.
    """
    print(f"\nGenerating 3x3 overfit comparison figure...")

    # Group dataset by mode
    groups_by_mode = {"1x": [], "1.5x": [], "2x": []}
    for g in dataset_groups:
        groups_by_mode[g["mode"]].append(g)

    # Pick 1 random spectrum per stretching mode
    selected_samples = []
    for mode in ["1x", "1.5x", "2x"]:
        g = random.choice(groups_by_mode[mode])
        sample_idx = random.randint(0, g["x"].size(0) - 1)
        selected_samples.append({
            "mode": mode,
            "bin_size": g["bin_size"],
            "raw": g["x"][sample_idx : sample_idx + 1],          # (1, 1, L)
            "pure": g["y_clean"][sample_idx : sample_idx + 1]    # (1, 1, L)
        })

    model_names = ["PolyGaussNet2", "PolyGaussNet3", "PolyGaussNet3+"]
    fig, axes = plt.subplots(3, 3, figsize=(18, 12), dpi=160)

    # Styling colors
    color_raw = "#a0a0a0"        # Light gray for raw spectrum
    color_pure = "#111111"       # Solid black for pure ground truth
    color_pred = "#d62728"       # Crimson red for model prediction

    for row_idx, m_name in enumerate(model_names):
        m = models_dict[m_name]
        m.eval()

        for col_idx, samp in enumerate(selected_samples):
            ax = axes[row_idx, col_idx]
            bx = samp["raw"].to(device)
            pure_gt = samp["pure"].squeeze().cpu().numpy()
            raw_input = samp["raw"].squeeze().cpu().numpy()

            with torch.no_grad():
                pred_clean, _, _, _ = m(bx)
                pred_clean_np = pred_clean.squeeze().cpu().numpy()

            # Plot the 3 curves requested: Raw spectra, Pure spectra, Model prediction
            ax.plot(raw_input, color=color_raw, lw=1.1, alpha=0.75, label="Raw Spectrum")
            ax.plot(pure_gt, color=color_pure, lw=1.3, label="Pure Ground Truth")
            ax.plot(pred_clean_np, color=color_pred, lw=1.4, linestyle="-", label=f"Pred ({m_name})")

            # Column headers (on top row)
            if row_idx == 0:
                ax.set_title(f"Mode: {samp['mode']}  |  {samp['bin_size']} Bins", fontsize=13, fontweight="bold", pad=8)

            # Row headers (on left column)
            if col_idx == 0:
                ax.set_ylabel(f"{m_name}\nIntensity (a.u.)", fontsize=11, fontweight="bold")
            else:
                ax.set_ylabel("Intensity (a.u.)", fontsize=9, color="#555555")

            ax.set_xlabel("Raman Shift / Spectral Bins", fontsize=9)
            ax.tick_params(direction="out", labelsize=8)
            ax.grid(True, linestyle=":", alpha=0.5)
            ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    plt.suptitle(f"PolyGaussNet Round 5 Overfitting Benchmark (480 Multi-Resolution Spectra, {len(selected_samples)} Modes)",
                 fontsize=15, fontweight="bold", y=0.995)
    plt.tight_layout()

    # Save figure
    resolved_save_path = save_path
    if not os.path.isabs(resolved_save_path):
        resolved_save_path = os.path.join(SCRIPT_DIR, save_path)

    plt.savefig(resolved_save_path, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved 3x3 overfit visualization plot to: '{resolved_save_path}'")


# =====================================================================
# 6. MAIN EXECUTION
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Run overfit check on PolyGaussNet2, PolyGaussNet3, and PolyGaussNet3+."
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help="Number of epochs per model")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE,
                        help="Initial learning rate (default: 1e-3, 1 order of magnitude above training)")
    parser.add_argument("--data-dir", type=str, default=DATA_DIR,
                        help="Path to training_data5 directory")
    parser.add_argument("--save-plot", type=str, default=OUTPUT_PLOT_PATH,
                        help="Output path for the 3x3 comparison figure")
    parser.add_argument("--poly-order", type=int, default=POLY_ORDER,
                        help="Order of polynomial baseline")
    parser.add_argument("--filter-kernel-size", type=int, default=FILTER_KERNEL_SIZE,
                        help="Odd kernel window size for adaptive filter")
    args = parser.parse_args()

    # Hardware device setup
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("=" * 82)
    print(" ROUND 5 OVERFIT CHECK: PolyGaussNet2 vs. PolyGaussNet3 vs. PolyGaussNet3+ ")
    print("=" * 82)
    print(f"  - Device:             {device}")
    print(f"  - Epochs per Model:   {args.epochs}")
    print(f"  - Initial LR:         {args.lr} (10x training rate)")
    print(f"  - Weight Decay:       {WEIGHT_DECAY} (disabled for pure overfitting test)")
    print(f"  - Scheduler:          ReduceLROnPlateau(patience={LR_PATIENCE}, factor={LR_FACTOR}, min_lr={MIN_LR})")
    print(f"  - Polynomial Order:   {args.poly_order} (Degree {args.poly_order})")
    print(f"  - Filter Kernel Size: {args.filter_kernel_size}")
    print(f"  - Dataset:            32 spectra x 5 bins x 3 modes = 480 spectra total")
    print("=" * 82 + "\n")

    # 1. Load overfitting dataset (480 spectra total)
    dataset_groups = build_overfitting_dataset(args.data_dir, samples_per_bin=SAMPLES_PER_BIN)

    # 2. Instantiate the three models
    models = {
        "PolyGaussNet2": PolyGaussNet2(
            poly_order=args.poly_order,
            filter_kernel_size=args.filter_kernel_size
        ),
        "PolyGaussNet3": PolyGaussNet3(
            poly_order=args.poly_order,
            filter_kernel_size=args.filter_kernel_size,
            min_amplitude=MIN_AMPLITUDE,
            max_amplitude=MAX_AMPLITUDE
        ),
        "PolyGaussNet3+": PolyGaussNet3Plus(
            poly_order=args.poly_order,
            filter_kernel_size=args.filter_kernel_size,
            min_sigma=MIN_SIGMA,
            max_sigma=MAX_SIGMA,
            min_amplitude=MIN_AMPLITUDE,
            max_amplitude=MAX_AMPLITUDE
        )
    }

    # 3. Train all three models sequentially
    print("Starting overfit training for all 3 models (per-epoch prints suppressed)...")
    results = []

    for name, model in models.items():
        res = train_model_overfit(
            model_name=name,
            model=model,
            dataset_groups=dataset_groups,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=WEIGHT_DECAY,
            lambda_bc=LAMBDA_BC,
            lambda_clean=LAMBDA_CLEAN,
            lambda_asym=LAMBDA_ASYM
        )
        results.append(res)

    # 4. Print final terminal summary report
    print("\n" + "=" * 88)
    print(f" FINAL OVERFIT LOSS BENCHMARK REPORT (480 SPECTRA - {args.epochs} EPOCHS) ")
    print("=" * 88)
    print(f"{'Model Name':<16} | {'Params':>10} | {'Initial Loss':>13} | {'Final Loss':>12} | {'Clean Loss':>11} | {'BC Loss':>10} | {'Asym Loss':>10}")
    print("-" * 88)

    for res in results:
        m = models[res["model_name"]]
        p_count = sum(p.numel() for p in m.parameters())
        print(
            f"{res['model_name']:<16} | "
            f"{p_count:>10,} | "
            f"{res['initial_loss']:>13.6f} | "
            f"{res['final_loss']:>12.6f} | "
            f"{res['final_clean']:>11.6f} | "
            f"{res['final_bc']:>10.6f} | "
            f"{res['final_asym']:>10.6f}"
        )
    print("=" * 88)

    # 5. Generate 3x3 comparison plot
    plot_overfit_comparison(
        models_dict=models,
        dataset_groups=dataset_groups,
        device=device,
        save_path=args.save_plot
    )
    print("\nOverfit check completed successfully!")


if __name__ == "__main__":
    main()
