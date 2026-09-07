"""
TRAINING SCRIPT - ROUND 5 (PolyGaussNet3+ on Multi-Resolution Raman Dataset)
----------------------------------------------------------------------------
Trains the input-size agnostic PolyGaussNet3+ architecture:
  Stage 1: Dynamic Polynomial Baseline Subtraction (Degree 7)
  Stage 2: Full-Resolution Dilated Residual CNN predicting per-bin [sigma, amplitude]
  Stage 3: Differentiable Adaptive Gaussian Filter Denoising (K=63) with Bounded Amplitude

KEY FEATURES & UPGRADES (ROUND 5):
- 100% Dynamic Input Length Agnostic (trained across 800 to 3200 spectral bins).
- Multi-Resolution Stretching Mode Support (1x, 1.5x, 2x optical sampling modes).
- Dynamic Stretching Mode Permutation (Default: random_order_all_modes):
    * Randomly permutes all 3 stretching modes (e.g., 1.5x -> 2x -> 1x) each epoch.
    * Memory-safe chunk streaming: loads strictly ONE chunk file at a time into RAM (~140 MB),
      freeing it immediately with gc.collect() and MPS cache cleanup before loading the next.
- Dual Supervised Log-Cosh loss on Baseline-Corrected and Clean/Pure spectra.
- Asymmetric Baseline Overprediction Penalty Loss (penalizes B_pred > X_raw).
- Scheduled learning rate (ReduceLROnPlateau with min_lr).
- Direct Apple Silicon Mac hardware acceleration (MPS) & CUDA / CPU fallback.
- Fast asynchronous dataset streaming from .npz files using ThreadPoolExecutor.
- Safe pause-and-resume on Ctrl+C (saves to `training_checkpoint5.pth`).
- Automatic checkpointing and best-model saving (`polygaussnet5.pth`).
"""

import os
import sys
import time
import glob
import random
import argparse
import gc
import importlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Configure paths so imports and models resolve whether running from project root or inside training_round5
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
for path in [PROJECT_ROOT, SCRIPT_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Dynamically import PolyGaussNet from polygaussnet3+ module
try:
    polygaussnet3_plus = importlib.import_module("polygaussnet3+")
    PolyGaussNet = polygaussnet3_plus.PolyGaussNet
except ImportError as e:
    raise ImportError(f"Could not import PolyGaussNet from 'polygaussnet3+.py': {e}")


# =====================================================================
# 1. TRAINING CONFIGURATION & HYPERPARAMETERS
# =====================================================================
EPOCHS = 16                          # Total number of training epochs
BATCH_SIZE = 128                      # Batch size for training
LEARNING_RATE = 1e-4                  # Initial learning rate for AdamW
MIN_LEARNING_RATE = 9e-8              # Minimum learning rate for scheduler
LR_PATIENCE = 0                       # Patience for ReduceLROnPlateau scheduler (0 = step after 1 stagnant epoch)
WEIGHT_DECAY = 1e-4                   # Weight decay for AdamW optimizer

# Paths
DATA_DIR = "training_data5"           # Dataset directory (contains 1x, 1.5x, 2x subfolders)
SAVE_MODEL_PATH = "polygaussnet5.pth" # Final best model weights
CHECKPOINT_PATH = "training_checkpoint5.pth"  # Checkpoint file for pause/resume
NO_RESUME = False                     # True = ignore existing checkpoint and start from scratch

# Dataset Modes
AVAILABLE_MODES = ["1x", "1.5x", "2x"]

# Model Architecture Hyperparameters (PolyGaussNet3+)
POLY_ORDER = 7                        # Order of baseline polynomial (Degree 7)
FILTER_KERNEL_SIZE = 63               # Odd kernel window size for adaptive Gaussian filter
MIN_SIGMA = 0.2                       # Minimum width (sigma) for adaptive Gaussian filter
MAX_SIGMA = 10.0                      # Maximum width (sigma) for adaptive Gaussian filter
MIN_AMPLITUDE = 0.85                  # Bounded amplitude modulation minimum
MAX_AMPLITUDE = 1.15                  # Bounded amplitude modulation maximum

# Loss Weighting Factors
LAMBDA_BC = 1.0                       # Scaling factor for baseline-corrected latent loss
LAMBDA_CLEAN = 1.0                    # Scaling factor for final pure/clean output loss
LAMBDA_ASYM = 15.0                    # Scaling factor for asymmetric baseline overprediction penalty


# =====================================================================
# 2. CLI ARGUMENT PARSER
# =====================================================================
def parse_args():
    """Parse command-line arguments, defaulting to the variables defined above."""
    parser = argparse.ArgumentParser(
        description="Train PolyGaussNet3+ on Multi-Resolution Raman Dataset (Round 5).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Batch size for training")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE,
                        help="Initial learning rate")
    parser.add_argument("--min-lr", type=float, default=MIN_LEARNING_RATE,
                        help="Minimum learning rate")
    parser.add_argument("--lr-patience", type=int, default=LR_PATIENCE,
                        help="Patience for ReduceLROnPlateau scheduler")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Weight decay for AdamW")
    parser.add_argument("--data-dir", type=str, default=DATA_DIR,
                        help="Directory containing 1x, 1.5x, 2x multi-resolution dataset folders")
    parser.add_argument("--save-model", type=str, default=SAVE_MODEL_PATH,
                        help="Path to save the best model weights")
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT_PATH,
                        help="Checkpoint filename to save/resume training state")
    parser.add_argument("--no-resume", action="store_true", default=NO_RESUME,
                        help="Ignore existing checkpoint and start training from scratch")
    parser.add_argument("--grouped-modes", action="store_true", default=False,
                        help="Train mode-by-mode sequentially (e.g. all 1x, then 1.5x, then 2x) instead of fully interleaving all 120 chunks")
    parser.add_argument("--modes", nargs="+", default=AVAILABLE_MODES,
                        help="List of stretching modes available in the dataset")
    parser.add_argument("--poly-order", type=int, default=POLY_ORDER,
                        help="Order of the baseline polynomial")
    parser.add_argument("--filter-kernel-size", type=int, default=FILTER_KERNEL_SIZE,
                        help="Odd kernel window size for adaptive filter")
    parser.add_argument("--min-sigma", type=float, default=MIN_SIGMA,
                        help="Minimum sigma for adaptive Gaussian filter")
    parser.add_argument("--max-sigma", type=float, default=MAX_SIGMA,
                        help="Maximum sigma for adaptive Gaussian filter")
    parser.add_argument("--min-amplitude", type=float, default=MIN_AMPLITUDE,
                        help="Minimum bounded amplitude modulation")
    parser.add_argument("--max-amplitude", type=float, default=MAX_AMPLITUDE,
                        help="Maximum bounded amplitude modulation")
    parser.add_argument("--lambda-bc", type=float, default=LAMBDA_BC,
                        help="Weight for baseline-corrected loss")
    parser.add_argument("--lambda-clean", type=float, default=LAMBDA_CLEAN,
                        help="Weight for clean pure loss")
    parser.add_argument("--lambda-asym", type=float, default=LAMBDA_ASYM,
                        help="Weight for asymmetric baseline overprediction penalty")
    return parser.parse_args()


# =====================================================================
# 3. LOSS FUNCTIONS
# =====================================================================
class LogCoshLoss(nn.Module):
    """
    Logarithm of the hyperbolic cosine loss:
        L(y_pred, y_true) = log(cosh(y_pred - y_true))
    Smooth, outlier-robust approximation to L1 (MAE) for stable autograd optimization.
    """
    def __init__(self):
        super(LogCoshLoss, self).__init__()
        # Stable formula: log(cosh(x)) = |x| + log1p(exp(-2|x|)) - log(2)
        # Precompute log(2) constant to avoid tensor allocations on every forward step
        self.log_2 = float(np.log(2.0))

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        x = y_pred - y_true
        return torch.mean(
            torch.abs(x) +
            torch.log1p(torch.exp(-2.0 * torch.abs(x))) -
            self.log_2
        )


class AsymmetricBaselinePenalty(nn.Module):
    """
    Penalizes baseline predictions that exceed the raw input signal (B_pred > X_raw).
    The penalty is proportional to:
      1. The constant hyperparameter (lambda_asym)
      2. The number of bins on which the predicted baseline is higher than raw spectrum
      3. The squared magnitude of the overprediction: ReLU(B_pred - X_raw)^2
    """
    def __init__(self, lambda_asym: float = 1.0):
        super(AsymmetricBaselinePenalty, self).__init__()
        self.lambda_asym = lambda_asym

    def forward(self, pred_baseline: torch.Tensor, raw_spectrum: torch.Tensor) -> torch.Tensor:
        # Violation magnitude where baseline rises above raw spectrum: (Batch, 1, L)
        excess = F.relu(pred_baseline - raw_spectrum)

        # Number of violating bins per spectrum: (Batch, 1, 1)
        with torch.no_grad():
            num_violating_bins = (pred_baseline > raw_spectrum).float().sum(dim=-1, keepdim=True)

        # Loss is proportional to constant hyperparameter * number of violating bins * squared excess
        loss = self.lambda_asym * torch.mean((excess ** 2) * num_violating_bins)
        return loss


# =====================================================================
# 4. DATA LOADING & PATH RESOLUTION
# =====================================================================
def resolve_data_path(path_str: str) -> str:
    """Finds data directory whether executed from workspace root or subdirectory."""
    if os.path.isabs(path_str) and os.path.exists(path_str):
        return path_str
    if os.path.exists(path_str):
        return os.path.abspath(path_str)

    # Try relative to script directory
    script_rel = os.path.join(SCRIPT_DIR, path_str)
    if os.path.exists(script_rel):
        return script_rel

    # Try relative to project root
    root_rel = os.path.join(PROJECT_ROOT, path_str)
    if os.path.exists(root_rel):
        return root_rel

    # Try within training_round5
    round5_rel = os.path.join(PROJECT_ROOT, "training_round5", path_str)
    if os.path.exists(round5_rel):
        return round5_rel

    return path_str


def discover_mode_files(data_dir: str, modes: list[str]) -> dict[str, list[str]]:
    """Discovers all .npz dataset files grouped by stretching mode."""
    mode_files = {}
    for mode in modes:
        mode_path = os.path.join(data_dir, mode)
        if not os.path.exists(mode_path):
            continue

        # Look recursively inside bin subfolders (e.g. 1x/800/*.npz) or directly in mode folder
        files = sorted(glob.glob(os.path.join(mode_path, "**", "*.npz"), recursive=True))
        if not files:
            files = sorted(glob.glob(os.path.join(mode_path, "*.npz")))

        if files:
            mode_files[mode] = files

    return mode_files


def select_epoch_files(mode_files: dict[str, list[str]], fully_interleaved: bool = True) -> tuple[list[str], str]:
    """
    Selects and orders dataset files for the epoch:
    - If fully_interleaved=True (default): Randomly shuffles ALL 120 chunk files together across
      all stretching modes (1x, 1.5x, 2x) and bin resolutions. Each loaded chunk is chosen at random
      from the entire multi-resolution dataset, ensuring uniform gradient dynamics without mode bias.
    - If fully_interleaved=False: Groups chunks by stretching mode in randomized order (e.g. 1.5x -> 2x -> 1x).
    """
    available_modes = list(mode_files.keys())
    if not available_modes:
        return [], "none"

    if fully_interleaved:
        epoch_files = [f for file_list in mode_files.values() for f in file_list]
        random.shuffle(epoch_files)
        return epoch_files, f"Fully Interleaved ({len(epoch_files)} chunks across {', '.join(available_modes)})"
    else:
        shuffled_modes = list(available_modes)
        random.shuffle(shuffled_modes)
        epoch_files = []
        for mode in shuffled_modes:
            mode_file_list = list(mode_files[mode])
            np.random.shuffle(mode_file_list)
            epoch_files.extend(mode_file_list)
        return epoch_files, " -> ".join(shuffled_modes)


def load_file_data(filepath: str):
    """Load an .npz dataset file from disk into PyTorch float tensors with minimal memory overhead."""
    with np.load(filepath) as data:
        X = torch.from_numpy(data["full_matrix"]).float().unsqueeze(1)
        Y_bc = torch.from_numpy(data["pure_noise_cosmic_matrix"]).float().unsqueeze(1)
        Y_clean = torch.from_numpy(data["pure_matrix"]).float().unsqueeze(1)
    return X, Y_bc, Y_clean


# =====================================================================
# 5. MAIN TRAINING ROUTINE WITH PAUSE / RESUME
# =====================================================================
def train(args=None):
    if args is None:
        args = parse_args()

    # Hardware device setup
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("=" * 78)
    print(" PolyGaussNet3+ Training (Round 5 Multi-Resolution Raman Preprocessing) ")
    print("=" * 78)
    print(f"  - Device:             {device}")
    print(f"  - Batch Size:         {args.batch_size}")
    print(f"  - Initial LR:         {args.lr}")
    print(f"  - Min LR:             {args.min_lr}")
    print(f"  - Weight Decay:       {args.weight_decay}")
    print(f"  - Total Epochs:       {args.epochs}")
    print(f"  - Polynomial Order:   {args.poly_order} (Degree {args.poly_order})")
    print(f"  - Filter Kernel Size: {args.filter_kernel_size}")
    print(f"  - Sigma Range:        [{args.min_sigma}, {args.max_sigma}]")
    print(f"  - Amplitude Range:    [{args.min_amplitude}, {args.max_amplitude}]")
    print(f"  - Lambda BC:          {args.lambda_bc}")
    print(f"  - Lambda Clean:       {args.lambda_clean}")
    print(f"  - Lambda Asymmetric:  {args.lambda_asym}")

    # Resolve Data Directory & Files
    resolved_data_dir = resolve_data_path(args.data_dir)
    mode_files = discover_mode_files(resolved_data_dir, args.modes)

    if not mode_files:
        # Fallback: check if .npz files exist directly in the folder
        fallback_files = sorted(glob.glob(os.path.join(resolved_data_dir, "**", "*.npz"), recursive=True))
        if fallback_files:
            mode_files["default"] = fallback_files

    if not mode_files:
        print(f"\n[ERROR] No .npz dataset files found in '{resolved_data_dir}'.")
        print("Please verify the Round 5 dataset has been generated via: python create_dataset5.py")
        return

    total_chunks = sum(len(f) for f in mode_files.values())
    print(f"  - Dataset Location:   {resolved_data_dir}")
    print(f"  - Discovered Modes:   {list(mode_files.keys())}")
    for mode, files in mode_files.items():
        print(f"      * Mode '{mode}': {len(files)} files")
    print(f"  - Total Dataset Size: {total_chunks} chunk files")
    print("=" * 78 + "\n")

    # Initialize PolyGaussNet3+
    model = PolyGaussNet(
        poly_order=args.poly_order,
        filter_kernel_size=args.filter_kernel_size,
        min_sigma=args.min_sigma,
        max_sigma=args.max_sigma,
        min_amplitude=args.min_amplitude,
        max_amplitude=args.max_amplitude
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"PolyGaussNet3+ Parameters: {total_params:,} total ({trainable_params:,} trainable)")

    # Losses & Optimizer
    criterion = LogCoshLoss().to(device)
    asym_penalty = AsymmetricBaselinePenalty(lambda_asym=args.lambda_asym).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.8,
        patience=args.lr_patience,
        min_lr=args.min_lr
    )

    # -----------------------------------------------------------------
    # Checkpoint Resume Logic
    # -----------------------------------------------------------------
    start_epoch = 0
    best_loss = float('inf')
    checkpoint_path = args.checkpoint
    if not os.path.isabs(checkpoint_path):
        checkpoint_path = os.path.join(SCRIPT_DIR, args.checkpoint)

    save_model_dest = args.save_model
    if not os.path.isabs(save_model_dest):
        save_model_dest = os.path.join(SCRIPT_DIR, args.save_model)

    if not args.no_resume and os.path.exists(checkpoint_path):
        print(f"Found checkpoint at '{checkpoint_path}'. Resuming training...")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])

        if 'optimizer_state_dict' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except Exception as e:
                print(f"Note: Could not restore optimizer state ({e}). Continuing with initialized optimizer.")

        if 'scheduler_state_dict' in checkpoint:
            try:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                scheduler.patience = args.lr_patience
            except Exception as e:
                print(f"Note: Could not restore scheduler state ({e}). Continuing with initialized scheduler.")

        best_loss = checkpoint.get('best_loss', checkpoint.get('loss', float('inf')))
        scheduler.best = best_loss
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"Successfully resumed from Epoch {start_epoch + 1} of {args.epochs} (Previous best loss: {best_loss:.6f}).\n")
    else:
        if args.no_resume:
            print("(no_resume=True): Starting training from scratch.\n")
        else:
            print("No checkpoint found. Starting training from scratch.\n")

    if start_epoch >= args.epochs:
        print(f"Model has already been trained for {start_epoch} epochs (target: {args.epochs}).")
        print(f"To train further, increase epochs (e.g., pass --epochs {start_epoch + 50}).")
        return

    print("=" * 78)
    print(f"Starting Training: Epochs {start_epoch + 1} to {args.epochs}")
    print("=" * 78)

    executor = ThreadPoolExecutor(max_workers=1)

    try:
        for epoch in range(start_epoch, args.epochs):
            epoch_start = time.time()
            model.train()

            # Dynamic Stretching Mode Permutation for this epoch
            epoch_files, epoch_mode_desc = select_epoch_files(mode_files, fully_interleaved=not args.grouped_modes)
            if not epoch_files:
                print(f"[Warning] No files selected for Epoch {epoch + 1}. Skipping.")
                continue

            print(f"\n>>> Epoch [{epoch + 1:3d}/{args.epochs}] Mode Selection: [{epoch_mode_desc}] ({len(epoch_files)} files to train)")

            running_total_loss = 0.0
            running_bc_loss = 0.0
            running_clean_loss = 0.0
            running_asym_loss = 0.0
            total_samples = 0

            # Asynchronously prefetch the first file
            future = executor.submit(load_file_data, epoch_files[0])

            for file_idx in range(len(epoch_files)):
                X, Y_bc, Y_clean = future.result()

                # Prefetch next file in background
                if file_idx + 1 < len(epoch_files):
                    future = executor.submit(load_file_data, epoch_files[file_idx + 1])

                dataset = TensorDataset(X, Y_bc, Y_clean)
                loader = DataLoader(
                    dataset,
                    batch_size=args.batch_size,
                    shuffle=True,
                    drop_last=False
                )

                file_start_time = time.time()
                file_total_loss = 0.0
                file_bc_loss = 0.0
                file_clean_loss = 0.0
                file_asym_loss = 0.0
                file_samples = 0

                # Extract chunk resolution info from tensor
                current_bins = X.shape[-1]

                for batch_idx, (batch_x, batch_y_bc, batch_y_clean) in enumerate(loader):
                    batch_x = batch_x.to(device)
                    batch_y_bc = batch_y_bc.to(device)
                    batch_y_clean = batch_y_clean.to(device)

                    # Forward pass through PolyGaussNet3+
                    clean_pred, pred_baseline, bc_pred, latent_params = model(batch_x)

                    # Multi-component Loss Computation
                    loss_bc = criterion(bc_pred, batch_y_bc)
                    loss_clean = criterion(clean_pred, batch_y_clean)
                    loss_asym = asym_penalty(pred_baseline, batch_x)

                    total_loss = (args.lambda_bc * loss_bc) + (args.lambda_clean * loss_clean) + loss_asym

                    # Backward & Step
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()

                    bs = batch_x.size(0)
                    file_total_loss += total_loss.item() * bs
                    file_bc_loss += loss_bc.item() * bs
                    file_clean_loss += loss_clean.item() * bs
                    file_asym_loss += loss_asym.item() * bs
                    file_samples += bs

                    running_total_loss += total_loss.item() * bs
                    running_bc_loss += loss_bc.item() * bs
                    running_clean_loss += loss_clean.item() * bs
                    running_asym_loss += loss_asym.item() * bs
                    total_samples += bs

                # Per-file update printout
                file_duration = time.time() - file_start_time
                file_avg_loss = file_total_loss / file_samples if file_samples > 0 else 0
                file_avg_bc = file_bc_loss / file_samples if file_samples > 0 else 0
                file_avg_clean = file_clean_loss / file_samples if file_samples > 0 else 0
                file_avg_asym = file_asym_loss / file_samples if file_samples > 0 else 0
                file_throughput = file_samples / file_duration if file_duration > 0 else 0

                rel_file_path = os.path.relpath(epoch_files[file_idx], resolved_data_dir)
                print(
                    f"Epoch [{epoch + 1:3d}/{args.epochs}] | "
                    f"File [{file_idx + 1:3d}/{len(epoch_files)}] ({rel_file_path}, L={current_bins}) | "
                    f"Loss: {file_avg_loss:.5f} (BC: {file_avg_bc:.4f}, Clean: {file_avg_clean:.4f}, Asym: {file_avg_asym:.4f}) | "
                    f"{file_throughput:6.1f} spec/s"
                )

                del X, Y_bc, Y_clean, dataset, loader
                gc.collect()
                if device.type == "mps":
                    torch.mps.empty_cache()
                elif device.type == "cuda":
                    torch.cuda.empty_cache()

            epoch_duration = time.time() - epoch_start
            avg_loss = running_total_loss / total_samples if total_samples > 0 else 0
            avg_bc_loss = running_bc_loss / total_samples if total_samples > 0 else 0
            avg_clean_loss = running_clean_loss / total_samples if total_samples > 0 else 0
            avg_asym_loss = running_asym_loss / total_samples if total_samples > 0 else 0
            throughput = total_samples / epoch_duration if epoch_duration > 0 else 0

            scheduler.step(avg_loss)
            current_lr = optimizer.param_groups[0]['lr']

            # Epoch completion summary
            print("-" * 78)
            print(f"==> Epoch [{epoch + 1:3d}/{args.epochs}] Completed in {epoch_duration:.1f}s ({throughput:.1f} spectra/s) [Mode: {epoch_mode_desc}]")
            print(f"    Avg Total Loss: {avg_loss:.6f} | Avg BC: {avg_bc_loss:.6f} | Avg Clean: {avg_clean_loss:.6f} | Avg Asym: {avg_asym_loss:.6f}")
            print(f"    Learning Rate:  {current_lr:.6e}")

            # Best model auto-save
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.state_dict(), save_model_dest)
                print(f"    ★ New best model saved to '{save_model_dest}' (Loss: {best_loss:.6f})")

            # Save full training checkpoint for pause/resume
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_loss': best_loss,
                'loss': avg_loss,
                'poly_order': args.poly_order,
                'filter_kernel_size': args.filter_kernel_size,
                'min_sigma': args.min_sigma,
                'max_sigma': args.max_sigma,
                'min_amplitude': args.min_amplitude,
                'max_amplitude': args.max_amplitude,
                'lambda_asym': args.lambda_asym,
                'last_mode': epoch_mode_desc
            }, checkpoint_path)
            print(f"    Saved checkpoint to '{checkpoint_path}'")
            print("-" * 78)

        print(f"\nTraining complete! Best overall loss: {best_loss:.6f}. Best model saved to '{save_model_dest}'.")

    except KeyboardInterrupt:
        print("\n" + "=" * 78)
        print(" Training safely paused by user (Ctrl+C).")
        # Mark the last completed epoch so resuming restarts the interrupted epoch
        last_completed = (epoch - 1) if 'epoch' in locals() else max(-1, start_epoch - 1)

        # Calculate current running loss or fallback safely
        if 'running_total_loss' in locals() and 'total_samples' in locals() and total_samples > 0:
            calc_loss = running_total_loss / total_samples
        elif 'avg_loss' in locals():
            calc_loss = avg_loss
        elif 'best_loss' in locals() and best_loss != float('inf'):
            calc_loss = best_loss
        else:
            calc_loss = 0.0

        current_best = best_loss if 'best_loss' in locals() else calc_loss
        current_mode = epoch_mode_desc if 'epoch_mode_desc' in locals() else "interrupted"

        torch.save({
            'epoch': last_completed,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_loss': current_best,
            'loss': calc_loss,
            'poly_order': args.poly_order,
            'filter_kernel_size': args.filter_kernel_size,
            'min_sigma': args.min_sigma,
            'max_sigma': args.max_sigma,
            'min_amplitude': args.min_amplitude,
            'max_amplitude': args.max_amplitude,
            'lambda_asym': args.lambda_asym,
            'last_mode': current_mode
        }, checkpoint_path)
        print(f" Checkpoint preserved at: '{checkpoint_path}'")
        print(" Resume anytime by running: python training5.py")
        print("=" * 78 + "\n")
    finally:
        executor.shutdown(wait=False)
        gc.collect()
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    train()
