"""
TRAINING SCRIPT - ROUND 6 (PolyGaussNet3 on Variable-Bin Raman Dataset)
----------------------------------------------------------------------------
Trains the input-size agnostic PolyGaussNet3 architecture:
  Stage 1: Dynamic Polynomial Baseline Subtraction (Degree 7)
  Stage 2: Full-Resolution Dilated Residual CNN predicting per-bin [sigma, amplitude]
  Stage 3: Differentiable Adaptive Gaussian Filter Denoising (K=31) with Bounded Amplitude

KEY FEATURES (ROUND 6):
- Architecture: PolyGaussNet3 (148,946 parameters, RF = 127 bins, K=31).
- Dynamic Input Length Agnostic: Trained across variable bin resolutions (800 to 1600 bins).
- Interleaved Multi-Bin Training: Randomly shuffles chunk files across all bin sizes each epoch.
- Dual Supervised Log-Cosh loss on Baseline-Corrected and Clean/Pure spectra.
- Asymmetric Baseline Overprediction Penalty Loss (penalizes B_pred > X_raw, lambda=15.0).
- Scheduled learning rate (ReduceLROnPlateau with min_lr=9e-8).
- Direct Apple Silicon Mac hardware acceleration (MPS) & CUDA / CPU fallback.
- Fast asynchronous dataset streaming from .npz files using ThreadPoolExecutor.
- Memory-safe chunk streaming: loads strictly ONE chunk file at a time into RAM (~140 MB).
- Safe pause-and-resume on Ctrl+C (saves to `training_checkpoint6.pth`).
- Automatic checkpointing and best-model saving (`polygaussnet6.pth`).
- Dual logging to stdout and `training_history6.txt`.
"""

import os
import sys
import time
import glob
import random
import argparse
import gc
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Configure paths so imports and models resolve whether running from project root or inside training_round6
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
for path in [PROJECT_ROOT, SCRIPT_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

from polygaussnet3 import PolyGaussNet

# =====================================================================
# 1. TRAINING CONFIGURATION & HYPERPARAMETERS
# =====================================================================
EPOCHS = 128                          # Total number of training epochs
BATCH_SIZE = 512                      # Batch size for training
LEARNING_RATE = 1e-4                  # Initial learning rate for AdamW
MIN_LEARNING_RATE = 9e-8              # Minimum learning rate for scheduler
LR_PATIENCE = 1                       # Patience for ReduceLROnPlateau scheduler (step after 1 stagnant epoch)
LR_FACTOR = 0.8                       # Decay factor for ReduceLROnPlateau (matches Round 4: 20% reduction)
WEIGHT_DECAY = 1e-4                   # Weight decay for AdamW optimizer

# Paths
DATA_DIR = "training_data6"           # Dataset directory (contains 800, 1016, 1200, 1400, 1600 subfolders)
SAVE_MODEL_PATH = "polygaussnet6.pth" # Final best model weights
CHECKPOINT_PATH = "training_checkpoint6.pth"  # Checkpoint file for pause/resume
LOG_FILE_PATH = "training_history6.txt"        # Training stdout log

# Model Architecture Hyperparameters (PolyGaussNet3)
POLY_ORDER = 7                        # Order of baseline polynomial (Degree 7)
FILTER_KERNEL_SIZE = 31               # Odd kernel window size for adaptive Gaussian filter
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
    parser = argparse.ArgumentParser(
        description="Train PolyGaussNet3 on Variable-Bin Raman Dataset (Round 6).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Mini-batch size for training")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE,
                        help="Initial learning rate")
    parser.add_argument("--min-lr", type=float, default=MIN_LEARNING_RATE,
                        help="Minimum learning rate for scheduler")
    parser.add_argument("--lr-factor", type=float, default=LR_FACTOR,
                        help="Decay factor for ReduceLROnPlateau scheduler")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="AdamW weight decay coefficient")
    parser.add_argument("--poly-order", type=int, default=POLY_ORDER,
                        help="Order of Chebyshev polynomial baseline")
    parser.add_argument("--filter-kernel-size", type=int, default=FILTER_KERNEL_SIZE,
                        help="Kernel size for Gaussian filter (must be odd)")
    parser.add_argument("--min-amplitude", type=float, default=MIN_AMPLITUDE,
                        help="Minimum bounded amplitude modulation")
    parser.add_argument("--max-amplitude", type=float, default=MAX_AMPLITUDE,
                        help="Maximum bounded amplitude modulation")
    parser.add_argument("--lambda-bc", type=float, default=LAMBDA_BC,
                        help="Weight for baseline-corrected latent loss")
    parser.add_argument("--lambda-clean", type=float, default=LAMBDA_CLEAN,
                        help="Weight for final pure spectrum loss")
    parser.add_argument("--lambda-asym", type=float, default=LAMBDA_ASYM,
                        help="Weight for asymmetric baseline penalty")
    parser.add_argument("--data-dir", type=str, default=DATA_DIR,
                        help="Directory containing training dataset")
    parser.add_argument("--save-model", type=str, default=SAVE_MODEL_PATH,
                        help="Filename to save best model weights")
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT_PATH,
                        help="Filename for training pause/resume checkpoint")
    parser.add_argument("--no-resume", action="store_true",
                        help="Ignore existing checkpoint and start from epoch 1")
    return parser.parse_args()


# =====================================================================
# 3. LOSS FUNCTIONS
# =====================================================================
class LogCoshLoss(nn.Module):
    """Numerically stable Log-Cosh loss function."""
    def __init__(self):
        super().__init__()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        diff = y_pred - y_true
        return torch.mean(
            torch.abs(diff) +
            torch.log1p(torch.exp(-2.0 * torch.abs(diff))) -
            torch.log(torch.tensor(2.0, device=diff.device, dtype=diff.dtype))
        )


class AsymmetricBaselinePenalty(nn.Module):
    """
    Penalizes baseline overprediction where B_pred > X_raw.
    Penalty = lambda_asym * mean( ReLU(B_pred - X_raw)^2 * num_violating_bins )
    """
    def __init__(self, lambda_asym: float = 15.0):
        super().__init__()
        self.lambda_asym = lambda_asym

    def forward(self, pred_baseline: torch.Tensor, raw_spectrum: torch.Tensor) -> torch.Tensor:
        excess = F.relu(pred_baseline - raw_spectrum)
        with torch.no_grad():
            num_violating_bins = (pred_baseline > raw_spectrum).float().sum(dim=-1, keepdim=True)
        return self.lambda_asym * torch.mean((excess ** 2) * num_violating_bins)


# =====================================================================
# 4. DATASET STREAMING & PATH RESOLUTION
# =====================================================================
def resolve_path(path_str: str) -> str:
    """Finds target directory or file whether executed from root or subdirectory."""
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
    round6_rel = os.path.join(PROJECT_ROOT, "training_round6", path_str)
    if os.path.exists(round6_rel):
        return round6_rel
    return path_str


def discover_dataset_files(data_dir: str) -> list[str]:
    """Discovers all .npz chunk files inside data_dir (including subdirectories)."""
    files = sorted(glob.glob(os.path.join(data_dir, "**", "*.npz"), recursive=True))
    if not files:
        files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    return files


def load_file_data(filepath: str):
    """Load an .npz dataset file into PyTorch float tensors."""
    with np.load(filepath) as data:
        X = torch.from_numpy(data["full_matrix"]).float().unsqueeze(1)
        Y_bc = torch.from_numpy(data["pure_noise_cosmic_matrix"]).float().unsqueeze(1)
        Y_clean = torch.from_numpy(data["pure_matrix"]).float().unsqueeze(1)
    return X, Y_bc, Y_clean


# =====================================================================
# 5. DUAL LOGGER (Console + File)
# =====================================================================
class DualLogger:
    """Writes output simultaneously to stdout and a log file."""
    def __init__(self, filepath: str):
        self.terminal = sys.stdout
        self.log = open(filepath, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


# =====================================================================
# 6. MAIN TRAINING ROUTINE WITH PAUSE / RESUME
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

    # Set up dual logging
    log_path = resolve_path(LOG_FILE_PATH) if os.path.isabs(LOG_FILE_PATH) else os.path.join(SCRIPT_DIR, LOG_FILE_PATH)
    sys.stdout = DualLogger(log_path)

    print("\n" + "=" * 80)
    print(" PolyGaussNet3 Training (Round 6 Variable-Bin Raman Preprocessing) ")
    print("=" * 80)
    print(f"  - Device:             {device}")
    print(f"  - Batch Size:         {args.batch_size}")
    print(f"  - Initial LR:         {args.lr}")
    print(f"  - Min LR:             {args.min_lr}")
    print(f"  - LR Decay Factor:    {args.lr_factor}")
    print(f"  - Weight Decay:       {args.weight_decay}")
    print(f"  - Total Epochs:       {args.epochs}")
    print(f"  - Polynomial Order:   Degree {args.poly_order}")
    print(f"  - Filter Kernel Size: {args.filter_kernel_size}")
    print(f"  - Amplitude Range:    [{args.min_amplitude}, {args.max_amplitude}]")
    print(f"  - Lambda BC:          {args.lambda_bc}")
    print(f"  - Lambda Clean:       {args.lambda_clean}")
    print(f"  - Lambda Asymmetric:  {args.lambda_asym}")

    # Resolve Data Directory & Files
    resolved_data_dir = resolve_path(args.data_dir)
    chunk_files = discover_dataset_files(resolved_data_dir)

    if not chunk_files:
        print(f"\n[ERROR] No .npz dataset files found in '{resolved_data_dir}'.")
        print("Please generate the Round 6 dataset first via:")
        print("  python training_round6/create_dataset6.py")
        return

    print(f"  - Dataset Location:   {resolved_data_dir}")
    print(f"  - Total Chunk Files:  {len(chunk_files)} files")
    print("=" * 80 + "\n")

    # Initialize PolyGaussNet3
    model = PolyGaussNet(
        poly_order=args.poly_order,
        filter_kernel_size=args.filter_kernel_size,
        min_amplitude=args.min_amplitude,
        max_amplitude=args.max_amplitude
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"PolyGaussNet3 Parameters: {total_params:,} total ({trainable_params:,} trainable)")

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
        factor=args.lr_factor,
        patience=LR_PATIENCE,
        min_lr=args.min_lr
    )

    # Resolve checkpoint & save paths
    checkpoint_file = resolve_path(args.checkpoint) if os.path.isabs(args.checkpoint) else os.path.join(SCRIPT_DIR, args.checkpoint)
    save_model_file = resolve_path(args.save_model) if os.path.isabs(args.save_model) else os.path.join(SCRIPT_DIR, args.save_model)

    start_epoch = 1
    best_loss = float('inf')

    # Resume from checkpoint if present
    if not args.no_resume and os.path.exists(checkpoint_file):
        try:
            print(f"Found checkpoint at '{checkpoint_file}'. Resuming...")
            ckpt = torch.load(checkpoint_file, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if "scheduler_state_dict" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
                # Override checkpoint factor with configured lr_factor and reset bad epochs counter
                scheduler.factor = args.lr_factor
                scheduler.num_bad_epochs = 0
            start_epoch = ckpt.get("epoch", 0) + 1
            best_loss = ckpt.get("best_loss", float('inf'))
            print(f"Successfully resumed at Epoch {start_epoch} (Best Loss so far: {best_loss:.6f})\n")
        except Exception as e:
            print(f"[WARNING] Could not load checkpoint: {e}. Starting from scratch.\n")

    if start_epoch > args.epochs:
        print(f"Model already fully trained to epoch {args.epochs}. Nothing to do.")
        return

    print("=" * 80)
    print(f"Starting Training: Epochs {start_epoch} to {args.epochs}")
    print("=" * 80 + "\n")

    thread_pool = ThreadPoolExecutor(max_workers=1)

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            epoch_start_time = time.time()
            epoch_loss_sum = 0.0
            epoch_bc_sum = 0.0
            epoch_clean_sum = 0.0
            epoch_asym_sum = 0.0
            epoch_samples = 0

            # Shuffle chunks across all bin sizes for interleaved learning
            epoch_files = list(chunk_files)
            random.shuffle(epoch_files)
            num_files = len(epoch_files)

            print(f"\n>>> Epoch [{epoch:3d}/{args.epochs:3d}] Interleaved Training ({num_files} chunk files)")

            # Prefetch the first file asynchronously
            future_data = thread_pool.submit(load_file_data, epoch_files[0])

            for file_idx, current_filepath in enumerate(epoch_files, 1):
                # Retrieve currently needed file data
                X, Y_bc, Y_clean = future_data.result()

                # Trigger prefetch of next file in background
                if file_idx < num_files:
                    next_filepath = epoch_files[file_idx]
                    future_data = thread_pool.submit(load_file_data, next_filepath)
                else:
                    future_data = None

                file_samples = X.shape[0]
                bin_length = X.shape[-1]
                dataset = TensorDataset(X, Y_bc, Y_clean)
                loader = DataLoader(
                    dataset,
                    batch_size=args.batch_size,
                    shuffle=True,
                    drop_last=False
                )

                model.train()
                file_loss_sum = 0.0
                file_bc_sum = 0.0
                file_clean_sum = 0.0
                file_asym_sum = 0.0
                file_start_time = time.time()

                for batch_x, batch_y_bc, batch_y_clean in loader:
                    batch_x = batch_x.to(device)
                    batch_y_bc = batch_y_bc.to(device)
                    batch_y_clean = batch_y_clean.to(device)

                    optimizer.zero_grad()

                    pred_clean, pred_baseline, pred_bc, _ = model(batch_x)

                    loss_bc = criterion(pred_bc, batch_y_bc)
                    loss_clean = criterion(pred_clean, batch_y_clean)
                    loss_asym = asym_penalty(pred_baseline, batch_x)

                    total_loss = (
                        args.lambda_bc * loss_bc +
                        args.lambda_clean * loss_clean +
                        loss_asym
                    )

                    total_loss.backward()
                    optimizer.step()

                    b_size = batch_x.size(0)
                    file_loss_sum += total_loss.item() * b_size
                    file_bc_sum += loss_bc.item() * b_size
                    file_clean_sum += loss_clean.item() * b_size
                    file_asym_sum += loss_asym.item() * b_size

                file_duration = time.time() - file_start_time
                file_tput = file_samples / file_duration if file_duration > 0 else 0.0

                f_loss = file_loss_sum / file_samples
                f_bc = file_bc_sum / file_samples
                f_clean = file_clean_sum / file_samples
                f_asym = file_asym_sum / file_samples

                epoch_loss_sum += file_loss_sum
                epoch_bc_sum += file_bc_sum
                epoch_clean_sum += file_clean_sum
                epoch_asym_sum += file_asym_sum
                epoch_samples += file_samples

                rel_name = os.path.basename(os.path.dirname(current_filepath)) + "/" + os.path.basename(current_filepath)
                print(
                    f"Epoch [{epoch:3d}/{args.epochs:3d}] | "
                    f"File [{file_idx:3d}/{num_files:3d}] ({rel_name}, L={bin_length:4d}) | "
                    f"Loss: {f_loss:.5f} (BC: {f_bc:.4f}, Clean: {f_clean:.4f}, Asym: {f_asym:.4f}) | "
                    f"{file_tput:6.1f} spec/s"
                )

                # Free chunk memory immediately
                del X, Y_bc, Y_clean, dataset, loader
                gc.collect()
                if device.type == "mps":
                    torch.mps.empty_cache()
                elif device.type == "cuda":
                    torch.cuda.empty_cache()

            # Epoch summary & scheduler step
            epoch_duration = time.time() - epoch_start_time
            avg_loss = epoch_loss_sum / epoch_samples
            avg_bc = epoch_bc_sum / epoch_samples
            avg_clean = epoch_clean_sum / epoch_samples
            avg_asym = epoch_asym_sum / epoch_samples
            epoch_tput = epoch_samples / epoch_duration if epoch_duration > 0 else 0.0

            scheduler.step(avg_loss)
            current_lr = optimizer.param_groups[0]['lr']

            print("-" * 80)
            print(f"==> Epoch [{epoch:3d}/{args.epochs:3d}] Completed in {epoch_duration:.1f}s ({epoch_tput:.1f} spectra/s)")
            print(f"    Avg Total Loss: {avg_loss:.6f} | Avg BC: {avg_bc:.6f} | Avg Clean: {avg_clean:.6f} | Avg Asym: {avg_asym:.6f}")
            print(f"    Learning Rate:  {current_lr:.6e}")

            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.state_dict(), save_model_file)
                print(f"    ★ New best model saved to '{save_model_file}' (Loss: {best_loss:.6f})")

            # Save pause/resume checkpoint
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "best_loss": best_loss,
                "args": vars(args)
            }, checkpoint_file)
            print(f"    Saved checkpoint to '{checkpoint_file}'")
            print("-" * 80)

    except KeyboardInterrupt:
        print("\n" + "=" * 80)
        print(" Training safely paused by user (Ctrl+C).")
        print(f" Checkpoint preserved at: '{checkpoint_file}'")
        print(" Resume anytime by running: python training_round6/training6.py")
        print("=" * 80 + "\n")
        thread_pool.shutdown(wait=False)
        return

    thread_pool.shutdown(wait=True)
    print("\n" + "=" * 80)
    print("TRAINING PROCESS COMPLETED SUCCESSFULLY")
    print(f"Best Loss: {best_loss:.6f}")
    print(f"Model saved to: {save_model_file}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    train()
