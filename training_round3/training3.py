"""
TRAINING SCRIPT - ROUND 3 (PolyGaussNet)
----------------------------------------
Trains the input-size agnostic PolyGaussNet architecture:
  Stage 1: Polynomial Baseline Subtraction (Evaluated over [-1, 1])
  Stage 2: Full-Resolution Dilated Residual CNN predicting per-bin [sigma, amplitude, beta]
  Stage 3: Differentiable Generalized Gaussian Filter Denoising

FEATURES:
- Log-Cosh loss evaluated on both Latent (Baseline-Corrected) and Final (Pure) spectra
- Scheduled learning rate (ReduceLROnPlateau with minimum lr)
- Centralized configuration parameters at the top of the file
- Direct Apple Silicon Mac hardware acceleration (MPS) & CUDA / CPU fallback
- Fast asynchronous dataset streaming
- Safe pause-and-resume on Ctrl+C (saves to `training_checkpoint3.pth`)
- Checkpoints automatically saved at the end of every epoch
"""

import os
import sys
import time
import glob
import argparse
import gc
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Configure paths so imports and models resolve whether running from project root or inside training_round3
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
for path in [PROJECT_ROOT, SCRIPT_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Import PolyGaussNet
from polygaussnet import PolyGaussNet


# =====================================================================
# 1. TRAINING CONFIGURATION & HYPERPARAMETERS
# =====================================================================
EPOCHS = 128                     # Total number of training epochs
BATCH_SIZE = 512                 # Batch size for training
LEARNING_RATE = 8e-6             # Initial learning rate for AdamW
MIN_LEARNING_RATE = 1e-7         # Minimum learning rate for scheduler
WEIGHT_DECAY = 1e-4              # Weight decay for AdamW optimizer

DATA_DIR = "training_spectra3"       # Dataset directory
SAVE_MODEL_PATH = "polygaussnet3.pth"  # Final trained model weights file
CHECKPOINT_PATH = "training_checkpoint3.pth"  # Checkpoint file for pause/resume
NO_RESUME = False                # True = ignore existing checkpoint and start from scratch

# Model Hyperparameters
POLY_ORDER = 15                  # Order of the baseline polynomial
FILTER_KERNEL_SIZE = 31          # Odd kernel window size for adaptive filter

# Loss Weighting Factors (Dual Supervision)
LAMBDA_BC = 1.0                  # Scaling factor for baseline-corrected latent loss
LAMBDA_CLEAN = 1.0               # Scaling factor for final pure/clean output loss


# =====================================================================
# 2. CLI ARGUMENT PARSER
# =====================================================================
def parse_args():
    """Parse command-line arguments, defaulting to the variables defined above."""
    parser = argparse.ArgumentParser(
        description="Train PolyGaussNet for Raman spectral preprocessing with pause/resume support.",
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
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Weight decay for AdamW")
    parser.add_argument("--data-dir", type=str, default=DATA_DIR,
                        help="Directory containing .npz dataset files")
    parser.add_argument("--save-model", type=str, default=SAVE_MODEL_PATH,
                        help="Path to save the best model weights")
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT_PATH,
                        help="Checkpoint filename to save/resume training state")
    parser.add_argument("--no-resume", action="store_true", default=NO_RESUME,
                        help="Ignore existing checkpoint and start training from scratch")
    parser.add_argument("--poly-order", type=int, default=POLY_ORDER,
                        help="Order of the baseline polynomial")
    parser.add_argument("--filter-kernel-size", type=int, default=FILTER_KERNEL_SIZE,
                        help="Odd kernel window size for adaptive filter")
    return parser.parse_args()


# =====================================================================
# 3. NUMERICALLY STABLE LOG-COSH LOSS
# =====================================================================
class LogCoshLoss(nn.Module):
    """
    Logarithm of the hyperbolic cosine loss:
        L(y_pred, y_true) = log(cosh(y_pred - y_true))
    Smooth, outlier-robust approximation to L1 (MAE) for stable autograd optimization.
    """
    def __init__(self):
        super(LogCoshLoss, self).__init__()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # Stable formula: log(cosh(x)) = |x| + log1p(exp(-2|x|)) - log(2)
        x = y_pred - y_true
        return torch.mean(
            torch.abs(x) +
            torch.log1p(torch.exp(-2.0 * torch.abs(x))) -
            torch.log(torch.tensor(2.0, device=x.device, dtype=x.dtype))
        )


# =====================================================================
# 4. DATA LOADING HELPER
# =====================================================================
def load_file_data(filepath: str):
    """Load an .npz dataset file from disk into PyTorch float tensors."""
    data = np.load(filepath)
    X = torch.tensor(data["full_matrix"], dtype=torch.float32).unsqueeze(1)
    Y_bc = torch.tensor(data["pure_noise_cosmic_matrix"], dtype=torch.float32).unsqueeze(1)
    Y_clean = torch.tensor(data["pure_matrix"], dtype=torch.float32).unsqueeze(1)
    data.close()
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
    print(" PolyGaussNet Training (Apple Silicon MPS / CUDA / CPU) ")
    print("=" * 78)
    print(f"  - Device:             {device}")
    print(f"  - Batch Size:         {args.batch_size}")
    print(f"  - Initial LR:         {args.lr}")
    print(f"  - Min LR:             {args.min_lr}")
    print(f"  - Total Epochs:       {args.epochs}")
    print(f"  - Polynomial Order:   {args.poly_order}")
    print(f"  - Filter Kernel Size: {args.filter_kernel_size}")
    print(f"  - Checkpoint Path:    {args.checkpoint}")
    print(f"  - Save Model Path:    {args.save_model}")
    print("=" * 78)

    # Locate dataset files (check relative to current dir, SCRIPT_DIR, and PROJECT_ROOT)
    data_dir = args.data_dir
    if not os.path.isdir(data_dir):
        candidate_dir = os.path.join(SCRIPT_DIR, data_dir)
        if os.path.isdir(candidate_dir):
            data_dir = candidate_dir
        else:
            candidate_dir = os.path.join(PROJECT_ROOT, "training_round3", data_dir)
            if os.path.isdir(candidate_dir):
                data_dir = candidate_dir

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Could not find dataset directory '{args.data_dir}'.")

    file_list = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    if not file_list:
        raise FileNotFoundError(f"No .npz files found in '{data_dir}'.")

    # Read metadata from first file
    sample_data = np.load(file_list[0])
    input_length = sample_data["full_matrix"].shape[1]
    spectra_per_file = sample_data["full_matrix"].shape[0]
    sample_data.close()
    print(f"Loaded {len(file_list)} files | Spectral Resolution: {input_length} bins | Total Spectra: {spectra_per_file * len(file_list):,}\n")

    # Initialize model, loss, optimizer, and scheduler
    model = PolyGaussNet(
        poly_order=args.poly_order, 
        filter_kernel_size=args.filter_kernel_size
    ).to(device)

    criterion = LogCoshLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',
        factor=0.5,
        patience=1,
        min_lr=args.min_lr
    )

    # -----------------------------------------------------------------
    # Checkpoint Resume Logic
    # -----------------------------------------------------------------
    start_epoch = 0
    best_loss = float('inf')
    checkpoint_path = args.checkpoint

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

            running_total_loss = 0.0
            running_bc_loss = 0.0
            running_clean_loss = 0.0
            total_samples = 0

            # Shuffle dataset file chunks for randomness across epochs
            epoch_files = list(file_list)
            np.random.shuffle(epoch_files)

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
                file_samples = 0

                for batch_idx, (batch_x, batch_y_bc, batch_y_clean) in enumerate(loader):
                    batch_x = batch_x.to(device)
                    batch_y_bc = batch_y_bc.to(device)
                    batch_y_clean = batch_y_clean.to(device)

                    # Forward pass
                    clean_pred, pred_baseline, bc_pred, latent_params = model(batch_x)

                    # Dual Supervised Log-Cosh Loss
                    loss_bc = criterion(bc_pred, batch_y_bc)
                    loss_clean = criterion(clean_pred, batch_y_clean)
                    total_loss = LAMBDA_BC * loss_bc + LAMBDA_CLEAN * loss_clean

                    # Backward & Step
                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()

                    bs = batch_x.size(0)
                    file_total_loss += total_loss.item() * bs
                    file_bc_loss += loss_bc.item() * bs
                    file_clean_loss += loss_clean.item() * bs
                    file_samples += bs

                    running_total_loss += total_loss.item() * bs
                    running_bc_loss += loss_bc.item() * bs
                    running_clean_loss += loss_clean.item() * bs
                    total_samples += bs

                # Per-file update printout
                file_duration = time.time() - file_start_time
                file_avg_loss = file_total_loss / file_samples if file_samples > 0 else 0
                file_avg_bc = file_bc_loss / file_samples if file_samples > 0 else 0
                file_avg_clean = file_clean_loss / file_samples if file_samples > 0 else 0
                file_throughput = file_samples / file_duration if file_duration > 0 else 0
                print(
                    f"Epoch [{epoch + 1:3d}/{args.epochs}] | File [{file_idx + 1:2d}/{len(epoch_files)}] | "
                    f"Batch [{len(loader):3d}/{len(loader)}] | "
                    f"Loss: {file_avg_loss:.5f} (BC: {file_avg_bc:.4f}, Clean: {file_avg_clean:.4f}) | {file_throughput:6.1f} spectra/s"
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
            throughput = total_samples / epoch_duration if epoch_duration > 0 else 0

            scheduler.step(avg_loss)
            current_lr = optimizer.param_groups[0]['lr']

            # Epoch completion summary
            print("-" * 78)
            print(f"==> Epoch [{epoch + 1:3d}/{args.epochs}] Completed in {epoch_duration:.1f}s ({throughput:.1f} spectra/s)")
            print(f"    Avg Total Loss: {avg_loss:.6f} | Avg BC Loss: {avg_bc_loss:.6f} | Avg Clean Loss: {avg_clean_loss:.6f}")
            print(f"    Learning Rate:  {current_lr:.6e}")

            # Best model auto-save
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.state_dict(), args.save_model)
                print(f"    ★ New best model saved to '{args.save_model}' (Loss: {best_loss:.6f})")

            # Save full training checkpoint for pause/resume
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_loss': best_loss,
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"    Saved checkpoint to '{checkpoint_path}'")
            print("-" * 78)

        print(f"\nTraining complete! Best overall loss: {best_loss:.6f}. Model saved to '{args.save_model}'.")

    except KeyboardInterrupt:
        print("\n" + "=" * 78)
        print(" Training safely paused by user (Ctrl+C).")
        if os.path.exists(checkpoint_path):
            print(f" Checkpoint preserved at: '{checkpoint_path}'")
        else:
            # If interrupted in epoch 0 before epoch 1 completed, save current progress
            torch.save({
                'epoch': max(0, start_epoch - 1),
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_loss': best_loss,
                'loss': avg_loss if 'avg_loss' in locals() else (running_total_loss / total_samples if total_samples > 0 else float('inf')),
            }, checkpoint_path)
            print(f" Checkpoint saved to: '{checkpoint_path}'")
        print(f" Resume anytime by running: python training3.py")
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
