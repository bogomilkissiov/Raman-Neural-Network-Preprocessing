"""
TRAINING SCRIPT - ROUND 3 (PolyGaussNet)
----------------------------------------
Trains the input-size agnostic PolyGaussNet architecture:
  Stage 1: Polynomial Baseline Subtraction (Evaluated over [-1, 1])
  Stage 2: Full-Resolution Dilated Residual CNN predicting per-bin [sigma, amplitude, beta]
  Stage 3: Differentiable Generalized Gaussian Filter Denoising

FEATURES:
- Log-Cosh loss evaluated on both Latent (Baseline-Corrected) and Final (Pure) spectra
- Scheduled learning rate (Cosine Annealing with minimum lr)
- Centralized configuration parameters at the top of the file
- Direct Apple Silicon Mac hardware acceleration (MPS)
- Fast asynchronous dataset streaming
"""

import os
import time
import glob
import gc
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Import PolyGaussNet
from polygaussnet import PolyGaussNet


# =====================================================================
# 1. TRAINING CONFIGURATION & HYPERPARAMETERS
# =====================================================================
EPOCHS = 64                 # Total number of training epochs
BATCH_SIZE = 512             # Batch size for training
LEARNING_RATE = 0.001        # Initial learning rate for AdamW
MIN_LEARNING_RATE = 1e-6     # Minimum learning rate for Cosine Annealing scheduler
WEIGHT_DECAY = 1e-4          # Weight decay for AdamW optimizer

DATA_DIR = "training_spectra3"   # Dataset directory
SAVE_MODEL_PATH = "polygaussnet3.pth"  # Final trained model weights file

# Model Hyperparameters
POLY_ORDER = 15              # Order of the baseline polynomial
FILTER_KERNEL_SIZE = 31      # Odd kernel window size for adaptive filter

# Loss Weighting Factors (Dual Supervision)
LAMBDA_BC = 1.0              # Scaling factor for baseline-corrected latent loss
LAMBDA_CLEAN = 1.0           # Scaling factor for final pure/clean output loss


# =====================================================================
# 2. NUMERICALLY STABLE LOG-COSH LOSS
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
# 3. DATA LOADING HELPER
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
# 4. MAIN TRAINING ROUTINE
# =====================================================================
def train():
    # Direct Apple Silicon Mac device setup
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    print("=" * 78)
    print(" PolyGaussNet Training (Apple Silicon MPS) ")
    print("=" * 78)
    print(f"  - Device:             {device}")
    print(f"  - Batch Size:         {BATCH_SIZE}")
    print(f"  - Initial LR:         {LEARNING_RATE}")
    print(f"  - Min LR:             {MIN_LEARNING_RATE}")
    print(f"  - Total Epochs:       {EPOCHS}")
    print(f"  - Polynomial Order:   {POLY_ORDER}")
    print(f"  - Filter Kernel Size: {FILTER_KERNEL_SIZE}")
    print(f"  - Save Model Path:    {SAVE_MODEL_PATH}")
    print("=" * 78)

    # Locate dataset files
    if not os.path.isdir(DATA_DIR):
        raise FileNotFoundError(f"Could not find dataset directory '{DATA_DIR}'.")

    file_list = sorted(glob.glob(os.path.join(DATA_DIR, "*.npz")))
    if not file_list:
        raise FileNotFoundError(f"No .npz files found in '{DATA_DIR}'.")

    # Read metadata from first file
    sample_data = np.load(file_list[0])
    input_length = sample_data["full_matrix"].shape[1]
    spectra_per_file = sample_data["full_matrix"].shape[0]
    sample_data.close()
    print(f"Loaded {len(file_list)} files | Spectral Resolution: {input_length} bins | Total Spectra: {spectra_per_file * len(file_list):,}\n")

    # Initialize model, loss, optimizer, and scheduler
    model = PolyGaussNet(
        poly_order=POLY_ORDER, 
        filter_kernel_size=FILTER_KERNEL_SIZE
    ).to(device)

    criterion = LogCoshLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=EPOCHS, 
        eta_min=MIN_LEARNING_RATE
    )

    print("=" * 78)
    print(f"Starting Training: Epochs 1 to {EPOCHS}")
    print("=" * 78)

    executor = ThreadPoolExecutor(max_workers=1)
    best_loss = float('inf')

    for epoch in range(EPOCHS):
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
                batch_size=BATCH_SIZE,
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

            # Per-file update printout (printed for each of the 16 dataset files in training_history.txt style)
            file_duration = time.time() - file_start_time
            file_avg_loss = file_total_loss / file_samples
            file_avg_bc = file_bc_loss / file_samples
            file_avg_clean = file_clean_loss / file_samples
            file_throughput = file_samples / file_duration if file_duration > 0 else 0
            print(
                f"Epoch [{epoch + 1:3d}/{EPOCHS}] | File [{file_idx + 1:2d}/{len(epoch_files)}] | "
                f"Batch [{len(loader):3d}/{len(loader)}] | "
                f"Loss: {file_avg_loss:.5f} (BC: {file_avg_bc:.4f}, Clean: {file_avg_clean:.4f}) | {file_throughput:6.1f} spectra/s"
            )

            del X, Y_bc, Y_clean, dataset, loader
            gc.collect()

        epoch_duration = time.time() - epoch_start
        avg_loss = running_total_loss / total_samples
        avg_bc_loss = running_bc_loss / total_samples
        avg_clean_loss = running_clean_loss / total_samples
        throughput = total_samples / epoch_duration

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Epoch completion summary matching training_history.txt
        print("-" * 78)
        print(f"==> Epoch [{epoch + 1:3d}/{EPOCHS}] Completed in {epoch_duration:.1f}s ({throughput:.1f} spectra/s)")
        print(f"    Avg Total Loss: {avg_loss:.6f} | Avg BC Loss: {avg_bc_loss:.6f} | Avg Clean Loss: {avg_clean_loss:.6f}")
        print(f"    Learning Rate:  {current_lr:.6e}")

        # Best model auto-save
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), SAVE_MODEL_PATH)
            print(f"    ★ New best model saved to '{SAVE_MODEL_PATH}' (Loss: {best_loss:.6f})")

        print("-" * 78)

    print(f"\nTraining complete! Best overall loss: {best_loss:.6f}. Model saved to '{SAVE_MODEL_PATH}'.")


if __name__ == "__main__":
    train()
