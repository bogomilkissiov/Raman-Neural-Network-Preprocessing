"""
ASYMMETRIC PENALTY COMPARISON TRAINING SCRIPT - ROUND 5
-------------------------------------------------------
Sequentially trains PolyGaussNet3 under Dual Supervised Loss with two different
asymmetric baseline penalty weights for 16 epochs each:
  1. Normal Asymmetric Penalty: lambda_asym = 15.0
       L_total = LogCosh(BC) + LogCosh(Clean) + 15.0 * AsymmetricBaselinePenalty
  2. High Asymmetric Penalty:   lambda_asym = 60.0 (4x scaling)
       L_total = LogCosh(BC) + LogCosh(Clean) + 60.0 * AsymmetricBaselinePenalty

Key Highlights:
  - Architecture held constant: PolyGaussNet3 (Degree 7 Baseline + Bounded Amplitude [0.85, 1.15])
  - Reaches directly into `training_round4/training_data4` without copying or moving files.
  - Phase 1: Trains Normal (lambda=15) -> saves best weights to `training_round5/asymcomp_normal_lambda15.pth`
  - Phase 2: Trains High   (lambda=60) -> saves best weights to `training_round5/asymcomp_high_lambda60.pth`
  - Prefetched asynchronous data streaming via ThreadPoolExecutor.
  - Generates a side-by-side training comparison table upon completion.
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
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# =====================================================================
# PATH RESOLUTION
# =====================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

for p in [PROJECT_ROOT, SCRIPT_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

from polygaussnet3 import PolyGaussNet

# Default paths
DEFAULT_DATA_DIR = os.path.join(PROJECT_ROOT, "training_round4", "training_data4")
DEFAULT_SAVE_NORMAL = os.path.join(SCRIPT_DIR, "asymcomp_normal_lambda15.pth")
DEFAULT_SAVE_HIGH = os.path.join(SCRIPT_DIR, "asymcomp_high_lambda60.pth")

# =====================================================================
# HYPERPARAMETERS (Identical to Round 4 / 5)
# =====================================================================
EPOCHS = 16                           # 16 epochs per model
BATCH_SIZE = 512                      # Batch size
LEARNING_RATE = 1e-4                  # Initial learning rate for AdamW
MIN_LEARNING_RATE = 9e-8              # Minimum learning rate for scheduler
WEIGHT_DECAY = 1e-4                   # Weight decay for AdamW

# Model Architecture Hyperparameters
POLY_ORDER = 7                        # Degree 7 Polynomial Baseline Head
FILTER_KERNEL_SIZE = 31               # Gaussian Filter Kernel Size
MIN_AMPLITUDE = 0.85                  # Bounded Amplitude Minimum
MAX_AMPLITUDE = 1.15                  # Bounded Amplitude Maximum

# Asymmetric Penalty Scales to Compare
LAMBDA_NORMAL = 15.0                  # Normal Asymmetric Penalty Weight
LAMBDA_HIGH = 60.0                    # High Asymmetric Penalty Weight (4x)


# =====================================================================
# CLI ARGUMENT PARSER
# =====================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Asymmetric Loss Comparison: Normal (λ=15) vs. High (λ=60) on PolyGaussNet3",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help="Number of epochs per model")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE,
                        help="Initial learning rate")
    parser.add_argument("--min-lr", type=float, default=MIN_LEARNING_RATE,
                        help="Minimum learning rate")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Weight decay for AdamW")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR,
                        help="Path to training data folder containing .npz chunks")
    parser.add_argument("--save-normal", type=str, default=DEFAULT_SAVE_NORMAL,
                        help="Destination for best Normal (λ=15) model weights")
    parser.add_argument("--save-high", type=str, default=DEFAULT_SAVE_HIGH,
                        help="Destination for best High (λ=60) model weights")
    parser.add_argument("--lambda-normal", type=float, default=LAMBDA_NORMAL,
                        help="Weight for Normal asymmetric baseline penalty")
    parser.add_argument("--lambda-high", type=float, default=LAMBDA_HIGH,
                        help="Weight for High asymmetric baseline penalty")
    parser.add_argument("--poly-order", type=int, default=POLY_ORDER,
                        help="Polynomial degree for baseline head")
    parser.add_argument("--filter-kernel-size", type=int, default=FILTER_KERNEL_SIZE,
                        help="Kernel size for adaptive Gaussian filter")
    parser.add_argument("--min-amplitude", type=float, default=MIN_AMPLITUDE,
                        help="Minimum amplitude bound")
    parser.add_argument("--max-amplitude", type=float, default=MAX_AMPLITUDE,
                        help="Maximum amplitude bound")
    return parser.parse_args()


# =====================================================================
# LOSS FUNCTIONS
# =====================================================================
class LogCoshLoss(nn.Module):
    """
    Logarithm of the hyperbolic cosine loss:
        L(y_pred, y_true) = log(cosh(y_pred - y_true))
    Smooth, outlier-robust approximation to L1 (MAE).
    """
    def __init__(self):
        super(LogCoshLoss, self).__init__()

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        x = y_pred - y_true
        return torch.mean(
            torch.abs(x) +
            torch.log1p(torch.exp(-2.0 * torch.abs(x))) -
            torch.log(torch.tensor(2.0, device=x.device, dtype=x.dtype))
        )


class AsymmetricBaselinePenalty(nn.Module):
    """
    Penalizes baseline predictions exceeding the raw input signal (B_pred > X_raw).
    Proportional to lambda_asym * num_violating_bins * squared excess.
    """
    def __init__(self, lambda_asym: float = 15.0):
        super(AsymmetricBaselinePenalty, self).__init__()
        self.lambda_asym = lambda_asym

    def forward(self, pred_baseline: torch.Tensor, raw_spectrum: torch.Tensor) -> torch.Tensor:
        excess = F.relu(pred_baseline - raw_spectrum)
        with torch.no_grad():
            num_violating_bins = (pred_baseline > raw_spectrum).float().sum(dim=-1, keepdim=True)
        loss = self.lambda_asym * torch.mean((excess ** 2) * num_violating_bins)
        return loss


# =====================================================================
# DATA LOADING
# =====================================================================
def load_file_data(filepath: str):
    """Loads an .npz dataset file from disk into PyTorch float tensors."""
    data = np.load(filepath)
    X = torch.tensor(data["full_matrix"], dtype=torch.float32).unsqueeze(1)
    Y_bc = torch.tensor(data["pure_noise_cosmic_matrix"], dtype=torch.float32).unsqueeze(1)
    Y_clean = torch.tensor(data["pure_matrix"], dtype=torch.float32).unsqueeze(1)
    data.close()
    return X, Y_bc, Y_clean


# =====================================================================
# TRAINING ROUTINE FOR A SPECIFIC ASYMMETRIC SCALE
# =====================================================================
def train_asym_regime(
    regime_name: str,
    lambda_asym: float,
    model: nn.Module,
    file_list: list,
    save_path: str,
    device: torch.device,
    args
):
    print("\n" + "=" * 82)
    print(f" TRAINING: {regime_name} ")
    print(f" Formulation: Dual Supervised [LogCosh(BC) + LogCosh(Clean) + {lambda_asym:.1f} * Asym] ")
    print("=" * 82)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  - Model Architecture: PolyGaussNet3 (Degree {args.poly_order}, Kernel {args.filter_kernel_size})")
    print(f"  - Total Parameters:   {total_params:,}")
    print(f"  - Device:             {device}")
    print(f"  - Target Epochs:      {args.epochs}")
    print(f"  - Batch Size:         {args.batch_size}")
    print(f"  - Initial LR:         {args.lr}")
    print(f"  - Min LR:             {args.min_lr}")
    print(f"  - Lambda Asymmetric:  {lambda_asym:.1f}")
    print(f"  - Weights Destination:{save_path}")
    print("=" * 82 + "\n")

    criterion = LogCoshLoss().to(device)
    asym_penalty = AsymmetricBaselinePenalty(lambda_asym=lambda_asym).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.8,
        patience=1,
        min_lr=args.min_lr
    )

    best_loss = float('inf')
    best_epoch = 0
    training_start_time = time.time()
    history = []

    executor = ThreadPoolExecutor(max_workers=1)

    for epoch in range(args.epochs):
        epoch_start = time.time()
        model.train()

        running_total_loss = 0.0
        running_bc_loss = 0.0
        running_clean_loss = 0.0
        running_asym_loss = 0.0
        total_samples = 0

        # Shuffle dataset file chunks for randomness across epochs
        epoch_files = list(file_list)
        np.random.shuffle(epoch_files)

        # Asynchronously prefetch first file chunk
        future = executor.submit(load_file_data, epoch_files[0])

        for file_idx in range(len(epoch_files)):
            X, Y_bc, Y_clean = future.result()

            # Prefetch next file chunk
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

            for batch_idx, (batch_x, batch_y_bc, batch_y_clean) in enumerate(loader):
                batch_x = batch_x.to(device)
                batch_y_bc = batch_y_bc.to(device)
                batch_y_clean = batch_y_clean.to(device)

                # Forward pass
                clean_pred, pred_baseline, bc_pred, latent_params = model(batch_x)

                # Dual Supervised Loss + Asymmetric Penalty
                loss_bc = criterion(bc_pred, batch_y_bc)
                loss_clean = criterion(clean_pred, batch_y_clean)
                loss_asym = asym_penalty(pred_baseline, batch_x)
                total_loss = loss_bc + loss_clean + loss_asym

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

            # Per-file progress printout
            file_duration = time.time() - file_start_time
            file_avg_loss = file_total_loss / file_samples if file_samples > 0 else 0
            file_avg_bc = file_bc_loss / file_samples if file_samples > 0 else 0
            file_avg_clean = file_clean_loss / file_samples if file_samples > 0 else 0
            file_avg_asym = file_asym_loss / file_samples if file_samples > 0 else 0
            file_throughput = file_samples / file_duration if file_duration > 0 else 0

            print(
                f"[{regime_name}] Ep [{epoch + 1:2d}/{args.epochs:2d}] | "
                f"File [{file_idx + 1:2d}/{len(epoch_files):2d}] | "
                f"TrainLoss: {file_avg_loss:.5f} (Clean: {file_avg_clean:.4f}, BC: {file_avg_bc:.4f}, Asym: {file_avg_asym:.4f}) | "
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

        # Epoch Summary
        print("-" * 82)
        print(f"==> [{regime_name}] Epoch [{epoch + 1:2d}/{args.epochs:2d}] Finished in {epoch_duration:.1f}s ({throughput:.1f} spec/s)")
        print(f"    Avg Train Loss: {avg_loss:.6f} | Clean: {avg_clean_loss:.6f} | BC: {avg_bc_loss:.6f} | Asym: {avg_asym_loss:.6f}")
        print(f"    Learning Rate:  {current_lr:.6e}")

        # Save Best Model Weights
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_epoch = epoch + 1
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"    ★ New best model for {regime_name} saved to '{save_path}' (Loss: {best_loss:.6f})")

        print("-" * 82)

        history.append({
            'epoch': epoch + 1,
            'total_loss': avg_loss,
            'clean_loss': avg_clean_loss,
            'bc_loss': avg_bc_loss,
            'asym_loss': avg_asym_loss,
            'lr': current_lr,
            'duration': epoch_duration
        })

    total_training_time = time.time() - training_start_time
    print(f"\n[{regime_name}] Training Complete in {total_training_time / 60:.2f} min! Best Loss: {best_loss:.6f} (at Epoch {best_epoch}).\n")

    return {
        'regime_name': regime_name,
        'lambda_asym': lambda_asym,
        'best_loss': best_loss,
        'best_epoch': best_epoch,
        'final_loss': avg_loss,
        'final_clean_loss': avg_clean_loss,
        'total_time': total_training_time,
        'history': history,
        'save_path': save_path
    }


# =====================================================================
# MAIN SEQUENTIAL EXECUTION
# =====================================================================
def main():
    args = parse_args()

    # Hardware device setup
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Resolve Data Directory & Files
    data_dir = args.data_dir
    if not os.path.exists(data_dir):
        alt_path = os.path.join(PROJECT_ROOT, "training_round4", "training_data4")
        if os.path.exists(alt_path):
            data_dir = alt_path

    file_list = sorted(glob.glob(os.path.join(data_dir, "dataset_part_*.npz")))
    if not file_list:
        file_list = sorted(glob.glob(os.path.join(data_dir, "*.npz")))

    if not file_list:
        print(f"\n[ERROR] No dataset .npz files found in '{data_dir}'.")
        print("Please check that training_round4/training_data4 contains dataset_part_*.npz files.")
        return

    print("=" * 84)
    print(" ROUND 5: ASYMMETRIC LOSS SCALE SPLIT TEST (16 Epochs Each) ")
    print(" Normal Scaling (λ=15.0) vs. High Scaling (λ=60.0) on PolyGaussNet3 ")
    print("=" * 84)
    print(f"  - Model Architecture: PolyGaussNet3 (Degree {args.poly_order}, Kernel {args.filter_kernel_size}, Amp [{args.min_amplitude}, {args.max_amplitude}])")
    print(f"  - Loss Formulation:    Dual Supervised [LogCosh(BC) + LogCosh(Clean) + λ_asym * Asym]")
    print(f"  - Dataset Location:    {data_dir}")
    print(f"  - Dataset Files:       {len(file_list)} chunks found")
    print(f"  - Device:              {device}")
    print(f"  - Epochs per Model:    {args.epochs}")
    print(f"  - Batch Size:          {args.batch_size}")
    print("=" * 84 + "\n")

    overall_start_time = time.time()

    # -----------------------------------------------------------------
    # PHASE 1: Train with Normal Asymmetric Scaling (lambda = 15.0)
    # -----------------------------------------------------------------
    model_normal = PolyGaussNet(
        poly_order=args.poly_order,
        filter_kernel_size=args.filter_kernel_size,
        min_amplitude=args.min_amplitude,
        max_amplitude=args.max_amplitude
    ).to(device)

    summary_normal = train_asym_regime(
        regime_name="Normal Asym Penalty (λ=15.0)",
        lambda_asym=args.lambda_normal,
        model=model_normal,
        file_list=file_list,
        save_path=args.save_normal,
        device=device,
        args=args
    )

    del model_normal
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()

    # -----------------------------------------------------------------
    # PHASE 2: Train with High Asymmetric Scaling (lambda = 60.0)
    # -----------------------------------------------------------------
    model_high = PolyGaussNet(
        poly_order=args.poly_order,
        filter_kernel_size=args.filter_kernel_size,
        min_amplitude=args.min_amplitude,
        max_amplitude=args.max_amplitude
    ).to(device)

    summary_high = train_asym_regime(
        regime_name="High Asym Penalty (λ=60.0)",
        lambda_asym=args.lambda_high,
        model=model_high,
        file_list=file_list,
        save_path=args.save_high,
        device=device,
        args=args
    )

    del model_high
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()

    total_all_time = time.time() - overall_start_time

    # -----------------------------------------------------------------
    # FINAL COMPARISON SUMMARY TABLE
    # -----------------------------------------------------------------
    print("\n" + "=" * 84)
    print(" ASYMMETRIC SCALE TRAINING SUMMARY (16 Epochs on PolyGaussNet3) ")
    print("=" * 84)
    print(f"{'Metric / Parameter':<30} | {'Normal Penalty (λ=15.0)':<24} | {'High Penalty (λ=60.0)':<24}")
    print("-" * 84)
    print(f"{'Asymmetric Scaling Weight':<30} | {'15.0':<24} | {'60.0 (4x)':<24}")
    print(f"{'Loss Formulation':<30} | {'Dual Supervised':<24} | {'Dual Supervised':<24}")
    print(f"{'Best Training Loss (↓)':<30} | {summary_normal['best_loss']:<24.6f} | {summary_high['best_loss']:<24.6f}")
    print(f"{'Best Epoch':<30} | {summary_normal['best_epoch']:<24d} | {summary_high['best_epoch']:<24d}")
    print(f"{'Final Clean Loss (Epoch 16)':<30} | {summary_normal['final_clean_loss']:<24.6f} | {summary_high['final_clean_loss']:<24.6f}")
    print(f"{'Training Time (min)':<30} | {summary_normal['total_time'] / 60:<24.2f} | {summary_high['total_time'] / 60:<24.2f}")
    print(f"{'Saved Weights':<30} | {os.path.basename(summary_normal['save_path']):<24} | {os.path.basename(summary_high['save_path']):<24}")
    print("=" * 84)
    print(f"Total Sequential Training Elapsed Time: {total_all_time / 60:.2f} minutes\n")


if __name__ == "__main__":
    main()
