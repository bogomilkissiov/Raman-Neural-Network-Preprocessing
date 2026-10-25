
"""
HEAD-TO-HEAD TRAINING PIPELINE - ROUND 7 (PolyGaussNet V3 vs PolyGaussNet V3.1)
-------------------------------------------------------------------------------
Trains PolyGaussNet3 and PolyGaussNet3.1 under strictly identical conditions:
  - Dataset: training_round7/training_data7 (800, 1016, 1200, 1400, 1600 bins)
  - Epochs: 8
  - Batch Size: 512
  - Optimizer: AdamW (lr=1e-4, min_lr=9e-8, weight_decay=1e-4)
  - Scheduler: ReduceLROnPlateau (factor=0.8, patience=1)
  - Loss Function: Dual Log-Cosh (BC + Clean) + Asymmetric Penalty (lambda=15.0)
  - Seed: 42 (identical weight initialization RNG & batch ordering)

Outputs:
  - Models: polygaussnet3.pth, polygaussnet3_1.pth
  - Progress: training_progression_v3.txt, training_progression_v3_1.txt
-------------------------------------------------------------------------------
"""

import os
import sys
import time
import glob
import random
import argparse
import gc

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Resolve imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
for path in [PROJECT_ROOT, SCRIPT_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

from polygaussnet3 import PolyGaussNet as PolyGaussNet3
from polygaussnet3_1 import PolyGaussNet as PolyGaussNet3_1

# Hyperparameters
EPOCHS = 8
BATCH_SIZE = 512
LEARNING_RATE = 1e-4
MIN_LEARNING_RATE = 9e-8
LR_PATIENCE = 1
LR_FACTOR = 0.8
WEIGHT_DECAY = 1e-4
POLY_ORDER = 7
FILTER_KERNEL_SIZE = 31

LAMBDA_BC = 1.0
LAMBDA_CLEAN = 1.0
LAMBDA_ASYM = 15.0

DATA_DIR = os.path.join(SCRIPT_DIR, "training_data7")


# =====================================================================
# LOSS FUNCTIONS
# =====================================================================
class LogCoshLoss(nn.Module):
    """Numerically stable Log-Cosh loss function (prevents cosh float32 overflow)."""
    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        abs_diff = torch.abs(diff)
        return torch.mean(
            abs_diff +
            torch.log1p(torch.exp(-2.0 * abs_diff)) -
            torch.log(torch.tensor(2.0, device=diff.device, dtype=diff.dtype))
        )


class AsymmetricBaselineLoss(nn.Module):
    def __init__(self, lambda_asym: float = 15.0):
        super().__init__()
        self.lambda_asym = lambda_asym

    def forward(self, pred_baseline: torch.Tensor, raw_spectrum: torch.Tensor) -> torch.Tensor:
        excess = F.relu(pred_baseline - raw_spectrum)
        with torch.no_grad():
            num_violating_bins = (pred_baseline > raw_spectrum).float().sum(dim=-1, keepdim=True)
        return self.lambda_asym * torch.mean((excess ** 2) * num_violating_bins)


def discover_dataset_files(data_dir: str) -> list[str]:
    files = sorted(glob.glob(os.path.join(data_dir, "**", "*.npz"), recursive=True))
    return files


def load_file_data(filepath: str):
    with np.load(filepath) as data:
        X = torch.from_numpy(data["full_matrix"]).float().unsqueeze(1)
        Y_bc = torch.from_numpy(data["pure_noise_cosmic_matrix"]).float().unsqueeze(1)
        Y_clean = torch.from_numpy(data["pure_matrix"]).float().unsqueeze(1)
    return X, Y_bc, Y_clean


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_single_model(model_name: str, device: torch.device, epochs: int = 8, batch_size: int = BATCH_SIZE):
    print("\n" + "=" * 80)
    print(f" STARTING HEAD-TO-HEAD TRAINING: {model_name.upper()} ")
    print("=" * 80)
    
    set_seed(42)
    
    if model_name.lower() in ["v3", "polygaussnet3"]:
        model = PolyGaussNet3(poly_order=POLY_ORDER, filter_kernel_size=FILTER_KERNEL_SIZE).to(device)
        save_path = os.path.join(SCRIPT_DIR, "polygaussnet3.pth")
        log_path = os.path.join(SCRIPT_DIR, "training_progression_v3.txt")
    elif model_name.lower() in ["v3.1", "v3_1", "polygaussnet3_1"]:
        model = PolyGaussNet3_1(poly_order=POLY_ORDER, filter_kernel_size=FILTER_KERNEL_SIZE).to(device)
        save_path = os.path.join(SCRIPT_DIR, "polygaussnet3_1.pth")
        log_path = os.path.join(SCRIPT_DIR, "training_progression_v3_1.txt")
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    total_params = sum(p.numel() for p in model.parameters())
    print(f" Model: {model.__class__.__name__} | Total Params: {total_params:,}")
    print(f" Saving weights to: {save_path}")
    print(f" Saving log to:     {log_path}")

    criterion_cosh = LogCoshLoss()
    criterion_asym = AsymmetricBaselineLoss(lambda_asym=LAMBDA_ASYM)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=LR_FACTOR, patience=LR_PATIENCE, min_lr=MIN_LEARNING_RATE
    )

    data_files = discover_dataset_files(DATA_DIR)
    if not data_files:
        raise FileNotFoundError(f"No .npz files found in {DATA_DIR}!")
    print(f" Discovered {len(data_files)} dataset chunks across all resolutions.")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"=== TRAINING PROGRESSION: {model_name.upper()} ===\n")
        f.write(f"Parameters: {total_params:,}\n")
        f.write(f"Batch Size: {batch_size} | Epochs: {epochs} | Initial LR: {LEARNING_RATE}\n")
        f.write("-" * 90 + "\n")
        f.write(f"{'Epoch':<6} | {'LR':<9} | {'Total Loss':<12} | {'Clean Loss':<12} | {'BC Loss':<12} | {'Asym Loss':<12} | {'Time (s)':<8}\n")
        f.write("-" * 90 + "\n")

    best_loss = float("inf")
    start_total_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        
        # Interleave files deterministically for fair comparison
        rng_epoch = random.Random(42 + epoch)
        shuffled_files = list(data_files)
        rng_epoch.shuffle(shuffled_files)

        total_loss_accum = 0.0
        clean_loss_accum = 0.0
        bc_loss_accum = 0.0
        asym_loss_accum = 0.0
        total_samples = 0

        current_lr = optimizer.param_groups[0]["lr"]

        for file_idx, filepath in enumerate(shuffled_files, 1):
            file_start = time.time()
            res_name = os.path.basename(os.path.dirname(filepath))
            X_tensor, Y_bc_tensor, Y_clean_tensor = load_file_data(filepath)
            n_samples = X_tensor.shape[0]
            
            # Deterministic batch generator per file
            dataset = TensorDataset(X_tensor, Y_bc_tensor, Y_clean_tensor)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

            file_loss_accum = 0.0
            file_clean_accum = 0.0
            file_bc_accum = 0.0
            file_asym_accum = 0.0

            for b_idx, (b_x, b_y_bc, b_y_clean) in enumerate(loader):
                b_x = b_x.to(device)
                b_y_bc = b_y_bc.to(device)
                b_y_clean = b_y_clean.to(device)
                bsz = b_x.shape[0]

                optimizer.zero_grad()
                pred_clean, pred_base, pred_bc, _ = model(b_x)

                loss_clean = criterion_cosh(pred_clean, b_y_clean)
                loss_bc = criterion_cosh(pred_bc, b_y_bc)
                loss_asym = criterion_asym(pred_base, b_x)

                loss = (LAMBDA_CLEAN * loss_clean) + (LAMBDA_BC * loss_bc) + loss_asym

                if torch.isnan(loss) or torch.isinf(loss):
                    raise FloatingPointError(
                        f"Loss became NaN/Inf at Epoch {epoch}, File {filepath}, Batch {b_idx}! "
                        f"Clean={loss_clean.item()}, BC={loss_bc.item()}, Asym={loss_asym.item()}"
                    )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

                loss_val = loss.item()
                lc_val = loss_clean.item()
                lbc_val = loss_bc.item()
                la_val = loss_asym.item()

                total_loss_accum += loss_val * bsz
                clean_loss_accum += lc_val * bsz
                bc_loss_accum += lbc_val * bsz
                asym_loss_accum += la_val * bsz
                total_samples += bsz

                file_loss_accum += loss_val * bsz
                file_clean_accum += lc_val * bsz
                file_bc_accum += lbc_val * bsz
                file_asym_accum += la_val * bsz

            file_elapsed = time.time() - file_start
            f_loss = file_loss_accum / max(1, n_samples)
            f_clean = file_clean_accum / max(1, n_samples)
            f_bc = file_bc_accum / max(1, n_samples)
            f_asym = file_asym_accum / max(1, n_samples)
            print(f"  [Epoch {epoch:2d}/{epochs:2d}] File {file_idx}/{len(shuffled_files)} ({res_name:>4} bins) | "
                  f"Loss: {f_loss:.5f} (Clean: {f_clean:.4f}, BC: {f_bc:.4f}, Asym: {f_asym:.4f}) | {file_elapsed:.1f}s")

            del X_tensor, Y_bc_tensor, Y_clean_tensor, dataset, loader
            gc.collect()
            if device.type == "mps":
                torch.mps.empty_cache()
            elif device.type == "cuda":
                torch.cuda.empty_cache()

        epoch_loss = total_loss_accum / max(1, total_samples)
        epoch_clean = clean_loss_accum / max(1, total_samples)
        epoch_bc = bc_loss_accum / max(1, total_samples)
        epoch_asym = asym_loss_accum / max(1, total_samples)
        epoch_time = time.time() - epoch_start

        scheduler.step(epoch_loss)

        # Console print
        print(f"==> Epoch {epoch:2d}/{epochs:2d} Finished | LR: {current_lr:.2e} | Loss: {epoch_loss:.6f} "
              f"(Clean: {epoch_clean:.6f}, BC: {epoch_bc:.6f}, Asym: {epoch_asym:.6f}) | Time: {epoch_time:.2f}s")

        # Log write
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{epoch:<6d} | {current_lr:<9.2e} | {epoch_loss:<12.6f} | {epoch_clean:<12.6f} | {epoch_bc:<12.6f} | {epoch_asym:<12.6f} | {epoch_time:<8.2f}\n")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), save_path)

    total_duration = time.time() - start_total_time
    print(f"\n Finished training {model_name.upper()} in {total_duration:.2f}s. Best Loss: {best_loss:.6f}")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("-" * 90 + "\n")
        f.write(f"Total Duration: {total_duration:.2f}s | Best Loss: {best_loss:.6f}\n")
        f.write(f"Final Weights Saved To: {save_path}\n")

    return save_path, log_path


def main():
    parser = argparse.ArgumentParser(description="Head-to-head training: PolyGaussNet V3 vs V3.1")
    parser.add_argument("--model", type=str, default="both", choices=["v3", "v3_1", "both"],
                        help="Which model to train: v3, v3_1, or both")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    args = parser.parse_args()

    # Hardware device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using Compute Device: {device}")

    if args.model in ["v3", "both"]:
        train_single_model("v3", device=device, epochs=args.epochs, batch_size=args.batch_size)

    if args.model in ["v3_1", "both"]:
        train_single_model("v3_1", device=device, epochs=args.epochs, batch_size=args.batch_size)

    print("\n" + "=" * 80)
    print(" HEAD-TO-HEAD TRAINING RUN COMPLETED SUCCESSFULLY! ")
    print("=" * 80)


if __name__ == "__main__":
    main()
