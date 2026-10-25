"""
MULTI-SEED HEAD-TO-HEAD TRAINING PIPELINE - ROUND 8
---------------------------------------------------
Models:
  - v3      : PolyGaussNet V3            (reference)
  - v3_1    : PolyGaussNet V3.1          (re-architected)
  - v3_len  : PolyGaussNet V3 + length input in baseline head ONLY (ablation)

Goals:
  1. Re-run V3 vs V3.1 under a NEW seed to check the Round 7 result wasn't a fluke.
  2. Measure how much of the gain comes purely from the length scalar (v3_len vs v3).

Identical conditions for every model (same as Round 7):
  - Dataset: training_round7/training_data7 (reused so results are directly comparable)
  - Epochs 8 | Batch 512 | AdamW lr=1e-4 wd=1e-4 | ReduceLROnPlateau(0.8, patience=1)
  - Loss: Dual Log-Cosh (BC + Clean) + Asymmetric Penalty (lambda=15)

Fairness fix vs Round 7:
  - Batch shuffling uses a DEDICATED torch.Generator seeded from (seed, epoch, file),
    so every model sees the exact same batch order regardless of how much RNG its
    weight initialization consumed.

Outputs:
  - 8-epoch results (pooled across seeds by testing8.py): runs/seed_<SEED>/<model>.pth
  - Longer runs (EPOCHS != 8):                            runs_<EPOCHS>ep/seed_<SEED>/<model>.pth
  - If SNAPSHOT_EPOCH < EPOCHS, the best-of-first-SNAPSHOT_EPOCH weights are ALSO saved
    to runs/seed_<SEED>/ (for SNAPSHOT_EPOCH=8). Training does not depend on the total
    epoch count, so this snapshot is identical to a standalone 8-epoch run with that seed.
  - Safety: the script refuses to start if any target checkpoint already exists.

Usage:
  Set SEED / MODELS / EPOCHS / SNAPSHOT_EPOCH at the top of this file, then:
  python train_round8.py
"""

import os
import sys
import time
import glob
import random
import gc
import shutil

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# =====================================================================
# RUN SETTINGS (edit these)
# =====================================================================
SEED = 202                            # Must be a seed that hasn't been run yet (used: 42, 101)
MODELS = ["v3", "v3_1"]               # Any subset of: "v3", "v3_len", "v3_1"
EPOCHS = 16                           # Total training epochs
SNAPSHOT_EPOCH = 8                    # Also save the epoch-8 result (set to None to disable)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
for path in [PROJECT_ROOT, SCRIPT_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

from polygaussnet3 import PolyGaussNet as PolyGaussNet3
from polygaussnet3_1 import PolyGaussNet as PolyGaussNet3_1
from polygaussnet3_len import PolyGaussNet as PolyGaussNet3_Len

MODEL_REGISTRY = {
    "v3": PolyGaussNet3,
    "v3_1": PolyGaussNet3_1,
    "v3_len": PolyGaussNet3_Len,
}

# Hyperparameters (identical to Round 7; EPOCHS is set in RUN SETTINGS above)
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

DATA_DIR = os.path.join(PROJECT_ROOT, "training_round7", "training_data7")
RUNS_DIR = os.path.join(SCRIPT_DIR, "runs")


def runs_dir_for(epochs: int) -> str:
    """8-epoch results live in runs/ (pooled); other lengths get their own folder."""
    return RUNS_DIR if epochs == 8 else os.path.join(SCRIPT_DIR, f"runs_{epochs}ep")


def snapshot_active() -> bool:
    return SNAPSHOT_EPOCH is not None and 0 < SNAPSHOT_EPOCH < EPOCHS


def target_paths(model_key: str, seed: int) -> list[str]:
    paths = [os.path.join(runs_dir_for(EPOCHS), f"seed_{seed}", f"{model_key}.pth")]
    if snapshot_active():
        paths.append(os.path.join(runs_dir_for(SNAPSHOT_EPOCH), f"seed_{seed}", f"{model_key}.pth"))
    return paths


# =====================================================================
# LOSS FUNCTIONS (identical to Round 7)
# =====================================================================
class LogCoshLoss(nn.Module):
    """Numerically stable Log-Cosh loss function (prevents cosh float32 overflow)."""
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        abs_diff = torch.abs(pred - target)
        return torch.mean(
            abs_diff +
            torch.log1p(torch.exp(-2.0 * abs_diff)) -
            torch.log(torch.tensor(2.0, device=abs_diff.device, dtype=abs_diff.dtype))
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
    return sorted(glob.glob(os.path.join(data_dir, "**", "*.npz"), recursive=True))


def load_file_data(filepath: str):
    with np.load(filepath) as data:
        X = torch.from_numpy(data["full_matrix"]).float().unsqueeze(1)
        Y_bc = torch.from_numpy(data["pure_noise_cosmic_matrix"]).float().unsqueeze(1)
        Y_clean = torch.from_numpy(data["pure_matrix"]).float().unsqueeze(1)
    return X, Y_bc, Y_clean


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_single_model(model_key: str, seed: int, device: torch.device, epochs: int, batch_size: int):
    print("\n" + "=" * 80)
    print(f" ROUND 8 TRAINING: {model_key.upper()} | SEED {seed} ")
    print("=" * 80)

    out_dir = os.path.join(runs_dir_for(epochs), f"seed_{seed}")
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f"{model_key}.pth")
    log_path = os.path.join(out_dir, f"{model_key}_progression.txt")

    snap_save_path = snap_log_path = None
    if snapshot_active():
        snap_dir = os.path.join(runs_dir_for(SNAPSHOT_EPOCH), f"seed_{seed}")
        os.makedirs(snap_dir, exist_ok=True)
        snap_save_path = os.path.join(snap_dir, f"{model_key}.pth")
        snap_log_path = os.path.join(snap_dir, f"{model_key}_progression.txt")

    set_seed(seed)
    model = MODEL_REGISTRY[model_key](poly_order=POLY_ORDER, filter_kernel_size=FILTER_KERNEL_SIZE).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f" Model: {model_key} | Total Params: {total_params:,}")
    print(f" Weights -> {save_path}")
    if snap_save_path:
        print(f" Epoch-{SNAPSHOT_EPOCH} snapshot -> {snap_save_path}")

    criterion_cosh = LogCoshLoss()
    criterion_asym = AsymmetricBaselineLoss(lambda_asym=LAMBDA_ASYM)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=LR_FACTOR, patience=LR_PATIENCE, min_lr=MIN_LEARNING_RATE
    )

    data_files = discover_dataset_files(DATA_DIR)
    if not data_files:
        raise FileNotFoundError(f"No .npz files found in {DATA_DIR}!")
    print(f" Discovered {len(data_files)} dataset chunks.")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"=== TRAINING PROGRESSION: {model_key.upper()} | SEED {seed} ===\n")
        f.write(f"Parameters: {total_params:,}\n")
        f.write(f"Batch Size: {batch_size} | Epochs: {epochs} | Initial LR: {LEARNING_RATE}\n")
        f.write("-" * 90 + "\n")
        f.write(f"{'Epoch':<6} | {'LR':<9} | {'Total Loss':<12} | {'Clean Loss':<12} | {'BC Loss':<12} | {'Asym Loss':<12} | {'Time (s)':<8}\n")
        f.write("-" * 90 + "\n")

    best_loss = float("inf")
    snap_best_loss = float("inf")
    start_total_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()

        # Deterministic file order (independent of global RNG)
        shuffled_files = list(data_files)
        random.Random(seed * 1000 + epoch).shuffle(shuffled_files)

        total_loss_accum = clean_loss_accum = bc_loss_accum = asym_loss_accum = 0.0
        total_samples = 0
        current_lr = optimizer.param_groups[0]["lr"]

        for file_idx, filepath in enumerate(shuffled_files, 1):
            file_start = time.time()
            res_name = os.path.basename(os.path.dirname(filepath))
            X_tensor, Y_bc_tensor, Y_clean_tensor = load_file_data(filepath)
            n_samples = X_tensor.shape[0]

            # Dedicated shuffle generator -> identical batch order for every model
            g = torch.Generator()
            g.manual_seed(seed * 1_000_000 + epoch * 1_000 + int(res_name))
            dataset = TensorDataset(X_tensor, Y_bc_tensor, Y_clean_tensor)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False, generator=g)

            file_loss = file_clean = file_bc = file_asym = 0.0

            for b_idx, (b_x, b_y_bc, b_y_clean) in enumerate(loader):
                b_x, b_y_bc, b_y_clean = b_x.to(device), b_y_bc.to(device), b_y_clean.to(device)
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

                lv, lc, lb, la = loss.item(), loss_clean.item(), loss_bc.item(), loss_asym.item()
                total_loss_accum += lv * bsz
                clean_loss_accum += lc * bsz
                bc_loss_accum += lb * bsz
                asym_loss_accum += la * bsz
                total_samples += bsz
                file_loss += lv * bsz
                file_clean += lc * bsz
                file_bc += lb * bsz
                file_asym += la * bsz

            n = max(1, n_samples)
            print(f"  [Epoch {epoch:2d}/{epochs:2d}] File {file_idx}/{len(shuffled_files)} ({res_name:>4} bins) | "
                  f"Loss: {file_loss/n:.5f} (Clean: {file_clean/n:.4f}, BC: {file_bc/n:.4f}, Asym: {file_asym/n:.4f}) | "
                  f"{time.time() - file_start:.1f}s")

            del X_tensor, Y_bc_tensor, Y_clean_tensor, dataset, loader
            gc.collect()
            if device.type == "mps":
                torch.mps.empty_cache()
            elif device.type == "cuda":
                torch.cuda.empty_cache()

        ts = max(1, total_samples)
        epoch_loss = total_loss_accum / ts
        epoch_clean = clean_loss_accum / ts
        epoch_bc = bc_loss_accum / ts
        epoch_asym = asym_loss_accum / ts
        epoch_time = time.time() - epoch_start

        scheduler.step(epoch_loss)

        print(f"==> Epoch {epoch:2d}/{epochs:2d} | LR: {current_lr:.2e} | Loss: {epoch_loss:.6f} "
              f"(Clean: {epoch_clean:.6f}, BC: {epoch_bc:.6f}, Asym: {epoch_asym:.6f}) | {epoch_time:.2f}s")

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{epoch:<6d} | {current_lr:<9.2e} | {epoch_loss:<12.6f} | {epoch_clean:<12.6f} | "
                    f"{epoch_bc:<12.6f} | {epoch_asym:<12.6f} | {epoch_time:<8.2f}\n")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), save_path)

        # Snapshot: same selection rule (best training loss) restricted to the first SNAPSHOT_EPOCH epochs
        if snap_save_path and epoch <= SNAPSHOT_EPOCH:
            if epoch_loss < snap_best_loss:
                snap_best_loss = epoch_loss
                torch.save(model.state_dict(), snap_save_path)
            if epoch == SNAPSHOT_EPOCH:
                shutil.copyfile(log_path, snap_log_path)
                with open(snap_log_path, "a", encoding="utf-8") as f:
                    f.write("-" * 90 + "\n")
                    f.write(f"Epoch-{SNAPSHOT_EPOCH} snapshot of a {epochs}-epoch run | Best Loss: {snap_best_loss:.6f}\n")
                    f.write(f"Snapshot Weights Saved To: {snap_save_path}\n")
                print(f" Saved epoch-{SNAPSHOT_EPOCH} snapshot -> {snap_save_path}")

    total_duration = time.time() - start_total_time
    print(f"\n Finished {model_key.upper()} (seed {seed}) in {total_duration:.2f}s. Best Loss: {best_loss:.6f}")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write("-" * 90 + "\n")
        f.write(f"Total Duration: {total_duration:.2f}s | Best Loss: {best_loss:.6f}\n")
        f.write(f"Final Weights Saved To: {save_path}\n")


def main():
    unknown = [m for m in MODELS if m not in MODEL_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown model(s) in MODELS: {unknown}. Choose from {list(MODEL_REGISTRY)}")

    # Safety guard: never overwrite existing results
    existing = [p for key in MODELS for p in target_paths(key, SEED) if os.path.exists(p)]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing checkpoints:\n  " + "\n  ".join(existing) +
            "\nChange SEED at the top of train_round8.py (or move/delete these files intentionally)."
        )

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using Compute Device: {device} | Seed: {SEED} | Models: {MODELS} | Epochs: {EPOCHS}"
          + (f" (+ epoch-{SNAPSHOT_EPOCH} snapshot)" if snapshot_active() else ""))

    for key in MODELS:
        train_single_model(key, seed=SEED, device=device, epochs=EPOCHS, batch_size=BATCH_SIZE)

    print("\n" + "=" * 80)
    print(f" ROUND 8 TRAINING COMPLETE (seed {SEED}) ")
    print("=" * 80)


if __name__ == "__main__":
    main()
