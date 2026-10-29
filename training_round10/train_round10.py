"""
MULTI-SEED HEAD-TO-HEAD TRAINING PIPELINE - ROUND 10
----------------------------------------------------
Models:
  - v3         : PolyGaussNet V3.0       (Original reference benchmark: 128-ch trunk, 1-bucket pool, 128->128->64->8 FC)
  - v3_1       : PolyGaussNet V3.1       (32-ch trunk, 1-bucket pool, 33->128->64->8 FC)
  - v3_2       : PolyGaussNet V3.2       (32-ch trunk + 1x1 proj to 128, 1-bucket pool, 129->64->8 contracting funnel)
  - v3_3       : PolyGaussNet V3.3       (32-ch trunk, 4-quadrant pool, 129->64->8 contracting funnel, WITH length scalar)
  - v3_3_nolen : PolyGaussNet V3.3-NoLen (Ablation: V3.3 with length scalar REMOVED, 128->64->8 contracting funnel)

Goals:
  1. Re-run the Round 9 benchmark (V3.0, V3.1, V3.2, V3.3) on a FRESH dataset across 5 NEW seeds.
  2. Length-scalar ablation: V3.3-NoLen vs V3.3 -> effect of REMOVING the length scalar from the best model.

Identical conditions for every model (same as Round 9):
  - Dataset: training_round10/training_data10 (fresh, Round 7 generation parameters, 40,960 spectra)
  - Epochs 16 | Batch 512 | AdamW lr=1e-4 wd=1e-4 | ReduceLROnPlateau(0.8, patience=1)
  - Loss: Dual Log-Cosh (BC + Clean) + Asymmetric Baseline Penalty (lambda=15)
  - Deterministic batch shuffling: dedicated torch.Generator seeded from (seed, epoch, file)
    ensuring all models see identical data and batch sequences.
  - Checkpoint selection: best TRAINING loss (test set is never touched during training).

Outputs:
  - 16-epoch checkpoints: runs/seed_<SEED>/<model>.pth
  - 8-epoch snapshots:   runs_8ep/seed_<SEED>/<model>.pth (when SNAPSHOT_EPOCH=8)
  - Progression logs:    runs/seed_<SEED>/<model>_progression.txt

Usage:
  1. python create_dataset10.py   (once, generates training_data10/ and testing_data10/)
  2. Set constants at the top of this file, then:
     python train_round10.py
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
# RUN SETTINGS (edit these constants)
# =====================================================================
SEEDS = [300, 301, 302, 303, 304]                                     # 5 fresh seeds
MODELS = ["v3", "v3_1", "v3_2", "v3_3", "v3_3_nolen"]       # Benchmark all 5 models
EPOCHS = 16                               # Total training epochs (16 epochs benchmark)
SNAPSHOT_EPOCH = 8                        # Save epoch-8 checkpoint snapshot (None to disable)
SKIP_EXISTING = True                      # Skip already finished checkpoints (allows safe restart)
OVERWRITE = False                         # Force overwrite existing checkpoints if True

# Hyperparameters (identical to Rounds 7, 8 & 9)
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
for path in [PROJECT_ROOT, SCRIPT_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

from polygaussnet3 import PolyGaussNet as PolyGaussNet3
from polygaussnet3_1 import PolyGaussNet as PolyGaussNet3_1
from polygaussnet3_2 import PolyGaussNet as PolyGaussNet3_2
from polygaussnet3_3 import PolyGaussNet as PolyGaussNet3_3
from polygaussnet3_3_nolen import PolyGaussNet as PolyGaussNet3_3_NoLen

MODEL_REGISTRY = {
    "v3": PolyGaussNet3,
    "v3_1": PolyGaussNet3_1,
    "v3_2": PolyGaussNet3_2,
    "v3_3": PolyGaussNet3_3,
    "v3_3_nolen": PolyGaussNet3_3_NoLen,
}

DATA_DIR = os.path.join(SCRIPT_DIR, "training_data10")
RUNS_DIR = os.path.join(SCRIPT_DIR, "runs")


def runs_dir_for(epochs: int) -> str:
    """Standard 16-epoch results live in runs/; other lengths get their own folder."""
    return RUNS_DIR if epochs == 16 else os.path.join(SCRIPT_DIR, f"runs_{epochs}ep")


def snapshot_active() -> bool:
    return SNAPSHOT_EPOCH is not None and 0 < SNAPSHOT_EPOCH < EPOCHS


def target_paths(model_key: str, seed: int) -> list[str]:
    paths = [os.path.join(runs_dir_for(EPOCHS), f"seed_{seed}", f"{model_key}.pth")]
    if snapshot_active():
        paths.append(os.path.join(runs_dir_for(SNAPSHOT_EPOCH), f"seed_{seed}", f"{model_key}.pth"))
    return paths


# =====================================================================
# LOSS FUNCTIONS (identical to Rounds 7, 8 & 9)
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
    print(f" ROUND 10 TRAINING: {model_key.upper()} | SEED {seed} ")
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
        raise FileNotFoundError(f"No .npz files found in {DATA_DIR}! Run create_dataset10.py first.")
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

    seeds = SEEDS if isinstance(SEEDS, list) else [SEEDS]

    # Safety guard: prevent accidental overwriting if neither OVERWRITE nor SKIP_EXISTING
    if not OVERWRITE and not SKIP_EXISTING:
        existing = [p for seed in seeds for key in MODELS for p in target_paths(key, seed) if os.path.exists(p)]
        if existing:
            raise FileExistsError(
                "Refusing to overwrite existing checkpoints:\n  " + "\n  ".join(existing) +
                "\nSet OVERWRITE = True or SKIP_EXISTING = True at the top of train_round10.py."
            )

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("=" * 80)
    print(f" ROUND 10 MULTI-SEED OVERNIGHT TRAINING PIPELINE ")
    print(f" Device: {device} | Seeds: {seeds} | Models: {MODELS} | Epochs: {EPOCHS}"
          + (f" (+ epoch-{SNAPSHOT_EPOCH} snapshot)" if snapshot_active() else ""))
    print("=" * 80)

    total_start = time.time()
    for s_idx, seed in enumerate(seeds, 1):
        print("\n" + "#" * 80)
        print(f" [{s_idx}/{len(seeds)}] STARTING EXPERIMENT WITH SEED {seed} ")
        print("#" * 80)

        for key in MODELS:
            t_paths = target_paths(key, seed)
            primary_path = t_paths[0]
            if os.path.exists(primary_path) and not OVERWRITE and SKIP_EXISTING:
                print(f"  [Skipping] {key.upper()} (seed {seed}) already trained -> {primary_path}")
                continue

            train_single_model(key, seed=seed, device=device, epochs=EPOCHS, batch_size=BATCH_SIZE)

    elapsed = time.time() - total_start
    print("\n" + "=" * 80)
    print(f" ROUND 10 OVERNIGHT TRAINING COMPLETE! Total Time: {elapsed / 3600.0:.2f} hours ({elapsed:.1f}s) ")
    print("=" * 80)


if __name__ == "__main__":
    main()
