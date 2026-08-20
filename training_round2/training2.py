"""
TRAINING SCRIPT - ROUND 2 (Local & Multi-Hardware Optimized)
------------------------------------------------------------
Optimized for local computers (Apple Silicon Mac via MPS, local CUDA GPUs, and CPU)
as well as scalable cluster execution.

FEATURES:
- Automatic hardware acceleration (Apple Silicon MPS / NVIDIA CUDA / CPU)
- Easy-to-edit configuration variables at the top of the file
- Memory-efficient streaming of .npz dataset chunks
- Live progress metrics (spectra/sec throughput, loss breakdown, ETA)
- Safe pause-and-resume on Ctrl+C (saves to `training_checkpoint2.pth`)

USAGE:
  # 1. Simply edit the CONFIGURATION block below and run:
  python training2.py

  # 2. Or optionally override parameters via command-line arguments:
  python training2.py --epochs 50 --lr 0.0002
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

# Configure paths so imports resolve whether running from project root or inside training_round2
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
for path in [PROJECT_ROOT, SCRIPT_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

# Import the existing network model and base loss
from dual_supervised_resnet import DualSupervisedNet, LogCoshLoss


# =====================================================================
# TRAINING CONFIGURATION & HYPERPARAMETERS
# =====================================================================
EPOCHS = 140                 # Total number of training epochs
BATCH_SIZE = None            # Batch size (None = auto: 128 for MPS/Mac, 256 for CUDA, 64 for CPU)
LEARNING_RATE = 0.0005       # Initial learning rate
DEVICE = "auto"              # Hardware: "auto", "mps" (Mac GPU), "cuda" (NVIDIA GPU), "cpu"
DATA_DIR = None              # Path to dataset folder (None = auto-detects training_spectra2)
MAX_FILES = None             # Limit to first N dataset files for fast tests (None = use all files)
CHECKPOINT_PATH = "training_checkpoint2.pth"   # Checkpoint file for pause/resume
SAVE_MODEL_PATH = "dual_supervised_resnet2.pth" # Final trained model weights file
LOG_INTERVAL = 64            # Print batch progress every N batches
NO_RESUME = False            # True = ignore existing checkpoint and start from scratch

# Loss function weighting factors
LAMBDA_BC = 1.5              # Scaling factor for baseline-corrected latent loss
LAMBDA_CLEAN = 1.0           # Scaling factor for final denoised/clean output loss
LAMBDA_L2 = 1e-4             # L2 penalty on clean output to prevent peak hallucination
OVERPRED_PENALTY = 2.0       # Penalty multiplier for over-predicting peak heights


def parse_args():
    """Parse command-line arguments, defaulting to the variables defined above."""
    parser = argparse.ArgumentParser(
        description="Train DualSupervisedNet for Raman spectral preprocessing on local or remote hardware.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS, 
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, 
                        help="Batch size (defaults: 128 for MPS/CPU, 256 for CUDA)")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, 
                        help="Initial learning rate")
    parser.add_argument("--device", type=str, default=DEVICE, choices=["auto", "mps", "cuda", "cpu"],
                        help="Hardware device to train on")
    parser.add_argument("--data-dir", type=str, default=DATA_DIR, 
                        help="Directory containing .npz dataset files (default: training_spectra2)")
    parser.add_argument("--max-files", type=int, default=MAX_FILES, 
                        help="Limit training to the first N dataset files (useful for quick local tests)")
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT_PATH, 
                        help="Checkpoint filename to save/resume training state")
    parser.add_argument("--save-model", type=str, default=SAVE_MODEL_PATH, 
                        help="Final trained model weights output filename")
    parser.add_argument("--no-resume", action="store_true", default=NO_RESUME,
                        help="Ignore existing checkpoint and start training from scratch")
    parser.add_argument("--log-interval", type=int, default=LOG_INTERVAL, 
                        help="Print progress every N batches")
    return parser.parse_args()


def get_device(requested_device="auto"):
    """Detect and configure the optimal computing device."""
    if requested_device == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
    else:
        if requested_device == "mps" and not torch.backends.mps.is_available():
            print("Warning: MPS requested but not available. Falling back to CPU.")
            device = torch.device("cpu")
        elif requested_device == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available. Falling back to CPU.")
            device = torch.device("cpu")
        else:
            device = torch.device(requested_device)

    # Print device details and apply device-specific performance optimizations
    if device.type == "mps":
        print(f"Hardware Acceleration: Apple Silicon GPU (MPS)")
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    elif device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        print(f"Hardware Acceleration: NVIDIA CUDA GPU ({gpu_name})")
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    else:
        num_threads = torch.get_num_threads()
        print(f"Hardware Acceleration: CPU ({num_threads} threads)")

    return device


def load_file_data(filepath):
    """
    Helper function to load data from disk into PyTorch tensors.
    Performs direct float32 conversion to minimize memory allocations.
    """
    with np.load(filepath) as data:
        full = np.ascontiguousarray(data["full_matrix"], dtype=np.float32)
        bc = np.ascontiguousarray(data["pure_noise_cosmic_matrix"], dtype=np.float32)
        clean = np.ascontiguousarray(data["pure_matrix"], dtype=np.float32)

    X = torch.from_numpy(full).unsqueeze(1)
    Y_bc = torch.from_numpy(bc).unsqueeze(1)
    Y_clean = torch.from_numpy(clean).unsqueeze(1)
    return X, Y_bc, Y_clean


# =====================================================================
# LOSS FUNCTION WITH ASYMMETRIC PENALTY (PREVENTS PEAK HALLUCINATION)
# =====================================================================
class AsymmetricLogCoshLoss(nn.Module):
    """
    Log-Cosh Loss with an asymmetric multiplier to penalize over-predictions 
    (y_pred > y_true, i.e., hallucinating peaks/intensity) more heavily than under-predictions.
    """
    def __init__(self, overpred_penalty=2.0):
        super(AsymmetricLogCoshLoss, self).__init__()
        self.penalty = overpred_penalty

    def forward(self, y_pred, y_true):
        x = y_pred - y_true
        # Numerically stable Log-Cosh pointwise loss
        pointwise_loss = (
            torch.abs(x) +
            torch.log1p(torch.exp(-2.0 * torch.abs(x))) -
            torch.log(torch.tensor(2.0, device=x.device))
        )
        # Apply asymmetric penalty when prediction > ground truth
        weights = torch.where(x > 0, self.penalty, 1.0)
        return torch.mean(weights * pointwise_loss)


def train():
    args = parse_args()

    # -----------------------------------------------------------------
    # 1. SETUP DEVICE & HYPERPARAMETERS
    # -----------------------------------------------------------------
    device = get_device(args.device)

    # Set batch size according to device if not explicitly provided
    if args.batch_size is not None:
        batch_size = args.batch_size
    else:
        if device.type == "cuda":
            batch_size = 256
        elif device.type == "mps":
            batch_size = 128
        else:
            batch_size = 64

    learning_rate = args.lr
    epochs = args.epochs

    # Loss weighting factors from config
    lambda_bc = LAMBDA_BC
    lambda_clean = LAMBDA_CLEAN
    lambda_l2 = LAMBDA_L2
    overpred_penalty = OVERPRED_PENALTY

    print(f"\nConfiguration:")
    print(f"  - Device:        {device}")
    print(f"  - Batch Size:    {batch_size}")
    print(f"  - Learning Rate: {learning_rate}")
    print(f"  - Epochs:        {epochs}")
    print(f"  - Checkpoint:    {args.checkpoint}")
    print(f"  - Final Model:   {args.save_model}\n")

    # -----------------------------------------------------------------
    # 2. FIND DATASET FILES
    # -----------------------------------------------------------------
    if args.data_dir:
        data_dir = args.data_dir
        if not os.path.isdir(data_dir):
            raise FileNotFoundError(f"Specified dataset folder does not exist: {data_dir}")
        file_list = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    else:
        # Search candidate directories with priority for training_spectra2
        candidate_dirs = [
            os.path.join(SCRIPT_DIR, "training_spectra2"),
            os.path.join(PROJECT_ROOT, "training_round2", "training_spectra2"),
            "training_spectra2",
            os.path.join(SCRIPT_DIR, "training_spectra"),
            "training_spectra",
        ]
        # Also check any wildcard matches
        candidate_dirs.extend(glob.glob(os.path.join(SCRIPT_DIR, "training_spectra*")))
        candidate_dirs.extend(glob.glob(os.path.join(SCRIPT_DIR, "generated_spectra*")))
        candidate_dirs.extend(glob.glob("training_spectra*"))
        candidate_dirs.extend(glob.glob("generated_spectra*"))

        # Filter existing directories while preserving order
        valid_dirs = [d for d in dict.fromkeys(candidate_dirs) if os.path.isdir(d)]

        if valid_dirs:
            data_dir = valid_dirs[0]
            print(f"Loading training spectra from: {data_dir}")
            file_list = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        else:
            data_dir = "."
            file_list = sorted(glob.glob("dataset_part_*.npz"))

    if not file_list:
        raise FileNotFoundError(f"No .npz dataset files found in '{data_dir}'. Please verify dataset folder.")

    if args.max_files is not None and args.max_files > 0:
        file_list = file_list[:args.max_files]
        print(f"Dataset limited by MAX_FILES / --max-files to {len(file_list)} file(s) from: {data_dir}")
    else:
        print(f"Loaded {len(file_list)} dataset file(s) from: {data_dir}")

    # Determine input spectrum length from first file
    sample_data = np.load(file_list[0])
    input_length = sample_data["full_matrix"].shape[1]
    sample_count_per_file = sample_data["full_matrix"].shape[0]
    sample_data.close()
    total_spectra = sample_count_per_file * len(file_list)
    print(f"Spectral length: {input_length} points | Total dataset size: {total_spectra:,} spectra\n")

    # -----------------------------------------------------------------
    # 3. INITIALIZE MODEL, LOSS, OPTIMIZER, SCHEDULER
    # -----------------------------------------------------------------
    model = DualSupervisedNet(input_length=input_length).to(device)

    criterion_bc = LogCoshLoss()
    criterion_clean = AsymmetricLogCoshLoss(overpred_penalty=overpred_penalty)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(epochs * 1.5), eta_min=1e-5)

    # Automatic Mixed Precision (AMP) configuration
    use_amp = (device.type == "cuda")
    if use_amp:
        amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        print(f"Automatic Mixed Precision (AMP) enabled (dtype: {amp_dtype})")
        scaler = torch.amp.GradScaler('cuda', enabled=(amp_dtype == torch.float16))
    else:
        scaler = None

    # -----------------------------------------------------------------
    # 3.5 CHECKPOINT RESUME
    # -----------------------------------------------------------------
    start_epoch = 0
    checkpoint_path = args.checkpoint
    if not os.path.isabs(checkpoint_path):
        if os.path.exists(os.path.join(SCRIPT_DIR, checkpoint_path)):
            checkpoint_path = os.path.join(SCRIPT_DIR, checkpoint_path)

    if not args.no_resume and os.path.exists(checkpoint_path):
        print(f"Found checkpoint at '{checkpoint_path}'. Resuming training...")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint['model_state_dict']
        try:
            model.load_state_dict(state_dict)
        except RuntimeError:
            raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            cleaned_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
            raw_model.load_state_dict(cleaned_state_dict)

        try:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        except Exception as e:
            print(f"Note: Could not restore optimizer/scheduler state ({e}). Continuing with initialized state.")

        # If user explicitly updated the learning rate for the resume run, apply it
        for param_group in optimizer.param_groups:
            if param_group['lr'] != learning_rate:
                print(f"Updating optimizer learning rate to: {learning_rate}")
                param_group['lr'] = learning_rate

        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"Successfully resumed from epoch {start_epoch}.\n")
    else:
        if args.no_resume:
            print("(no_resume=True): Starting training from scratch.\n")
        else:
            print("No checkpoint found. Starting training from scratch.\n")

    if start_epoch >= epochs:
        print(f"Model is already trained for {start_epoch} epochs (target: {epochs}).")
        print(f"To train further, set EPOCHS / --epochs to a value greater than {start_epoch}.")
        return

    # -----------------------------------------------------------------
    # 4. TRAINING LOOP
    # -----------------------------------------------------------------
    print("=" * 78)
    print(f"Starting Training: Epochs {start_epoch + 1} to {epochs}")
    print("=" * 78)

    # Asynchronous single-worker loader for next file prefetching
    executor = ThreadPoolExecutor(max_workers=1)

    try:
        for epoch in range(start_epoch, epochs):
            epoch_start_time = time.time()
            model.train()
            running_loss = 0.0
            running_bc_loss = 0.0
            running_clean_loss = 0.0
            total_batches = 0
            total_samples_processed = 0

            # Shuffle file order each epoch
            epoch_files = list(file_list)
            np.random.shuffle(epoch_files)

            # Start prefetching the first file
            future = executor.submit(load_file_data, epoch_files[0])

            for file_idx in range(len(epoch_files)):
                # Wait for current file to be ready in RAM
                X, Y_bc, Y_clean = future.result()

                # Start prefetching next file in background immediately
                if file_idx + 1 < len(epoch_files):
                    future = executor.submit(load_file_data, epoch_files[file_idx + 1])

                dataset = TensorDataset(X, Y_bc, Y_clean)
                loader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    pin_memory=(device.type == "cuda"),
                    drop_last=False
                )

                file_start_time = time.time()
                file_batches = len(loader)

                for batch_idx, (batch_x, batch_y_bc, batch_y_clean) in enumerate(loader):
                    batch_x = batch_x.to(device, non_blocking=(device.type == "cuda"))
                    batch_y_bc = batch_y_bc.to(device, non_blocking=(device.type == "cuda"))
                    batch_y_clean = batch_y_clean.to(device, non_blocking=(device.type == "cuda"))

                    optimizer.zero_grad(set_to_none=True)

                    if use_amp:
                        with torch.amp.autocast(device_type="cuda", dtype=amp_dtype):
                            pred_bc, pred_clean = model(batch_x)
                            loss_bc = criterion_bc(pred_bc, batch_y_bc)
                            loss_clean = criterion_clean(pred_clean, batch_y_clean)
                            loss_l2 = torch.mean(pred_clean ** 2)
                            loss = (lambda_bc * loss_bc) + (lambda_clean * loss_clean) + (lambda_l2 * loss_l2)
                    else:
                        pred_bc, pred_clean = model(batch_x)
                        loss_bc = criterion_bc(pred_bc, batch_y_bc)
                        loss_clean = criterion_clean(pred_clean, batch_y_clean)
                        loss_l2 = torch.mean(pred_clean ** 2)
                        loss = (lambda_bc * loss_bc) + (lambda_clean * loss_clean) + (lambda_l2 * loss_l2)

                    if scaler is not None and scaler.is_enabled():
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()

                    batch_samples = batch_x.size(0)
                    running_loss += loss.item()
                    running_bc_loss += loss_bc.item()
                    running_clean_loss += loss_clean.item()
                    total_batches += 1
                    total_samples_processed += batch_samples

                    # Periodic step logging
                    if (batch_idx + 1) % args.log_interval == 0 or (batch_idx + 1) == file_batches:
                        elapsed_file = time.time() - file_start_time
                        samples_in_file = (batch_idx + 1) * batch_size
                        speed = samples_in_file / elapsed_file if elapsed_file > 0 else 0
                        print(
                            f"Epoch [{epoch+1:3d}/{epochs}] | "
                            f"File [{file_idx+1:2d}/{len(epoch_files)}] | "
                            f"Batch [{batch_idx+1:3d}/{file_batches}] | "
                            f"Loss: {loss.item():.5f} "
                            f"(BC: {loss_bc.item():.4f}, Clean: {loss_clean.item():.4f}) | "
                            f"{speed:6.1f} spectra/s"
                        )

                # Clean memory before loading next file
                del X, Y_bc, Y_clean, dataset, loader
                gc.collect()
                if device.type == "mps":
                    torch.mps.empty_cache()
                elif device.type == "cuda":
                    torch.cuda.empty_cache()

            # End of Epoch
            scheduler.step()
            epoch_time = time.time() - epoch_start_time
            avg_loss = running_loss / total_batches if total_batches > 0 else 0
            avg_bc = running_bc_loss / total_batches if total_batches > 0 else 0
            avg_clean = running_clean_loss / total_batches if total_batches > 0 else 0
            current_lr = scheduler.get_last_lr()[0]
            avg_throughput = total_samples_processed / epoch_time if epoch_time > 0 else 0

            print("-" * 78)
            print(
                f"==> Epoch [{epoch+1:3d}/{epochs}] Completed in {epoch_time:.1f}s ({avg_throughput:.1f} spectra/s)\n"
                f"    Avg Total Loss: {avg_loss:.6f} | Avg BC Loss: {avg_bc:.6f} | Avg Clean Loss: {avg_clean:.6f}\n"
                f"    Learning Rate:  {current_lr:.6e}"
            )

            # Save checkpoint
            save_checkpoint_target = checkpoint_path
            if not os.path.isabs(save_checkpoint_target):
                save_checkpoint_target = os.path.join(SCRIPT_DIR, save_checkpoint_target)
            raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            torch.save({
                'epoch': epoch,
                'model_state_dict': raw_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
            }, save_checkpoint_target)
            print(f"    Saved checkpoint to '{save_checkpoint_target}'")
            print("-" * 78)

    except KeyboardInterrupt:
        print("\n" + "!" * 78)
        print("Training safely paused by user (Ctrl+C).")
        print(f"Checkpoint preserved at: '{checkpoint_path}'")
        print(f"Resume anytime by running: python training2.py")
        print("!" * 78 + "\n")
    finally:
        executor.shutdown(wait=False)

    # -----------------------------------------------------------------
    # 5. SAVE FINAL TRAINED MODEL
    # -----------------------------------------------------------------
    save_path = args.save_model
    if not os.path.isabs(save_path):
        save_path = os.path.join(SCRIPT_DIR, save_path)
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    torch.save(raw_model.state_dict(), save_path)
    print(f"\nFinal model weights successfully saved to '{save_path}'")


if __name__ == "__main__":
    train()
