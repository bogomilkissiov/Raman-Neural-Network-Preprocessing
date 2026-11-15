"""
TRAINING SCRIPT - ROUND 12 (PolyGaussNet V3.4 8-Section on Round 6 Regime & Dataset)
-------------------------------------------------------------------------------------
Trains PolyGaussNet V3.4 (8-section spatial pooling, `num_quadrants=8`, 108,394 params)
for 97 epochs using the EXACT Round 6 training regime and dataset (`training_round6/training_data6`),
optimized for both Brown's OSCAR GPU cluster (CUDA / Slurm) and local Apple Silicon (MPS).

ROUND 6 EXACT TRAINING REGIME (preserved faithfully):
- Dataset:             training_round6/training_data6 (40 chunks, 655,360 spectra, L in {800,1016,1200,1400,1600})
- Epochs:              97 (exact stopping epoch of Round 6's `polygaussnet6.pth`)
- Architecture:        PolyGaussNet V3.4 (`num_quadrants=8`, baseline_ch=32, filter_ch=32,
                       poly_order=7, filter_kernel_size=31, amplitude in [0.85, 1.15])
- Batch Size:          512 (interleaved chunk shuffling across resolutions each epoch)
- Optimizer:           AdamW (lr=1e-4, weight_decay=1e-4)
- LR Scheduler:        ReduceLROnPlateau (mode='min', factor=0.8, patience=1, min_lr=9e-8)
- Loss:                1.0 * LogCosh(pred_bc, y_bc)
                     + 1.0 * LogCosh(pred_clean, y_clean)
                     + AsymmetricBaselinePenalty(pred_baseline, raw_spectrum, lambda_asym=15.0)

OSCAR CLUSTER OPTIMIZATIONS:
- One-time dataset preloading (--data-cache auto|gpu|ram|stream) eliminates 3,880+ repeated
  .npz zip decompressions across 97 epochs on shared Lustre/VAST storage.
- Direct GPU VRAM caching (when VRAM headroom permits, e.g. A100/H100/L40s/A6000/RTX3090)
  or pinned CPU RAM with async non_blocking H2D transfers.
- Vectorized `torch.randperm` tensor slicing replaces slow Python `DataLoader` collation.
- Eliminates per-chunk `gc.collect()` and `empty_cache()` host-device synchronization stalls.
- Enables TF32 matmul/cuDNN on Ampere+ GPUs + optional `--amp` (BF16) and `--compile`.
- Atomic checkpoint writes (.tmp -> os.replace) + Slurm SIGTERM/SIGINT graceful handling
  so `--signal=B:TERM@60` checkpoints cleanly and resumes deterministically via `--resume`.
"""

import argparse
import glob
import os
import random
import signal
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Configure paths so imports resolve cleanly from project root or training_round12/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
for path in (PROJECT_ROOT, SCRIPT_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from polygaussnet3_4 import PolyGaussNet

# =====================================================================
# 1. EXACT ROUND 6 HYPERPARAMETERS
# =====================================================================
EPOCHS = 97                           # Round 6 best checkpoint epoch
BATCH_SIZE = 512                      # Mini-batch size
LEARNING_RATE = 1e-4                  # Initial AdamW LR
MIN_LEARNING_RATE = 9e-8              # ReduceLROnPlateau floor
LR_PATIENCE = 1                       # Patience before decaying LR
LR_FACTOR = 0.8                       # 20% LR decay per stagnant epoch
WEIGHT_DECAY = 1e-4                   # AdamW weight decay

# Architecture Hyperparameters (V3.4 8-Section)
POLY_ORDER = 7
FILTER_KERNEL_SIZE = 31
MIN_AMPLITUDE = 0.85
MAX_AMPLITUDE = 1.15
BASELINE_CHANNELS = 32
FILTER_CHANNELS = 32
NUM_QUADRANTS = 8                     # 8-section spatial pooling

# Loss Weights
LAMBDA_BC = 1.0
LAMBDA_CLEAN = 1.0
LAMBDA_ASYM = 15.0

# Paths
DATA_DIR = os.path.join("training_round6", "training_data6")
SAVE_MODEL_PATH = "polygaussnet12.pth"
CHECKPOINT_PATH = "training_checkpoint12.pth"
LOG_FILE_PATH = "training_history12.txt"


# =====================================================================
# 2. CLI PARSER
# =====================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train PolyGaussNet V3.4 (8-section) for 97 epochs on Round 6 regime & dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Round 6 Training Regime
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Total training epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Initial learning rate")
    parser.add_argument("--min-lr", type=float, default=MIN_LEARNING_RATE, help="Minimum LR floor")
    parser.add_argument("--lr-patience", type=int, default=LR_PATIENCE, help="ReduceLROnPlateau patience")
    parser.add_argument("--lr-factor", type=float, default=LR_FACTOR, help="ReduceLROnPlateau factor")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY, help="AdamW weight decay")
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed for reproducibility")

    # Model Architecture (V3.4 8-section)
    parser.add_argument("--poly-order", type=int, default=POLY_ORDER, help="Polynomial baseline order")
    parser.add_argument("--filter-kernel-size", type=int, default=FILTER_KERNEL_SIZE, help="Gaussian kernel size (odd)")
    parser.add_argument("--min-amplitude", type=float, default=MIN_AMPLITUDE, help="Min amplitude modulation")
    parser.add_argument("--max-amplitude", type=float, default=MAX_AMPLITUDE, help="Max amplitude modulation")
    parser.add_argument("--baseline-channels", type=int, default=BASELINE_CHANNELS, help="Baseline stem width")
    parser.add_argument("--filter-channels", type=int, default=FILTER_CHANNELS, help="Filter CNN channel width")
    parser.add_argument("--num-quadrants", type=int, default=NUM_QUADRANTS, help="Global spatial pooling sections")

    # Loss Weights
    parser.add_argument("--lambda-bc", type=float, default=LAMBDA_BC, help="Baseline-corrected target loss weight")
    parser.add_argument("--lambda-clean", type=float, default=LAMBDA_CLEAN, help="Clean target loss weight")
    parser.add_argument("--lambda-asym", type=float, default=LAMBDA_ASYM, help="Asymmetric baseline overprediction weight")

    # Paths
    parser.add_argument("--data-dir", type=str, default=DATA_DIR, help="Path to Round 6 training_data6")
    parser.add_argument("--output-dir", type=str, default=SCRIPT_DIR, help="Directory for checkpoints, weights, and logs")
    parser.add_argument("--save-model", type=str, default=SAVE_MODEL_PATH, help="Best model weights filename/path")
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT_PATH, help="Resumable checkpoint filename/path")
    parser.add_argument("--log-file", type=str, default=LOG_FILE_PATH, help="Training log filename/path")
    parser.add_argument("--no-resume", action="store_true", help="Ignore existing checkpoint and train from epoch 1")

    # OSCAR / Hardware Optimization Flags
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"],
        help="Compute device (`auto` prefers CUDA > MPS > CPU)",
    )
    parser.add_argument(
        "--data-cache", type=str, default="auto", choices=["auto", "gpu", "ram", "stream"],
        help="Dataset caching strategy: preload onto GPU VRAM (`gpu`), CPU RAM (`ram`), "
             "chunk streaming (`stream`), or `auto` (VRAM if >=8GB headroom, else CPU RAM)",
    )
    parser.add_argument("--io-workers", type=int, default=0, help="Threads for initial .npz loading (0=auto)")
    parser.add_argument("--amp", action="store_true", help="Enable CUDA bfloat16 autocast for Conv/Linear layers")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile(dynamic=True) on CUDA")
    parser.add_argument("--no-cudnn-benchmark", action="store_true", help="Disable cudnn.benchmark")
    parser.add_argument("--log-every-chunk", action="store_true", help="Print per-chunk metrics every epoch (verbose)")
    parser.add_argument("--max-files-per-epoch", type=int, default=None, help="Cap chunks per epoch (smoke testing)")
    return parser.parse_args()


# =====================================================================
# 3. EXACT ROUND 6 LOSS FUNCTIONS
# =====================================================================
class LogCoshLoss(nn.Module):
    """Numerically stable Log-Cosh loss function."""

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        diff = y_pred - y_true
        abs_diff = torch.abs(diff)
        log2 = torch.log(torch.tensor(2.0, device=diff.device, dtype=diff.dtype))
        return torch.mean(abs_diff + torch.log1p(torch.exp(-2.0 * abs_diff)) - log2)


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
# 4. PATH, LOGGING, CHECKPOINT & DEVICE HELPERS
# =====================================================================
def resolve_path(path_str: str) -> str:
    if os.path.isabs(path_str):
        return path_str
    candidates = [
        os.path.abspath(path_str),
        os.path.join(SCRIPT_DIR, path_str),
        os.path.join(PROJECT_ROOT, path_str),
        os.path.join(PROJECT_ROOT, "training_round6", path_str),
    ]
    for cand in candidates:
        if os.path.exists(cand):
            return cand
    return os.path.abspath(path_str)


def resolve_output_path(path_str: str, output_dir: str) -> str:
    if os.path.isabs(path_str):
        return path_str
    return os.path.join(os.path.abspath(output_dir), path_str)


def discover_dataset_files(data_dir: str) -> list[str]:
    files = sorted(glob.glob(os.path.join(data_dir, "**", "*.npz"), recursive=True))
    if not files:
        files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
    return files


def load_file_data(filepath: str):
    """Load one .npz chunk into CPU float32 tensors with channel dim (N, 1, L)."""
    with np.load(filepath) as data:
        x = torch.from_numpy(np.ascontiguousarray(data["full_matrix"], dtype=np.float32)).unsqueeze(1)
        y_bc = torch.from_numpy(np.ascontiguousarray(data["pure_noise_cosmic_matrix"], dtype=np.float32)).unsqueeze(1)
        y_clean = torch.from_numpy(np.ascontiguousarray(data["pure_matrix"], dtype=np.float32)).unsqueeze(1)
    return x, y_bc, y_clean


def atomic_torch_save(obj, target_path: str):
    """Write checkpoint atomically so cluster SIGKILL/walltime cannot corrupt existing files."""
    tmp_path = f"{target_path}.tmp.{os.getpid()}"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, target_path)


class DualLogger:
    """Line-buffered logger duplicating stdout to an append-mode file."""

    def __init__(self, filepath: str):
        self.terminal = sys.stdout
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        self.log = open(filepath, "a", encoding="utf-8", buffering=1)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def select_device(pref: str) -> torch.device:
    if pref == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested, but CUDA is not available.")
        return torch.device("cuda")
    if pref == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("--device mps requested, but MPS is not available.")
        return torch.device("mps")
    if pref == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_rng_state(device: torch.device) -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
    }
    if device.type == "cuda":
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict, device: torch.device):
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch_cpu" in state:
        torch.set_rng_state(state["torch_cpu"].cpu().to(torch.uint8))
    if device.type == "cuda" and "torch_cuda" in state:
        try:
            cuda_states = [s.cpu().to(torch.uint8) for s in state["torch_cuda"]]
            torch.cuda.set_rng_state_all(cuda_states)
        except Exception:
            pass


def preload_all_chunks(chunk_files: list[str], data_dir: str, device: torch.device, cache_mode: str, io_workers: int):
    """
    Preload all .npz chunks once into memory to prevent repeated filesystem I/O across 97 epochs.
    Returns (cached_chunks, effective_mode).
    """
    t0 = time.time()
    workers = io_workers if io_workers > 0 else min(len(chunk_files), int(os.environ.get("SLURM_CPUS_PER_TASK", 4)), 8)
    workers = max(1, workers)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        loaded = list(pool.map(load_file_data, chunk_files))

    total_bytes = sum(t.numel() * t.element_size() for triple in loaded for t in triple)
    total_gb = total_bytes / (1024 ** 3)

    effective_mode = cache_mode
    if cache_mode == "auto":
        if device.type == "cuda":
            free_vram, _ = torch.cuda.mem_get_info(device)
            # Reserve 8 GB headroom for activations, cuDNN workspace, and PyTorch caching allocator
            headroom = 8 * (1024 ** 3)
            effective_mode = "gpu" if free_vram >= (total_bytes + headroom) else "ram"
        else:
            effective_mode = "ram"

    pin_ok = (device.type == "cuda" and effective_mode == "ram")
    cached_chunks = []

    for filepath, (x, y_bc, y_clean) in zip(chunk_files, loaded):
        rel_name = os.path.relpath(filepath, data_dir)
        if effective_mode == "gpu" and device.type == "cuda":
            x = x.to(device)
            y_bc = y_bc.to(device)
            y_clean = y_clean.to(device)
        elif pin_ok:
            try:
                x = x.pin_memory()
                y_bc = y_bc.pin_memory()
                y_clean = y_clean.pin_memory()
            except RuntimeError:
                pin_ok = False
        cached_chunks.append((rel_name, x, y_bc, y_clean))

    elapsed = time.time() - t0
    pin_tag = " (pinned)" if (effective_mode == "ram" and pin_ok) else ""
    print(f"Preloaded {len(cached_chunks)} chunks ({total_gb:.2f} GiB float32) into {effective_mode.upper()}{pin_tag} in {elapsed:.1f}s")
    return cached_chunks, effective_mode


# =====================================================================
# 5. MAIN TRAINING LOOP
# =====================================================================
def train(args=None):
    if args is None:
        args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = resolve_output_path(args.log_file, args.output_dir)
    if not isinstance(sys.stdout, DualLogger):
        sys.stdout = DualLogger(log_path)

    device = select_device(args.device)
    if args.seed is not None:
        set_seed(args.seed)

    # CUDA throughput tuning (does not alter Round 6 math when AMP is off)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = not args.no_cudnn_benchmark

    amp_enabled = bool(args.amp and device.type == "cuda")
    amp_dtype = torch.bfloat16 if (amp_enabled and torch.cuda.is_bf16_supported()) else torch.float16

    resolved_data_dir = resolve_path(args.data_dir)
    chunk_files = discover_dataset_files(resolved_data_dir)
    if not chunk_files:
        raise FileNotFoundError(
            f"No .npz files found in '{resolved_data_dir}'. Expected Round 6 dataset `training_round6/training_data6`."
        )
    if args.max_files_per_epoch is not None:
        chunk_files = chunk_files[:args.max_files_per_epoch]

    print("\n" + "=" * 84)
    print(" PolyGaussNet V3.4 (8-Section) Training — Round 12 (Exact Round 6 Regime) ")
    print("=" * 84)
    if "SLURM_JOB_ID" in os.environ:
        print(f"  - Slurm Job ID:       {os.environ['SLURM_JOB_ID']} (CPUs={os.environ.get('SLURM_CPUS_PER_TASK', '?')})")
    print(f"  - Device:             {device}" + (f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else ""))
    print(f"  - AMP / torch.compile:{' '}{amp_enabled} ({amp_dtype if amp_enabled else 'fp32'}) / {bool(args.compile and device.type == 'cuda')}")
    print(f"  - Epochs:             {args.epochs} | Batch Size: {args.batch_size}")
    print(f"  - Optimizer:          AdamW (lr={args.lr}, weight_decay={args.weight_decay})")
    print(f"  - LR Scheduler:       ReduceLROnPlateau (factor={args.lr_factor}, patience={args.lr_patience}, min_lr={args.min_lr})")
    print(f"  - Architecture:       V3.4 num_quadrants={args.num_quadrants}, poly_order={args.poly_order}, "
          f"K={args.filter_kernel_size}, amp=[{args.min_amplitude}, {args.max_amplitude}]")
    print(f"  - Loss Weights:       bc={args.lambda_bc}, clean={args.lambda_clean}, asym={args.lambda_asym}")
    print(f"  - Dataset Dir:        {resolved_data_dir} ({len(chunk_files)} chunks)")
    print(f"  - Output Dir:         {os.path.abspath(args.output_dir)}")
    print("=" * 84 + "\n")

    model = PolyGaussNet(
        poly_order=args.poly_order,
        filter_kernel_size=args.filter_kernel_size,
        min_amplitude=args.min_amplitude,
        max_amplitude=args.max_amplitude,
        baseline_channels=args.baseline_channels,
        filter_channels=args.filter_channels,
        num_quadrants=args.num_quadrants,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"PolyGaussNet V3.4 (8-sec) Parameters: {total_params:,} total ({trainable_params:,} trainable)")

    criterion = LogCoshLoss().to(device)
    asym_penalty = AsymmetricBaselinePenalty(lambda_asym=args.lambda_asym).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=args.lr_factor, patience=args.lr_patience, min_lr=args.min_lr
    )

    checkpoint_file = resolve_output_path(args.checkpoint, args.output_dir)
    save_model_file = resolve_output_path(args.save_model, args.output_dir)

    start_epoch = 1
    best_loss = float("inf")

    if not args.no_resume and os.path.exists(checkpoint_file):
        print(f"Found checkpoint at '{checkpoint_file}'. Resuming...")
        ckpt = torch.load(checkpoint_file, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        restore_rng_state(ckpt.get("rng_state"), device)
        start_epoch = ckpt.get("epoch", 0) + 1
        best_loss = ckpt.get("best_loss", float("inf"))
        print(f"Resumed at Epoch {start_epoch} (Best Loss so far: {best_loss:.6f})\n")

    if start_epoch > args.epochs:
        print(f"Model already completed Epoch {args.epochs}. Nothing to do.")
        return 0

    train_model = torch.compile(model, dynamic=True) if (args.compile and device.type == "cuda") else model

    cached_chunks = None
    effective_cache = args.data_cache
    if args.data_cache != "stream":
        cached_chunks, effective_cache = preload_all_chunks(
            chunk_files, resolved_data_dir, device, args.data_cache, args.io_workers
        )

    # Graceful Slurm walltime preemption / Ctrl+C handling:
    # finishes the active epoch cleanly so epoch checkpoint matches Round 6 boundaries.
    stop_requested = {"flag": False, "sig": None}

    def _handle_stop_signal(signum, _frame):
        stop_requested["flag"] = True
        stop_requested["sig"] = signum
        print(f"\n[SIGNAL {signum}] Graceful stop requested; finishing current epoch to save checkpoint...")

    prev_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, _handle_stop_signal)

    stream_pool = ThreadPoolExecutor(max_workers=1) if effective_cache == "stream" else None

    print("=" * 84)
    print(f"Starting Training: Epochs {start_epoch} to {args.epochs} (cache={effective_cache})")
    print("=" * 84 + "\n")

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            epoch_start = time.time()
            epoch_loss_sum = 0.0
            epoch_bc_sum = 0.0
            epoch_clean_sum = 0.0
            epoch_asym_sum = 0.0
            epoch_samples = 0

            train_model.train()

            if cached_chunks is not None:
                order = list(range(len(cached_chunks)))
                random.shuffle(order)
                chunk_iter = (cached_chunks[i] for i in order)
                num_files = len(order)
            else:
                epoch_files = list(chunk_files)
                random.shuffle(epoch_files)
                num_files = len(epoch_files)
                future_data = stream_pool.submit(load_file_data, epoch_files[0])

                def _stream_iter():
                    nonlocal future_data
                    for idx, fp in enumerate(epoch_files, 1):
                        x, y_bc, y_clean = future_data.result()
                        if idx < num_files:
                            future_data = stream_pool.submit(load_file_data, epoch_files[idx])
                        else:
                            future_data = None
                        yield os.path.relpath(fp, resolved_data_dir), x, y_bc, y_clean

                chunk_iter = _stream_iter()

            for file_idx, (rel_name, X, Y_bc, Y_clean) in enumerate(chunk_iter, 1):
                file_samples = X.shape[0]
                bin_length = X.shape[-1]
                perm = torch.randperm(file_samples, device=X.device)

                file_loss_sum = 0.0
                file_bc_sum = 0.0
                file_clean_sum = 0.0
                file_asym_sum = 0.0
                file_start = time.time()

                for start in range(0, file_samples, args.batch_size):
                    idx = perm[start : start + args.batch_size]
                    batch_x = X[idx]
                    batch_y_bc = Y_bc[idx]
                    batch_y_clean = Y_clean[idx]

                    if batch_x.device != device:
                        batch_x = batch_x.to(device, non_blocking=True)
                        batch_y_bc = batch_y_bc.to(device, non_blocking=True)
                        batch_y_clean = batch_y_clean.to(device, non_blocking=True)

                    optimizer.zero_grad(set_to_none=True)

                    if amp_enabled:
                        with torch.amp.autocast("cuda", dtype=amp_dtype):
                            pred_clean, pred_baseline, pred_bc, _ = train_model(batch_x)
                        loss_bc = criterion(pred_bc.float(), batch_y_bc)
                        loss_clean = criterion(pred_clean.float(), batch_y_clean)
                        loss_asym = asym_penalty(pred_baseline.float(), batch_x)
                    else:
                        pred_clean, pred_baseline, pred_bc, _ = train_model(batch_x)
                        loss_bc = criterion(pred_bc, batch_y_bc)
                        loss_clean = criterion(pred_clean, batch_y_clean)
                        loss_asym = asym_penalty(pred_baseline, batch_x)

                    total_loss = (
                        args.lambda_bc * loss_bc
                        + args.lambda_clean * loss_clean
                        + loss_asym
                    )

                    total_loss.backward()
                    optimizer.step()

                    b_size = batch_x.size(0)
                    file_loss_sum += total_loss.item() * b_size
                    file_bc_sum += loss_bc.item() * b_size
                    file_clean_sum += loss_clean.item() * b_size
                    file_asym_sum += loss_asym.item() * b_size

                epoch_loss_sum += file_loss_sum
                epoch_bc_sum += file_bc_sum
                epoch_clean_sum += file_clean_sum
                epoch_asym_sum += file_asym_sum
                epoch_samples += file_samples

                if args.log_every_chunk or epoch == start_epoch or epoch == args.epochs:
                    file_dur = time.time() - file_start
                    tput = file_samples / file_dur if file_dur > 0 else 0.0
                    print(
                        f"Epoch [{epoch:3d}/{args.epochs:3d}] | File [{file_idx:2d}/{num_files:2d}] "
                        f"({rel_name}, L={bin_length:4d}) | "
                        f"Loss: {file_loss_sum / file_samples:.5f} "
                        f"(BC: {file_bc_sum / file_samples:.4f}, Clean: {file_clean_sum / file_samples:.4f}, "
                        f"Asym: {file_asym_sum / file_samples:.4f}) | {tput:7.1f} spec/s"
                    )

            if device.type == "cuda":
                torch.cuda.synchronize()
            elif device.type == "mps":
                torch.mps.synchronize()

            epoch_dur = time.time() - epoch_start
            avg_loss = epoch_loss_sum / epoch_samples
            avg_bc = epoch_bc_sum / epoch_samples
            avg_clean = epoch_clean_sum / epoch_samples
            avg_asym = epoch_asym_sum / epoch_samples
            epoch_tput = epoch_samples / epoch_dur if epoch_dur > 0 else 0.0

            scheduler.step(avg_loss)
            current_lr = optimizer.param_groups[0]["lr"]

            print("-" * 84)
            print(f"==> Epoch [{epoch:3d}/{args.epochs:3d}] Completed in {epoch_dur:.2f}s ({epoch_tput:,.1f} spectra/s)")
            print(
                f"    Avg Total Loss: {avg_loss:.6f} | Avg BC: {avg_bc:.6f} | "
                f"Avg Clean: {avg_clean:.6f} | Avg Asym: {avg_asym:.6f}"
            )
            print(f"    Learning Rate:  {current_lr:.6e}")

            raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            if avg_loss < best_loss:
                best_loss = avg_loss
                atomic_torch_save(raw_model.state_dict(), save_model_file)
                print(f"    * New best model saved to '{save_model_file}' (Loss: {best_loss:.6f})")

            atomic_torch_save(
                {
                    "epoch": epoch,
                    "model_state_dict": raw_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "best_loss": best_loss,
                    "rng_state": get_rng_state(device),
                    "args": vars(args),
                },
                checkpoint_file,
            )
            print(f"    Saved resumable checkpoint to '{checkpoint_file}'")
            print("-" * 84)

            if stop_requested["flag"] and epoch < args.epochs:
                print(f"\nStopping cleanly after Epoch {epoch} due to signal {stop_requested['sig']}.")
                print(f"Resubmit job to resume from Epoch {epoch + 1} via '{checkpoint_file}'.")
                return 0

    except KeyboardInterrupt:
        print("\n" + "=" * 84)
        print(" Training interrupted (Ctrl+C). Last completed epoch checkpoint is preserved:")
        print(f"   {checkpoint_file}")
        print("=" * 84 + "\n")
        return 0
    finally:
        signal.signal(signal.SIGTERM, prev_sigterm)
        if stream_pool is not None:
            stream_pool.shutdown(wait=False)

    print("\n" + "=" * 84)
    print("TRAINING PROCESS COMPLETED SUCCESSFULLY")
    print(f"Best Training Loss: {best_loss:.6f}")
    print(f"Best Model Saved:   {save_model_file}")
    print("=" * 84 + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(train())
