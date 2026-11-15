"""
TESTING & BENCHMARKING SCRIPT - ROUND 12
----------------------------------------
Head-to-head evaluation on the Round 6 validation/testing dataset (`training_round6/testing_data6`,
5 resolutions x 4,096 spectra = 20,480 spectra) comparing:
  1. Conventional Preprocessing Pipeline (test_files/conventional/pre.py)
  2. PolyGaussNet V3 (Round 6 checkpoint: `training_round6/polygaussnet6.pth`, Epoch 97, 148,426 params)
  3. PolyGaussNet V3.4 (8-Section, Round 12: `training_round12/polygaussnet12.pth`, 108,394 params)

EVALUATION MODES:
  - PART A: Standard Raw Validation (unshifted outputs vs ground truth pure spectra + baseline MSE/overshoot)
  - PART B: Baseline-Shifted Validation (`mode='mean'` DC alignment applied identically to all 3 methods)
  - PART C: Computational Speed & Throughput (with proper CUDA/MPS warmup and synchronization)
  - PART D: Per-Resolution Breakdown (`800, 1016, 1200, 1400, 1600` bins)
  - PART E: Pairwise Win-Count Summary across resolutions

GENERATES:
  - `validation_results_round12.txt`: Formatted head-to-head report matching prior round reports.
  - `testing12_comparison_overlay.png`: Visual multi-spectrum comparison overlay (Gruvbox Dark).
  - `testing12_metrics_benchmark.png`: Summary bar chart across all 3 methods (unless `--no-plots`).
"""

import argparse
import glob
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib
matplotlib.use("Agg")  # Headless backend safe for OSCAR Slurm compute nodes
import matplotlib.pyplot as plt
import numpy as np
import torch

# Configure paths so imports resolve from project root or training_round12/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
CONVENTIONAL_DIR = os.path.join(PROJECT_ROOT, "test_files", "conventional")
RAMAN_PLOTTING_SIBLING = os.path.abspath(os.path.join(PROJECT_ROOT, "..", "raman plotting"))

for p in (PROJECT_ROOT, SCRIPT_DIR, CONVENTIONAL_DIR, RAMAN_PLOTTING_SIBLING):
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

from polygaussnet3 import PolyGaussNet as PolyGaussNet3
from polygaussnet3_4 import PolyGaussNet as PolyGaussNet3_4

# =====================================================================
# 1. DEFAULTS & CONSTANTS
# =====================================================================
DEFAULT_DATA_DIR = os.path.join("training_round6", "testing_data6")
DEFAULT_FALLBACK_DIR = os.path.join("training_round6", "training_data6")
DEFAULT_V3_MODEL = os.path.join("training_round6", "polygaussnet6.pth")
DEFAULT_V3_CKPT = os.path.join("training_round6", "training_checkpoint6.pth")
DEFAULT_V3_4_MODEL = "polygaussnet12.pth"
DEFAULT_V3_4_CKPT = "training_checkpoint12.pth"
DEFAULT_REPORT_PATH = "validation_results_round12.txt"
DEFAULT_PLOT_OVERLAY = "testing12_comparison_overlay.png"
DEFAULT_PLOT_METRICS = "testing12_metrics_benchmark.png"

BIN_SIZES = [800, 1016, 1200, 1400, 1600]
DEFAULT_BATCH_SIZE = 512

POLY_ORDER = 7
FILTER_KERNEL_SIZE = 31
MIN_AMPLITUDE = 0.85
MAX_AMPLITUDE = 1.15
V3_4_BASELINE_CH = 32
V3_4_FILTER_CH = 32
V3_4_QUADRANTS = 8

GRUVBOX = {
    "bg0_hard": "#1d2021",
    "bg0": "#282828",
    "bg1": "#3c3836",
    "fg0": "#fbf1c7",
    "fg1": "#ebdbb2",
    "gray": "#928374",
    "red": "#fb4934",
    "green": "#b8bb26",
    "yellow": "#fabd2f",
    "blue": "#83a598",
    "purple": "#d3869b",
    "aqua": "#8ec07c",
    "orange": "#fe8019",
}


def default_n_jobs() -> int:
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus and slurm_cpus.isdigit():
        return max(1, int(slurm_cpus))
    return min(os.cpu_count() or 4, 8)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Head-to-head benchmark: Conventional vs V3 (Round 6) vs V3.4-8sec (Round 12) on testing_data6.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR, help="Round 6 validation/test dataset dir")
    parser.add_argument("--v3-model", type=str, default=DEFAULT_V3_MODEL, help="Round 6 V3 weights (.pth)")
    parser.add_argument("--v3-ckpt", type=str, default=DEFAULT_V3_CKPT, help="Round 6 V3 checkpoint metadata (.pth)")
    parser.add_argument("--v3_4-model", "--v34-model", dest="v3_4_model", type=str, default=DEFAULT_V3_4_MODEL,
                        help="Round 12 V3.4-8sec weights (.pth)")
    parser.add_argument("--v3_4-ckpt", "--v34-ckpt", dest="v3_4_ckpt", type=str, default=DEFAULT_V3_4_CKPT,
                        help="Round 12 V3.4-8sec checkpoint metadata (.pth)")
    parser.add_argument("--require-epoch", type=int, default=0,
                        help="If >0, abort gracefully if Round 12 checkpoint epoch < require-epoch (prevents premature eval after Slurm preemption)")
    parser.add_argument("--bin-sizes", type=int, nargs="+", default=BIN_SIZES, help="Resolutions to evaluate")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="PyTorch inference batch size")
    parser.add_argument("--n-jobs", type=int, default=default_n_jobs(), help="CPU workers for conventional pipeline")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"], help="Compute device")
    parser.add_argument("--no-conv", action="store_true", help="Skip conventional preprocessing baseline")
    parser.add_argument("--no-shift", action="store_true", help="Skip baseline-shifted evaluation")
    parser.add_argument("--align-mode", type=str, default="mean", choices=["mean", "min_zero"],
                        help="Baseline alignment shift applied identically to all models")
    parser.add_argument("--no-plots", action="store_true", help="Skip generating PNG plots")
    parser.add_argument("--max-samples-per-chunk", type=int, default=None, help="Limit spectra per bin resolution (smoke test)")
    parser.add_argument("--output-dir", type=str, default=SCRIPT_DIR, help="Directory for report & plots")
    parser.add_argument("--report-path", type=str, default=DEFAULT_REPORT_PATH, help="Output .txt report filename/path")
    return parser.parse_args()


# =====================================================================
# 2. PATH & DEVICE HELPERS
# =====================================================================
def resolve_path(path_str: str) -> str:
    if os.path.isabs(path_str) and os.path.exists(path_str):
        return path_str
    for cand in (
        os.path.abspath(path_str),
        os.path.join(SCRIPT_DIR, path_str),
        os.path.join(PROJECT_ROOT, path_str),
        os.path.join(PROJECT_ROOT, "training_round6", path_str),
    ):
        if os.path.exists(cand):
            return cand
    return path_str


def resolve_output_path(path_str: str, output_dir: str) -> str:
    if os.path.isabs(path_str):
        return path_str
    return os.path.join(os.path.abspath(output_dir), path_str)


def select_device(pref: str) -> torch.device:
    if pref == "cuda":
        return torch.device("cuda")
    if pref == "mps":
        return torch.device("mps")
    if pref == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def discover_test_files(target_dir: str, fallback_dir: str, bin_sizes: list[int]) -> tuple[dict, str]:
    resolved_target = resolve_path(target_dir)
    files_by_bin = {}
    active_dir = resolved_target

    if os.path.exists(resolved_target):
        for b in bin_sizes:
            bin_path = os.path.join(resolved_target, str(b))
            if os.path.exists(bin_path):
                chunks = sorted(glob.glob(os.path.join(bin_path, "*.npz")))
                if chunks:
                    files_by_bin[b] = chunks[0]

    if not files_by_bin:
        resolved_fallback = resolve_path(fallback_dir)
        if os.path.exists(resolved_fallback):
            for b in bin_sizes:
                bin_path = os.path.join(resolved_fallback, str(b))
                if os.path.exists(bin_path):
                    chunks = sorted(glob.glob(os.path.join(bin_path, "*.npz")))
                    if chunks:
                        files_by_bin[b] = chunks[-1]
            if files_by_bin:
                active_dir = resolved_fallback

    return files_by_bin, active_dir


def load_model_weights(model, weight_candidates: list[str], ckpt_candidates: list[str], device: torch.device):
    weight_path = next((p for p in weight_candidates if p and os.path.exists(p)), None)
    if weight_path is None:
        raise FileNotFoundError(f"Model weights not found. Checked: {weight_candidates}")

    ckpt = torch.load(weight_path, map_location=device, weights_only=False)
    epoch = None
    best_loss = None
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        epoch = ckpt.get("epoch")
        best_loss = ckpt.get("best_loss", ckpt.get("loss"))
    else:
        state_dict = ckpt
        for cp in ckpt_candidates:
            if cp and os.path.exists(cp):
                try:
                    meta = torch.load(cp, map_location="cpu", weights_only=False)
                    if isinstance(meta, dict):
                        epoch = meta.get("epoch")
                        best_loss = meta.get("best_loss", meta.get("loss"))
                        break
                except Exception:
                    pass

    # Handle state_dict accidentally saved from torch.compile wrapper
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()
    return model, weight_path, epoch, best_loss


# =====================================================================
# 3. METRICS & BASELINE ALIGNMENT
# =====================================================================
def align_baseline(y_true: np.ndarray, y_pred: np.ndarray, mode: str = "mean") -> np.ndarray:
    if mode == "mean":
        return y_pred - np.mean(y_pred - y_true, axis=-1, keepdims=True)
    if mode == "min_zero":
        return y_pred - np.min(y_pred, axis=-1, keepdims=True)
    raise ValueError(f"Unknown alignment mode: {mode}")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Vectorized spectral metrics matching testing6.py."""
    y_t = y_true.astype(np.float64, copy=False)
    y_p = y_pred.astype(np.float64, copy=False)

    dot = np.sum(y_t * y_p, axis=1)
    norm_t = np.linalg.norm(y_t, axis=1)
    norm_p = np.linalg.norm(y_p, axis=1)
    valid = (norm_t > 1e-12) & (norm_p > 1e-12)
    both_zero = (norm_t <= 1e-12) & (norm_p <= 1e-12)

    cosine_sim = np.zeros(y_t.shape[0], dtype=np.float64)
    cosine_sim[both_zero] = 1.0
    cosine_sim[valid] = dot[valid] / (norm_t[valid] * norm_p[valid])

    diff = y_p - y_t
    mse = np.mean(diff ** 2, axis=1)
    abs_diff = np.abs(diff)
    log_cosh = np.mean(abs_diff + np.log1p(np.exp(-2.0 * abs_diff)) - np.log(2.0), axis=1)
    mae = np.mean(abs_diff, axis=1)
    rmse = np.sqrt(mse)

    # Top 5% Peak Summit MAE (vectorized equivalent to testing6.py)
    thresh = np.percentile(y_t, 95, axis=1, keepdims=True)
    peak_mask = y_t >= thresh
    counts = peak_mask.sum(axis=1)
    peak_mae = np.where(counts > 0, (abs_diff * peak_mask).sum(axis=1) / np.maximum(counts, 1), mae)

    arrays = {
        "cosine_sim": cosine_sim,
        "mse": mse,
        "log_cosh": log_cosh,
        "mae": mae,
        "rmse": rmse,
        "peak_mae": peak_mae,
    }
    summary = {
        "Cosine Similarity": (float(np.mean(cosine_sim)), float(np.std(cosine_sim))),
        "MSE": (float(np.mean(mse)), float(np.std(mse))),
        "Log-Cosh": (float(np.mean(log_cosh)), float(np.std(log_cosh))),
        "MAE": (float(np.mean(mae)), float(np.std(mae))),
        "RMSE": (float(np.mean(rmse)), float(np.std(rmse))),
        "Peak Summit MAE": (float(np.mean(peak_mae)), float(np.std(peak_mae))),
        "_arrays": arrays,
    }
    return summary


def aggregate_metrics(arrays_list: list[dict]) -> dict:
    if not arrays_list:
        return {}
    merged = {k: np.concatenate([a[k] for a in arrays_list]) for k in arrays_list[0]}
    return {
        "Cosine Similarity": (float(np.mean(merged["cosine_sim"])), float(np.std(merged["cosine_sim"]))),
        "MSE": (float(np.mean(merged["mse"])), float(np.std(merged["mse"]))),
        "Log-Cosh": (float(np.mean(merged["log_cosh"])), float(np.std(merged["log_cosh"]))),
        "MAE": (float(np.mean(merged["mae"])), float(np.std(merged["mae"]))),
        "RMSE": (float(np.mean(merged["rmse"])), float(np.std(merged["rmse"]))),
        "Peak Summit MAE": (float(np.mean(merged["peak_mae"])), float(np.std(merged["peak_mae"]))),
        "_arrays": merged,
    }


# =====================================================================
# 4. CONVENTIONAL PIPELINE & NEURAL INFERENCE RUNNERS
# =====================================================================
def _init_conv_worker():
    """Prevent BLAS thread oversubscription inside multiprocessing workers."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"


def _process_conv_chunk(wavenumbers_chunk: np.ndarray, raw_chunk: np.ndarray) -> np.ndarray:
    from spectra_class import spectra
    import pre

    conv_data = spectra.from_matrices(wavenumbers_chunk, raw_chunk)
    pre.preprocess_pipeline(conv_data, normalize=False, shift=False)
    return conv_data.intensity_matrix


def run_conventional_pipeline(raw_np: np.ndarray, wavenumbers_np: np.ndarray, n_jobs: int) -> np.ndarray:
    # Verify conventional dependencies up front
    try:
        import spectra_class  # noqa: F401
        import pre  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Conventional pipeline dependencies (`spectra_class` / `pre`) are missing. "
            "Install `pip install -r requirements.txt` (includes Raman-Plotting-Pipeline) or pass `--no-conv`."
        ) from exc

    n_samples = len(raw_np)
    if n_jobs <= 1 or n_samples < 32:
        return _process_conv_chunk(wavenumbers_np, np.copy(raw_np))

    chunk_size = int(np.ceil(n_samples / n_jobs))
    chunks = [
        (wavenumbers_np[i : min(i + chunk_size, n_samples)], np.copy(raw_np[i : min(i + chunk_size, n_samples)]))
        for i in range(0, n_samples, chunk_size)
    ]

    results = [None] * len(chunks)
    with ProcessPoolExecutor(max_workers=n_jobs, initializer=_init_conv_worker) as pool:
        futures = {pool.submit(_process_conv_chunk, wn, raw): idx for idx, (wn, raw) in enumerate(chunks)}
        for fut in as_completed(futures):
            results[futures[fut]] = fut.result()
    return np.vstack(results)


def run_nn_inference(model, raw_mat: np.ndarray, device: torch.device, batch_size: int):
    """Runs neural inference on one resolution matrix with warmup & device sync for accurate timing."""
    raw_tensor = torch.from_numpy(np.ascontiguousarray(raw_mat, dtype=np.float32)).unsqueeze(1)
    if device.type == "cuda":
        try:
            raw_tensor = raw_tensor.pin_memory()
        except RuntimeError:
            pass

    # Tiny warmup so first-call kernel setup does not skew timing
    with torch.no_grad():
        warm_x = raw_tensor[: min(4, len(raw_tensor))].to(device)
        _ = model(warm_x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()

    clean_chunks, base_chunks = [], []
    t0 = time.perf_counter()
    with torch.no_grad():
        for i in range(0, len(raw_tensor), batch_size):
            bx = raw_tensor[i : i + batch_size].to(device, non_blocking=(device.type == "cuda"))
            pred_clean, pred_base, _, _ = model(bx)
            clean_chunks.append(pred_clean.squeeze(1).cpu())
            base_chunks.append(pred_base.squeeze(1).cpu())
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()
    elapsed = time.perf_counter() - t0

    return torch.cat(clean_chunks, dim=0).numpy(), torch.cat(base_chunks, dim=0).numpy(), elapsed


# =====================================================================
# 5. FORMATTING & PLOTTING HELPERS
# =====================================================================
def fmt_cell(val_std) -> str:
    if val_std is None:
        return "N/A"
    val, std = val_std
    return f"{val:9.5f} (±{std:.4f})"


def fmt_delta(metric_name: str, new_val: float, ref_val: float | None) -> str:
    if ref_val is None or ref_val == 0:
        return "N/A"
    if "Cosine" in metric_name:
        d = (new_val - ref_val) * 100.0
        return f"{d:+7.2f}p"
    d = (new_val - ref_val) / abs(ref_val) * 100.0
    return f"{d:+7.2f}%"


def generate_plots(sample_overlays, overall, output_dir, align_mode):
    plt.rcParams.update({
        "figure.facecolor": GRUVBOX["bg0_hard"],
        "axes.facecolor": GRUVBOX["bg0"],
        "axes.edgecolor": GRUVBOX["fg1"],
        "axes.labelcolor": GRUVBOX["fg0"],
        "xtick.color": GRUVBOX["fg1"],
        "ytick.color": GRUVBOX["fg1"],
        "grid.color": GRUVBOX["bg1"],
        "text.color": GRUVBOX["fg0"],
        "font.family": "sans-serif",
    })

    # Plot 1: Per-resolution spectrum overlay
    fig, axes = plt.subplots(len(sample_overlays), 1, figsize=(15, 3.4 * len(sample_overlays)), dpi=160)
    if len(sample_overlays) == 1:
        axes = [axes]

    for idx, item in enumerate(sample_overlays):
        ax = axes[idx]
        wn = np.arange(item["bin_size"])
        ax.plot(wn, item["raw"], label="Raw Noisy Input", color=GRUVBOX["gray"], alpha=0.45, linewidth=0.8)
        ax.plot(wn, item["pure"], label="Ground Truth Pure", color=GRUVBOX["blue"], alpha=0.90, linewidth=1.6)

        key = "shifted" if align_mode else "raw"
        if item["conv_" + key] is not None:
            ax.plot(wn, item["conv_" + key], label=f"Conventional ({key})", color=GRUVBOX["orange"], alpha=0.85, linewidth=1.2)
        ax.plot(wn, item["v3_" + key], label=f"V3 Round 6 ({key})", color=GRUVBOX["purple"], alpha=0.88, linewidth=1.35)
        ax.plot(wn, item["v3_4_" + key], label=f"V3.4-8sec R12 ({key})", color=GRUVBOX["green"], linewidth=1.65)

        title = f"L = {item['bin_size']} bins | Shifted MSE: V3={item['v3_mse_s']:.5f}, V3.4-8sec={item['v3_4_mse_s']:.5f}"
        if item["conv_mse_s"] is not None:
            title += f", Conv={item['conv_mse_s']:.5f}"
        ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
        ax.set_ylabel("Intensity")
        ax.grid(True, linestyle=":", alpha=0.6)
        if idx == 0:
            ax.legend(loc="upper right", framealpha=0.9, fontsize=8.5, ncol=3)

    axes[-1].set_xlabel("Spectral Bin Index")
    plt.tight_layout()
    overlay_path = os.path.join(output_dir, DEFAULT_PLOT_OVERLAY)
    plt.savefig(overlay_path, dpi=160)
    plt.close()

    # Plot 2: Metric summary bar charts
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=160)
    methods = []
    colors = []
    if overall["conv_raw"] is not None:
        methods.append(("conv", "Conventional"))
        colors.append(GRUVBOX["orange"])
    methods.extend([("v3", "V3 (Round 6)"), ("v3_4", "V3.4-8sec (R12)")])
    colors.extend([GRUVBOX["purple"], GRUVBOX["green"]])

    x = np.arange(len(methods))
    width = 0.36

    # Panel 1: MSE (Raw vs Shifted)
    raw_mse = [overall[f"{k}_raw"]["MSE"][0] for k, _ in methods]
    axes[0].bar(x - width / 2, raw_mse, width, label="Raw MSE", color=GRUVBOX["gray"], alpha=0.85)
    if overall["v3_shifted"] is not None:
        shf_mse = [overall[f"{k}_shifted"]["MSE"][0] for k, _ in methods]
        axes[0].bar(x + width / 2, shf_mse, width, label="Shifted MSE", color=colors)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([lbl for _, lbl in methods])
    axes[0].set_title("MSE (Lower is Better)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, axis="y", linestyle=":", alpha=0.5)

    # Panel 2: Cosine Similarity (Raw vs Shifted)
    raw_cos = [overall[f"{k}_raw"]["Cosine Similarity"][0] for k, _ in methods]
    axes[1].bar(x - width / 2, raw_cos, width, label="Raw Cos", color=GRUVBOX["gray"], alpha=0.85)
    if overall["v3_shifted"] is not None:
        shf_cos = [overall[f"{k}_shifted"]["Cosine Similarity"][0] for k, _ in methods]
        axes[1].bar(x + width / 2, shf_cos, width, label="Shifted Cos", color=colors)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([lbl for _, lbl in methods])
    axes[1].set_ylim(0.5, 1.0)
    axes[1].set_title("Cosine Similarity (Higher is Better)")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, axis="y", linestyle=":", alpha=0.5)

    # Panel 3: Peak Summit MAE (Raw vs Shifted)
    raw_pk = [overall[f"{k}_raw"]["Peak Summit MAE"][0] for k, _ in methods]
    axes[2].bar(x - width / 2, raw_pk, width, label="Raw Peak MAE", color=GRUVBOX["gray"], alpha=0.85)
    if overall["v3_shifted"] is not None:
        shf_pk = [overall[f"{k}_shifted"]["Peak Summit MAE"][0] for k, _ in methods]
        axes[2].bar(x + width / 2, shf_pk, width, label="Shifted Peak MAE", color=colors)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels([lbl for _, lbl in methods])
    axes[2].set_title("Peak Summit MAE (Lower is Better)")
    axes[2].legend(fontsize=8)
    axes[2].grid(True, axis="y", linestyle=":", alpha=0.5)

    plt.tight_layout()
    metrics_path = os.path.join(output_dir, DEFAULT_PLOT_METRICS)
    plt.savefig(metrics_path, dpi=160)
    plt.close()
    return overlay_path, metrics_path


# =====================================================================
# 6. MAIN BENCHMARK
# =====================================================================
def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = select_device(args.device)

    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Guard against premature testing on Slurm preemption
    v3_4_ckpt_resolved = resolve_path(args.v3_4_ckpt)
    if args.require_epoch > 0 and os.path.exists(v3_4_ckpt_resolved):
        meta = torch.load(v3_4_ckpt_resolved, map_location="cpu", weights_only=False)
        done_epoch = meta.get("epoch", 0) if isinstance(meta, dict) else 0
        if done_epoch < args.require_epoch:
            print(
                f"[DEFERRED] Training checkpoint at epoch {done_epoch}/{args.require_epoch}. "
                "Re-submit Slurm job to finish training before running testing12.py."
            )
            return 0

    lines = []

    def log(msg: str = ""):
        print(msg)
        lines.append(msg)

    W = 122
    log("\n" + "=" * W)
    log(" ROUND 12 HEAD-TO-HEAD BENCHMARK: Conventional Pipeline vs PolyGaussNet V3 (Round 6) vs PolyGaussNet V3.4-8sec (Round 12) ")
    log(" Evaluation Dataset: Round 6 Held-Out Validation Set | Dual Validation: Raw & Baseline-Shifted Shape Fidelity ")
    log("=" * W)
    log(f"Compute Device:         {device}" + (f" ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else ""))
    log(f"Baseline Align Mode:    '{args.align_mode}' (applied identically across all 3 methods)")

    test_files_by_bin, active_data_dir = discover_test_files(args.data_dir, DEFAULT_FALLBACK_DIR, args.bin_sizes)
    if not test_files_by_bin:
        raise FileNotFoundError(
            f"No dataset files found in '{args.data_dir}' or '{DEFAULT_FALLBACK_DIR}'."
        )

    log(f"Validation Dataset:     {active_data_dir}")
    log(f"Bin Resolutions:        {sorted(test_files_by_bin.keys())}")
    for b, fp in sorted(test_files_by_bin.items()):
        log(f"  - {b:4d} bins -> {os.path.basename(os.path.dirname(fp))}/{os.path.basename(fp)}")

    # Load V3 (Round 6)
    v3_proto = PolyGaussNet3(
        poly_order=POLY_ORDER,
        filter_kernel_size=FILTER_KERNEL_SIZE,
        min_amplitude=MIN_AMPLITUDE,
        max_amplitude=MAX_AMPLITUDE,
    ).to(device)
    v3_model, v3_path, v3_epoch, v3_loss = load_model_weights(
        v3_proto,
        [args.v3_model, resolve_path(args.v3_model)],
        [args.v3_ckpt, resolve_path(args.v3_ckpt)],
        device,
    )
    v3_params = sum(p.numel() for p in v3_model.parameters())

    # Load V3.4-8sec (Round 12)
    v3_4_proto = PolyGaussNet3_4(
        poly_order=POLY_ORDER,
        filter_kernel_size=FILTER_KERNEL_SIZE,
        min_amplitude=MIN_AMPLITUDE,
        max_amplitude=MAX_AMPLITUDE,
        baseline_channels=V3_4_BASELINE_CH,
        filter_channels=V3_4_FILTER_CH,
        num_quadrants=V3_4_QUADRANTS,
    ).to(device)
    v3_4_model, v3_4_path, v3_4_epoch, v3_4_loss = load_model_weights(
        v3_4_proto,
        [resolve_output_path(args.v3_4_model, args.output_dir), resolve_path(args.v3_4_model),
         resolve_output_path(args.v3_4_ckpt, args.output_dir), v3_4_ckpt_resolved],
        [resolve_output_path(args.v3_4_ckpt, args.output_dir), v3_4_ckpt_resolved],
        device,
    )
    v3_4_params = sum(p.numel() for p in v3_4_model.parameters())

    if args.require_epoch > 0 and (v3_4_epoch is not None and v3_4_epoch < args.require_epoch):
        print(f"[DEFERRED] V3.4-8sec loaded epoch {v3_4_epoch} < required {args.require_epoch}.")
        return 0

    log("\n" + "-" * W)
    log("Models Loaded:")
    v3_meta = f"Epoch {v3_epoch}, Train Loss {v3_loss:.6f}" if v3_epoch is not None else "Metadata N/A"
    v3_4_meta = f"Epoch {v3_4_epoch}, Train Loss {v3_4_loss:.6f}" if v3_4_epoch is not None else "Metadata N/A"
    log(f"  1. Conventional Pipeline: test_files/conventional/pre.py (CPU workers={args.n_jobs if not args.no_conv else 0})")
    log(f"  2. V3 (Round 6):          {v3_path} ({v3_params:,} params | {v3_meta})")
    log(f"  3. V3.4-8sec (Round 12):  {v3_4_path} ({v3_4_params:,} params | {v3_4_meta})")
    log("-" * W + "\n")

    total_spectra = 0
    total_times = {"conv": 0.0, "v3": 0.0, "v3_4": 0.0}
    acc = {
        "conv_raw": [], "conv_shifted": [],
        "v3_raw": [], "v3_shifted": [], "v3_base_mse": [],
        "v3_4_raw": [], "v3_4_shifted": [], "v3_4_base_mse": [],
    }
    overshoot = {"v3_cnt": 0, "v3_bins": 0, "v3_4_cnt": 0, "v3_4_bins": 0}
    per_res = {}
    sample_overlays = []

    sorted_bins = sorted(test_files_by_bin.keys())
    for idx, b in enumerate(sorted_bins, 1):
        with np.load(test_files_by_bin[b]) as data:
            raw_mat = data["full_matrix"]
            pure_mat = data["pure_matrix"]
            if "baseline_matrix" in data:
                true_base = data["baseline_matrix"]
            elif "pure_noise_cosmic_matrix" in data:
                true_base = raw_mat - data["pure_noise_cosmic_matrix"]
            else:
                true_base = None

        if args.max_samples_per_chunk is not None and args.max_samples_per_chunk < len(raw_mat):
            raw_mat = raw_mat[: args.max_samples_per_chunk]
            pure_mat = pure_mat[: args.max_samples_per_chunk]
            if true_base is not None:
                true_base = true_base[: args.max_samples_per_chunk]

        n_samples, seq_len = raw_mat.shape
        total_spectra += n_samples
        per_res[b] = {"n": n_samples}

        # 1. Conventional Pipeline
        conv_raw = conv_shf = None
        if not args.no_conv:
            wn_mat = np.tile(np.arange(seq_len, dtype=np.float32), (n_samples, 1))
            t0_c = time.perf_counter()
            conv_raw = run_conventional_pipeline(raw_mat, wn_mat, n_jobs=args.n_jobs)
            total_times["conv"] += time.perf_counter() - t0_c

            mc_r = compute_metrics(pure_mat, conv_raw)
            acc["conv_raw"].append(mc_r["_arrays"])
            per_res[b]["conv_raw"] = mc_r["_arrays"]

            if not args.no_shift:
                conv_shf = align_baseline(pure_mat, conv_raw, mode=args.align_mode)
                mc_s = compute_metrics(pure_mat, conv_shf)
                acc["conv_shifted"].append(mc_s["_arrays"])
                per_res[b]["conv_shifted"] = mc_s["_arrays"]

        # 2. V3 (Round 6)
        v3_clean, v3_base, t_v3 = run_nn_inference(v3_model, raw_mat, device, args.batch_size)
        total_times["v3"] += t_v3
        overshoot["v3_cnt"] += int((v3_base > raw_mat).sum())
        overshoot["v3_bins"] += v3_base.size

        mv3_r = compute_metrics(pure_mat, v3_clean)
        acc["v3_raw"].append(mv3_r["_arrays"])
        per_res[b]["v3_raw"] = mv3_r["_arrays"]
        if true_base is not None:
            per_res[b]["v3_base_mse"] = np.mean((v3_base - true_base) ** 2, axis=1)
            acc["v3_base_mse"].append(per_res[b]["v3_base_mse"])

        v3_shf = None
        if not args.no_shift:
            v3_shf = align_baseline(pure_mat, v3_clean, mode=args.align_mode)
            mv3_s = compute_metrics(pure_mat, v3_shf)
            acc["v3_shifted"].append(mv3_s["_arrays"])
            per_res[b]["v3_shifted"] = mv3_s["_arrays"]

        # 3. V3.4-8sec (Round 12)
        v3_4_clean, v3_4_base, t_v3_4 = run_nn_inference(v3_4_model, raw_mat, device, args.batch_size)
        total_times["v3_4"] += t_v3_4
        overshoot["v3_4_cnt"] += int((v3_4_base > raw_mat).sum())
        overshoot["v3_4_bins"] += v3_4_base.size

        mv34_r = compute_metrics(pure_mat, v3_4_clean)
        acc["v3_4_raw"].append(mv34_r["_arrays"])
        per_res[b]["v3_4_raw"] = mv34_r["_arrays"]
        if true_base is not None:
            per_res[b]["v3_4_base_mse"] = np.mean((v3_4_base - true_base) ** 2, axis=1)
            acc["v3_4_base_mse"].append(per_res[b]["v3_4_base_mse"])

        v3_4_shf = None
        if not args.no_shift:
            v3_4_shf = align_baseline(pure_mat, v3_4_clean, mode=args.align_mode)
            mv34_s = compute_metrics(pure_mat, v3_4_shf)
            acc["v3_4_shifted"].append(mv34_s["_arrays"])
            per_res[b]["v3_4_shifted"] = mv34_s["_arrays"]

        sample_overlays.append({
            "bin_size": b,
            "raw": raw_mat[0],
            "pure": pure_mat[0],
            "conv_raw": conv_raw[0] if conv_raw is not None else None,
            "conv_shifted": conv_shf[0] if conv_shf is not None else None,
            "v3_raw": v3_clean[0],
            "v3_shifted": v3_shf[0] if v3_shf is not None else None,
            "v3_4_raw": v3_4_clean[0],
            "v3_4_shifted": v3_4_shf[0] if v3_4_shf is not None else None,
            "v3_mse_s": float(np.mean(per_res[b].get("v3_shifted", per_res[b]["v3_raw"])["mse"])),
            "v3_4_mse_s": float(np.mean(per_res[b].get("v3_4_shifted", per_res[b]["v3_4_raw"])["mse"])),
            "conv_mse_s": float(np.mean(per_res[b]["conv_shifted"]["mse"])) if "conv_shifted" in per_res[b] else None,
        })

        v3_s_val = np.mean(per_res[b]["v3_shifted"]["mse"]) if not args.no_shift else np.mean(per_res[b]["v3_raw"]["mse"])
        v34_s_val = np.mean(per_res[b]["v3_4_shifted"]["mse"]) if not args.no_shift else np.mean(per_res[b]["v3_4_raw"]["mse"])
        c_str = f"Conv={np.mean(per_res[b]['conv_shifted']['mse']):.5f}, " if "conv_shifted" in per_res[b] else ""
        log(
            f"[{idx}/{len(sorted_bins)}] L={b:4d} (N={n_samples:,}) | "
            f"RAW MSE: V3={np.mean(per_res[b]['v3_raw']['mse']):.5f}, V3.4-8={np.mean(per_res[b]['v3_4_raw']['mse']):.5f}"
            + (f", Conv={np.mean(per_res[b]['conv_raw']['mse']):.5f}" if "conv_raw" in per_res[b] else "")
            + f" | SHIFTED MSE: {c_str}V3={v3_s_val:.5f}, V3.4-8={v34_s_val:.5f}"
        )

    # Aggregate Global Metrics
    overall = {
        "conv_raw": aggregate_metrics(acc["conv_raw"]) if not args.no_conv else None,
        "conv_shifted": aggregate_metrics(acc["conv_shifted"]) if (not args.no_conv and not args.no_shift) else None,
        "v3_raw": aggregate_metrics(acc["v3_raw"]),
        "v3_shifted": aggregate_metrics(acc["v3_shifted"]) if not args.no_shift else None,
        "v3_4_raw": aggregate_metrics(acc["v3_4_raw"]),
        "v3_4_shifted": aggregate_metrics(acc["v3_4_shifted"]) if not args.no_shift else None,
    }

    log("\n" + "=" * W)
    log(f"{f'HEAD-TO-HEAD BENCHMARK REPORT (N = {total_spectra:,} Spectra)':^{W}}")
    log("=" * W)

    metric_specs = [
        ("Cosine Similarity", "Cosine Similarity (↑)", True),
        ("MSE", "Mean Squared Error (MSE ↓)", False),
        ("Log-Cosh", "Log-Cosh Loss (↓)", False),
        ("MAE", "Mean Absolute Error (MAE ↓)", False),
        ("RMSE", "Root Mean Sq. Error (RMSE ↓)", False),
        ("Peak Summit MAE", "Peak Summit MAE (↓)", False),
    ]

    # PART A: STANDARD RAW VALIDATION
    log("\n" + "-" * W)
    log("  PART A: STANDARD RAW VALIDATION (UNSHIFTED PREDICTIONS)")
    log("-" * W)
    log(
        f"{'Metric':<28} | {'Conventional (pre.py)':<21} | {'V3 (Round 6)':<21} | "
        f"{'V3.4-8sec (R12)':<21} | {'Δ V3.4 vs V3':>12} | {'Δ V3.4 vs Conv':>14}"
    )
    log("-" * W)

    for m_key, m_label, _ in metric_specs:
        c_vs = overall["conv_raw"][m_key] if overall["conv_raw"] else None
        v3_vs = overall["v3_raw"][m_key]
        v34_vs = overall["v3_4_raw"][m_key]
        d_v3 = fmt_delta(m_label, v34_vs[0], v3_vs[0])
        d_conv = fmt_delta(m_label, v34_vs[0], c_vs[0] if c_vs else None)
        log(
            f"{m_label:<28} | {fmt_cell(c_vs):<21} | {fmt_cell(v3_vs):<21} | "
            f"{fmt_cell(v34_vs):<21} | {d_v3:>12} | {d_conv:>14}"
        )

    if acc["v3_base_mse"] and acc["v3_4_base_mse"]:
        b_v3_all = np.concatenate(acc["v3_base_mse"])
        b_v34_all = np.concatenate(acc["v3_4_base_mse"])
        b_v3_vs = (float(np.mean(b_v3_all)), float(np.std(b_v3_all)))
        b_v34_vs = (float(np.mean(b_v34_all)), float(np.std(b_v34_all)))
        log(
            f"{'Baseline Est. MSE (↓)':<28} | {'N/A':<21} | {fmt_cell(b_v3_vs):<21} | "
            f"{fmt_cell(b_v34_vs):<21} | {fmt_delta('MSE', b_v34_vs[0], b_v3_vs[0]):>12} | {'N/A':>14}"
        )

    v3_over = overshoot["v3_cnt"] / overshoot["v3_bins"] * 100.0
    v34_over = overshoot["v3_4_cnt"] / overshoot["v3_4_bins"] * 100.0
    log(
        f"{'Baseline Overshoot Rate':<28} | {'N/A':<21} | {f'{v3_over:.3f}% of bins':<21} | "
        f"{f'{v34_over:.3f}% of bins':<21} | {f'{v34_over - v3_over:+.3f}p':>12} | {'N/A':>14}"
    )

    # PART B: BASELINE-SHIFTED VALIDATION
    if not args.no_shift:
        log("\n" + "-" * W)
        log("  PART B: BASELINE-SHIFTED VALIDATION (PURE SPECTRAL SHAPE & PEAK RECONSTRUCTION FIDELITY)")
        log(f"  [Identical `{args.align_mode}` baseline alignment shift applied to all 3 methods]")
        log("-" * W)
        log(
            f"{'Metric':<28} | {'Conventional (Shift)':<21} | {'V3 (R6 Shift)':<21} | "
            f"{'V3.4-8sec (Shift)':<21} | {'Δ V3.4 vs V3':>12} | {'Winner':>14}"
        )
        log("-" * W)
        for m_key, m_label, higher_better in metric_specs:
            c_vs = overall["conv_shifted"][m_key] if overall["conv_shifted"] else None
            v3_vs = overall["v3_shifted"][m_key]
            v34_vs = overall["v3_4_shifted"][m_key]
            cands = [("V3 (R6)", v3_vs[0]), ("V3.4-8sec", v34_vs[0])]
            if c_vs is not None:
                cands.append(("Conv", c_vs[0]))
            winner = (max if higher_better else min)(cands, key=lambda kv: kv[1])[0]
            log(
                f"{m_label:<28} | {fmt_cell(c_vs):<21} | {fmt_cell(v3_vs):<21} | "
                f"{fmt_cell(v34_vs):<21} | {fmt_delta(m_label, v34_vs[0], v3_vs[0]):>12} | {winner:>14}"
            )

    # PART C: SPEED & THROUGHPUT
    log("\n" + "-" * W)
    log("  PART C: COMPUTATIONAL SPEED & THROUGHPUT")
    log("-" * W)
    log(f"{'Metric':<28} | {'Conventional (CPU)':<21} | {'V3 (Round 6)':<21} | {'V3.4-8sec (R12)':<21} | {'Speedup vs Conv':>28}")
    log("-" * W)

    def _tput(t):
        return total_spectra / t if t > 0 else 0.0

    def _lat(t):
        return (t / total_spectra) * 1000.0 if t > 0 else 0.0

    tc, tv3, tv34 = total_times["conv"], total_times["v3"], total_times["v3_4"]
    spd_v34 = f"V3.4 {tc / tv34:.1f}x faster" if (tc > 0 and tv34 > 0) else "N/A"
    log(f"{'Total Inference Time':<28} | {f'{tc:8.3f} s' if tc > 0 else 'N/A':<21} | {f'{tv3:8.3f} s':<21} | {f'{tv34:8.3f} s':<21} | {spd_v34:>28}")
    log(f"{'Throughput (spectra/s)':<28} | {f'{_tput(tc):8.1f} s/s' if tc > 0 else 'N/A':<21} | {f'{_tput(tv3):8.1f} s/s':<21} | {f'{_tput(tv34):8.1f} s/s':<21} | {'':>28}")
    log(f"{'Latency (ms/spectrum)':<28} | {f'{_lat(tc):8.4f} ms' if tc > 0 else 'N/A':<21} | {f'{_lat(tv3):8.4f} ms':<21} | {f'{_lat(tv34):8.4f} ms':<21} | {'':>28}")
    log("=" * W)

    # PART D: RESOLUTION BREAKDOWN
    log("\n" + "=" * W)
    log(f"{'PART D: SPECTRAL BIN RESOLUTION BREAKDOWN (800 to 1600 Bins)':^{W}}")
    log("=" * W)
    log(
        f"{'Bins':<6} | {'N':<6} | {'Conv Raw':<10} | {'V3 Raw':<10} | {'V3.4 Raw':<10} | "
        f"{'Conv Shf':<10} | {'V3 Shf':<10} | {'V3.4 Shf':<10} | {'Δ Shf V3.4vV3':>13} | {'Best (Shf)':>10}"
    )
    log("-" * W)

    wins_vs_v3 = {"raw_mse": 0, "shf_mse": 0, "cos": 0, "base_mse": 0}
    wins_vs_conv = {"raw_mse": 0, "shf_mse": 0, "cos": 0}

    for b in sorted_bins:
        r = per_res[b]
        n = r["n"]
        v3_r = float(np.mean(r["v3_raw"]["mse"]))
        v34_r = float(np.mean(r["v3_4_raw"]["mse"]))
        c_r_val = float(np.mean(r["conv_raw"]["mse"])) if "conv_raw" in r else None
        c_r_str = f"{c_r_val:.6f}" if c_r_val is not None else "N/A"

        if v34_r < v3_r:
            wins_vs_v3["raw_mse"] += 1
        if c_r_val is not None and v34_r < c_r_val:
            wins_vs_conv["raw_mse"] += 1
        if float(np.mean(r["v3_4_raw"]["cosine_sim"])) > float(np.mean(r["v3_raw"]["cosine_sim"])):
            wins_vs_v3["cos"] += 1
        if "conv_raw" in r and float(np.mean(r["v3_4_raw"]["cosine_sim"])) > float(np.mean(r["conv_raw"]["cosine_sim"])):
            wins_vs_conv["cos"] += 1
        if "v3_4_base_mse" in r and "v3_base_mse" in r:
            if float(np.mean(r["v3_4_base_mse"])) < float(np.mean(r["v3_base_mse"])):
                wins_vs_v3["base_mse"] += 1

        if not args.no_shift:
            v3_s = float(np.mean(r["v3_shifted"]["mse"]))
            v34_s = float(np.mean(r["v3_4_shifted"]["mse"]))
            c_s_val = float(np.mean(r["conv_shifted"]["mse"])) if "conv_shifted" in r else None
            c_s_str = f"{c_s_val:.6f}" if c_s_val is not None else "N/A"
            if v34_s < v3_s:
                wins_vs_v3["shf_mse"] += 1
            if c_s_val is not None and v34_s < c_s_val:
                wins_vs_conv["shf_mse"] += 1
            d_shf = (v34_s - v3_s) / v3_s * 100.0
            cands = [("V3", v3_s), ("V3.4-8", v34_s)]
            if c_s_val is not None:
                cands.append(("Conv", c_s_val))
            best = min(cands, key=lambda kv: kv[1])[0]
            log(
                f"{b:<6} | {n:<6} | {c_r_str:<10} | {v3_r:<10.6f} | {v34_r:<10.6f} | "
                f"{c_s_str:<10} | {v3_s:<10.6f} | {v34_s:<10.6f} | {d_shf:>+12.2f}% | {best:>10}"
            )
        else:
            best = min([("V3", v3_r), ("V3.4-8", v34_r)] + ([("Conv", c_r_val)] if c_r_val is not None else []), key=lambda kv: kv[1])[0]
            log(f"{b:<6} | {n:<6} | {c_r_str:<10} | {v3_r:<10.6f} | {v34_r:<10.6f} | {'N/A':<10} | {'N/A':<10} | {'N/A':<10} | {'N/A':>13} | {best:>10}")

    log("-" * W)
    log("\nCosine Similarity & Log-Cosh by Resolution (Raw):")
    log(f"{'Bins':<6} | {'Conv Cos':<10} | {'V3 Cos':<10} | {'V3.4 Cos':<10} | {'Conv LC':<10} | {'V3 LC':<10} | {'V3.4 LC':<10}")
    log("-" * 80)
    for b in sorted_bins:
        r = per_res[b]
        cc = f"{np.mean(r['conv_raw']['cosine_sim']):.5f}" if "conv_raw" in r else "N/A"
        cl = f"{np.mean(r['conv_raw']['log_cosh']):.6f}" if "conv_raw" in r else "N/A"
        log(
            f"{b:<6} | {cc:<10} | {np.mean(r['v3_raw']['cosine_sim']):<10.5f} | {np.mean(r['v3_4_raw']['cosine_sim']):<10.5f} | "
            f"{cl:<10} | {np.mean(r['v3_raw']['log_cosh']):<10.6f} | {np.mean(r['v3_4_raw']['log_cosh']):<10.6f}"
        )

    # PART E: RESOLUTION WIN SUMMARY
    nb = len(sorted_bins)
    log("\n" + "=" * W)
    log(f" PART E: PER-RESOLUTION WIN COUNTS FOR V3.4-8sec ({nb} Resolutions: {sorted_bins})")
    log("=" * W)
    log(
        f"  V3.4-8sec vs V3 (Round 6):       Raw MSE: {wins_vs_v3['raw_mse']}/{nb} | "
        f"Shifted MSE: {wins_vs_v3['shf_mse']}/{nb} | Raw Cosine: {wins_vs_v3['cos']}/{nb}"
        + (f" | Baseline MSE: {wins_vs_v3['base_mse']}/{nb}" if acc["v3_4_base_mse"] else "")
    )
    if not args.no_conv:
        log(
            f"  V3.4-8sec vs Conventional:       Raw MSE: {wins_vs_conv['raw_mse']}/{nb} | "
            f"Shifted MSE: {wins_vs_conv['shf_mse']}/{nb} | Raw Cosine: {wins_vs_conv['cos']}/{nb}"
        )
    log("=" * W + "\n")

    # Save report atomically (archiving prior report if present)
    report_path = resolve_output_path(args.report_path, args.output_dir)
    if os.path.exists(report_path):
        shutil.copyfile(report_path, report_path.replace(".txt", "_prev.txt"))
    tmp_report = f"{report_path}.tmp.{os.getpid()}"
    with open(tmp_report, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    os.replace(tmp_report, report_path)
    print(f"Report saved to: {report_path}")

    if not args.no_plots:
        p1, p2 = generate_plots(sample_overlays, overall, args.output_dir, not args.no_shift)
        print(f"Plots saved to:  {p1}\n                 {p2}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
