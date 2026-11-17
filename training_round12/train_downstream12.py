"""
DOWNSTREAM CLASSIFICATION TRAINING SCRIPT - ROUND 12
----------------------------------------------------
Trains 3 downstream spectral classifiers across 2 real-world Raman datasets and
3 preprocessing methods (3 x 2 x 3 = 18 total runs) under an identical architecture
and matched-epoch convergence regime:

  Datasets:
    1. Bacteria-ID (`datasets/bacteria-ID/`):
       - Official splits: `reference` (60,000 train), `finetune` (3,000 val), `test` (3,000 test)
       - 30 bacterial isolate classes (clinical data excluded)
    2. MLROD (`datasets/MLROD/`):
       - Filtered to the 12 pure single-mineral classes (excluding `Granite slab - 50% dust`
         and the three `50% / 50% Mix` classes; 109,655 spectra across 1,836 wavenumber bins)
       - Deterministic stratified 65% train / 15% val / 20% test split (`random_state=42`)

  Upfront Normalization:
    - Every spectrum in both datasets is normalized to `[0, 1]` via per-spectrum min-max
      scaling BEFORE any preprocessing, training, or inference.

  Preprocessors (evaluated in strict order):
    1. `raw`          : Upfront [0, 1] min-max normalized spectra (no further preprocessing)
    2. `conventional` : CPU pipeline (`test_files/conventional/pre.py`: despike -> denoise -> baseline)
    3. `pgnet_v34`    : PolyGaussNet V3.4 (8-Section, `training_round12/polygaussnet12.pth`, 108,394 params)

  Classifiers & Fair Matched-Epoch Regime:
    Within each (dataset, classifier) pair, `raw` is trained FIRST to convergence using
    validation patience (`patience=7` for ResNet1D, `early_stopping_rounds=30` for XGBoost).
    The exact stopping epoch/round count `E_stop` reached by `raw` is then fixed for
    `conventional` and `pgnet_v34` (with patience disabled) so all 3 preprocessors receive
    the exact same number of training epochs / boosting rounds:
      1. `linear`   : `LinearSpectralClassifier` (`GlobalRangeScaler` + `LogisticRegression`)
      2. `xgb`      : `XGBSpectralClassifier` (`XGBClassifier`, `tree_method="hist"`)
      3. `resnet1d` : `BacteriaResNetClassifier` (`ResNet1D`, Ho et al. 2019 architecture)

  Outputs Saved:
    - `training_round12/downstream_logs/{dataset}_{classifier}_{prep}_training_history12.txt` (& `.csv`)
    - `training_round12/downstream_logs/training_summary12.txt` & `training_manifest12.json`
    - `training_round12/downstream_models/` (weights + metadata for all 18 runs)
    - `training_round12/downstream_cache/` (preprocessed val/test splits + split indices for fast local testing)
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import signal
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

# Configure paths so imports resolve whether invoked from repo root or training_round12/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
CONVENTIONAL_DIR = os.path.join(PROJECT_ROOT, "test_files", "conventional")
RAMAN_PLOTTING_SIBLING = os.path.abspath(os.path.join(PROJECT_ROOT, "..", "raman plotting"))

for p in (PROJECT_ROOT, SCRIPT_DIR, CONVENTIONAL_DIR, RAMAN_PLOTTING_SIBLING):
    if os.path.exists(p) and p not in sys.path:
        sys.path.insert(0, p)

from classifiers.linear import LinearSpectralClassifier
from classifiers.resnet1d import ResNet1D
from polygaussnet3_4 import PolyGaussNet as PolyGaussNet3_4

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None


# =====================================================================
# 1. CONSTANTS & DEFAULTS
# =====================================================================
DATASETS_ORDER = ("bacteria_id", "mlrod")
PREPROCESSORS_ORDER = ("raw", "conventional", "pgnet_v34")
CLASSIFIERS_ORDER = ("linear", "xgb", "resnet1d")

# Pure single-mineral class IDs in MLROD (excludes 0,1,2 mixes and 8 Granite slab)
MLROD_PURE12_IDS = (3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15)
MLROD_NONSLAB15_IDS = (0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15)

# PolyGaussNet V3.4-8sec exact Round 12 configuration
POLY_ORDER = 7
FILTER_KERNEL_SIZE = 31
MIN_AMPLITUDE = 0.85
MAX_AMPLITUDE = 1.15
V3_4_BASELINE_CH = 32
V3_4_FILTER_CH = 32
V3_4_QUADRANTS = 8

BACTERIA_SPECIES_NAMES = [
    "C. albicans", "C. glabrata", "K. aerogenes", "E. coli 1", "E. coli 2",
    "E. faecium", "E. faecalis 1", "E. faecalis 2", "E. cloacae", "K. pneumoniae 1",
    "K. pneumoniae 2", "B. fragilis", "S. epidermidis", "P. mirabilis", "P. aeruginosa 1",
    "P. aeruginosa 2", "M.SSA 1", "M.SSA 2", "M.SSA 3", "M.RSA 1", "M.RSA 2",
    "S. enterica", "S. marcescens", "S. pneumoniae 1", "S. pneumoniae 2", "S. sanguinis",
    "S. agalactiae", "S. dysgalactiae", "S. pyogenes", "S. lugdunensis",
]


# =====================================================================
# 2. PATH & LOGGING UTILITIES
# =====================================================================
def resolve_path(path: str) -> str:
    if os.path.isabs(path) or os.path.exists(path):
        return os.path.abspath(path)
    candidate_root = os.path.join(PROJECT_ROOT, path)
    if os.path.exists(candidate_root):
        return os.path.abspath(candidate_root)
    candidate_script = os.path.join(SCRIPT_DIR, path)
    if os.path.exists(candidate_script):
        return os.path.abspath(candidate_script)
    return os.path.abspath(candidate_root)


class DualLogger:
    """Writes simultaneously to stdout and a line-buffered .txt log file."""

    def __init__(self, filepath: str, append: bool = False) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        self.filepath = filepath
        mode = "a" if (append and os.path.exists(filepath)) else "w"
        self.fp = open(filepath, mode, encoding="utf-8", buffering=1)

    def log(self, msg: str = "") -> None:
        print(msg, flush=True)
        self.fp.write(msg + "\n")
        self.fp.flush()

    def close(self) -> None:
        if self.fp and not self.fp.closed:
            self.fp.close()


def atomic_json_dump(payload: Dict[str, Any], filepath: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    tmp_path = filepath + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, filepath)


def atomic_torch_save(payload: Any, filepath: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    tmp_path = filepath + ".tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, filepath)


def atomic_joblib_dump(payload: Any, filepath: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    tmp_path = filepath + ".tmp"
    joblib.dump(payload, tmp_path)
    os.replace(tmp_path, filepath)


# =====================================================================
# 3. NORMALIZATION & DATASET LOADERS
# =====================================================================
def normalize_per_spectrum_01(X: np.ndarray) -> np.ndarray:
    """Per-spectrum [0, 1] min-max normalization applied before any preprocessing."""
    arr = np.asarray(X, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array (N, L), got shape {arr.shape}")
    mins = np.min(arr, axis=1, keepdims=True)
    maxs = np.max(arr, axis=1, keepdims=True)
    spans = maxs - mins
    spans = np.where(spans > 1e-12, spans, 1.0)
    out = (arr - mins) / spans
    return np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)


def _subsample_stratified(
    X: np.ndarray, y: np.ndarray, max_per_class: int, seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    kept_indices: List[int] = []
    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        if len(cls_idx) > max_per_class:
            chosen = rng.choice(cls_idx, size=max_per_class, replace=False)
            chosen = np.sort(chosen)
        else:
            chosen = cls_idx
        kept_indices.extend(chosen.tolist())
    idx_arr = np.array(kept_indices, dtype=np.int64)
    return X[idx_arr], y[idx_arr], idx_arr


def load_bacteria_id_dataset(
    data_dir: str,
    seed: int = 42,
    smoke_test: bool = False,
) -> Dict[str, Any]:
    """Load Bacteria-ID reference (train), finetune (val), and test splits (no clinical)."""
    resolved_dir = resolve_path(data_dir)
    req_files = [
        "X_reference.npy", "y_reference.npy",
        "X_finetune.npy", "y_finetune.npy",
        "X_test.npy", "y_test.npy",
        "wavenumbers.npy",
    ]
    for fname in req_files:
        fpath = os.path.join(resolved_dir, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(
                f"Missing Bacteria-ID file '{fpath}'. Ensure datasets/bacteria-ID is synced."
            )

    X_train = np.load(os.path.join(resolved_dir, "X_reference.npy"))
    y_train = np.load(os.path.join(resolved_dir, "y_reference.npy")).astype(np.int64)
    X_val = np.load(os.path.join(resolved_dir, "X_finetune.npy"))
    y_val = np.load(os.path.join(resolved_dir, "y_finetune.npy")).astype(np.int64)
    X_test = np.load(os.path.join(resolved_dir, "X_test.npy"))
    y_test = np.load(os.path.join(resolved_dir, "y_test.npy")).astype(np.int64)
    wavenumbers = np.load(os.path.join(resolved_dir, "wavenumbers.npy")).astype(np.float64).reshape(-1)

    if smoke_test:
        X_train, y_train, _ = _subsample_stratified(X_train, y_train, max_per_class=12, seed=seed)
        X_val, y_val, _ = _subsample_stratified(X_val, y_val, max_per_class=4, seed=seed)
        X_test, y_test, _ = _subsample_stratified(X_test, y_test, max_per_class=4, seed=seed)

    # Apply upfront [0, 1] min-max normalization to ALL spectra before any preprocessing
    X_train = normalize_per_spectrum_01(X_train)
    X_val = normalize_per_spectrum_01(X_val)
    X_test = normalize_per_spectrum_01(X_test)

    n_classes = int(len(np.unique(y_train)))
    if len(BACTERIA_SPECIES_NAMES) == n_classes:
        class_names = [f"{i:02d}_{name}" for i, name in enumerate(BACTERIA_SPECIES_NAMES)]
    else:
        class_names = [f"Isolate_{i:02d}" for i in range(n_classes)]

    return {
        "name": "bacteria_id",
        "display_name": "Bacteria-ID (Isolates: Reference / Finetune / Test, Clinical Excluded)",
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "wavenumbers": wavenumbers,
        "class_names": class_names,
        "split_info": {
            "protocol": "official_reference_finetune_test_no_clinical",
            "upfront_normalization": "per_spectrum_min_max_0_1",
            "smoke_test": smoke_test,
        },
    }


def load_mlrod_dataset(
    data_dir: str,
    subset_mode: str = "pure12",
    val_size: float = 0.15,
    test_size: float = 0.20,
    split_seed: int = 42,
    smoke_test: bool = False,
) -> Dict[str, Any]:
    """Load MLROD, filter to pure single-mineral classes, normalize [0, 1], and split 65/15/20."""
    resolved_dir = resolve_path(data_dir)
    for fname in ("X.npy", "y.npy", "wavenumbers.npy", "class_names.npy"):
        fpath = os.path.join(resolved_dir, fname)
        if not os.path.exists(fpath):
            raise FileNotFoundError(
                f"Missing MLROD file '{fpath}'. Ensure datasets/MLROD is synced."
            )

    X_all = np.load(os.path.join(resolved_dir, "X.npy"), mmap_mode="r")
    y_all = np.load(os.path.join(resolved_dir, "y.npy")).astype(np.int64)
    wavenumbers = np.load(os.path.join(resolved_dir, "wavenumbers.npy")).astype(np.float64).reshape(-1)
    raw_class_names = [str(c) for c in np.load(os.path.join(resolved_dir, "class_names.npy")).tolist()]

    if subset_mode == "pure12":
        keep_ids = list(MLROD_PURE12_IDS)
    elif subset_mode == "nonslab15":
        keep_ids = list(MLROD_NONSLAB15_IDS)
    else:
        raise ValueError(f"Unknown --mlrod-subset '{subset_mode}'. Choose 'pure12' or 'nonslab15'.")

    mask = np.isin(y_all, keep_ids)
    kept_global_indices = np.where(mask)[0]
    y_filtered_orig = y_all[kept_global_indices]

    # Map original class IDs to contiguous 0..C-1
    orig_to_new = {orig_id: new_id for new_id, orig_id in enumerate(keep_ids)}
    y_filtered = np.array([orig_to_new[int(c)] for c in y_filtered_orig], dtype=np.int64)
    class_names = [raw_class_names[orig_id] for orig_id in keep_ids]

    if smoke_test:
        _, y_sub, sub_rel_idx = _subsample_stratified(
            np.zeros((len(y_filtered), 1), dtype=np.float32),
            y_filtered,
            max_per_class=30,
            seed=split_seed,
        )
        kept_global_indices = kept_global_indices[sub_rel_idx]
        y_filtered = y_sub

    X_filtered = np.asarray(X_all[kept_global_indices], dtype=np.float32)

    # Apply upfront [0, 1] min-max normalization to ALL spectra before splitting/preprocessing
    X_filtered = normalize_per_spectrum_01(X_filtered)

    # Deterministic stratified 65% train / 15% val / 20% test split (matching benchmark.py)
    rel_indices = np.arange(len(y_filtered), dtype=np.int64)
    idx_train_val, idx_test, y_train_val, y_test = train_test_split(
        rel_indices,
        y_filtered,
        test_size=test_size,
        stratify=y_filtered,
        random_state=split_seed,
    )
    rel_val = val_size / (1.0 - test_size)
    idx_train, idx_val, y_train, y_val = train_test_split(
        idx_train_val,
        y_train_val,
        test_size=rel_val,
        stratify=y_train_val,
        random_state=split_seed,
    )

    X_train = np.ascontiguousarray(X_filtered[idx_train], dtype=np.float32)
    X_val = np.ascontiguousarray(X_filtered[idx_val], dtype=np.float32)
    X_test = np.ascontiguousarray(X_filtered[idx_test], dtype=np.float32)

    return {
        "name": "mlrod",
        "display_name": f"MLROD ({len(keep_ids)} Pure Single-Mineral Classes, Stratified 65/15/20)",
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "wavenumbers": wavenumbers,
        "class_names": class_names,
        "split_info": {
            "protocol": "stratified_train_val_test_split",
            "subset_mode": subset_mode,
            "kept_original_class_ids": keep_ids,
            "val_size": val_size,
            "test_size": test_size,
            "split_seed": split_seed,
            "upfront_normalization": "per_spectrum_min_max_0_1",
            "global_train_indices": kept_global_indices[idx_train],
            "global_val_indices": kept_global_indices[idx_val],
            "global_test_indices": kept_global_indices[idx_test],
            "smoke_test": smoke_test,
        },
    }


# =====================================================================
# 4. PREPROCESSING PIPELINES (RAW, CONVENTIONAL, PGNET V3.4-8SEC)
# =====================================================================
def _init_conv_worker() -> None:
    """Prevent BLAS thread oversubscription inside multiprocessing workers."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"


def _process_conv_chunk(wavenumbers_1d: np.ndarray, raw_chunk: np.ndarray) -> np.ndarray:
    from spectra_class import spectra
    import pre

    wn_mat = np.broadcast_to(wavenumbers_1d.reshape(1, -1), raw_chunk.shape).copy()
    conv_data = spectra.from_matrices(wn_mat, np.array(raw_chunk, dtype=np.float64, copy=True))
    pre.preprocess_pipeline(conv_data, normalize=False, shift=False)
    out = np.asarray(conv_data.intensity_matrix, dtype=np.float32)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def run_conventional_preprocessing(
    X: np.ndarray, wavenumbers_1d: np.ndarray, n_jobs: int
) -> Tuple[np.ndarray, float]:
    """Run conventional CPU preprocessing (`pre.py`) across `n_jobs` processes."""
    try:
        import spectra_class  # noqa: F401
        import pre  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "Conventional pipeline dependencies (`spectra_class` / `pre`) are missing. "
            "Ensure `test_files/conventional` and `Raman-Plotting-Pipeline` are installed."
        ) from exc

    t0 = time.perf_counter()
    n_samples = len(X)
    if n_jobs <= 1 or n_samples < 32:
        out = _process_conv_chunk(wavenumbers_1d, X)
        return out, time.perf_counter() - t0

    # Use finer chunks so progress is balanced across workers
    num_chunks = max(n_jobs * 2, 1)
    chunk_size = int(np.ceil(n_samples / num_chunks))
    chunks = [
        X[i : min(i + chunk_size, n_samples)]
        for i in range(0, n_samples, chunk_size)
    ]

    results: List[Optional[np.ndarray]] = [None] * len(chunks)
    with ProcessPoolExecutor(max_workers=n_jobs, initializer=_init_conv_worker) as pool:
        futures = {
            pool.submit(_process_conv_chunk, wavenumbers_1d, chunk): idx
            for idx, chunk in enumerate(chunks)
        }
        for fut in as_completed(futures):
            results[futures[fut]] = fut.result()

    out = np.vstack([r for r in results if r is not None]).astype(np.float32)
    return out, time.perf_counter() - t0


def load_pgnet_v34_model(
    model_path: str, ckpt_path: str, device: torch.device
) -> Tuple[nn.Module, str, Optional[int], Optional[float]]:
    """Instantiate and load PolyGaussNet V3.4 (8-Section) weights."""
    model = PolyGaussNet3_4(
        poly_order=POLY_ORDER,
        filter_kernel_size=FILTER_KERNEL_SIZE,
        min_amplitude=MIN_AMPLITUDE,
        max_amplitude=MAX_AMPLITUDE,
        baseline_channels=V3_4_BASELINE_CH,
        filter_channels=V3_4_FILTER_CH,
        num_quadrants=V3_4_QUADRANTS,
    ).to(device)

    candidates = [
        resolve_path(model_path),
        os.path.join(SCRIPT_DIR, os.path.basename(model_path)),
        resolve_path(ckpt_path),
        os.path.join(SCRIPT_DIR, os.path.basename(ckpt_path)),
    ]
    weight_file = next((p for p in candidates if p and os.path.exists(p)), None)
    if weight_file is None:
        raise FileNotFoundError(f"PolyGaussNet V3.4 weights not found. Checked: {candidates}")

    ckpt = torch.load(weight_file, map_location=device, weights_only=False)
    epoch = None
    best_loss = None
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        epoch = ckpt.get("epoch")
        best_loss = ckpt.get("best_loss", ckpt.get("loss"))
    else:
        state_dict = ckpt
        for cp in (resolve_path(ckpt_path), os.path.join(SCRIPT_DIR, os.path.basename(ckpt_path))):
            if cp and os.path.exists(cp):
                try:
                    meta = torch.load(cp, map_location="cpu", weights_only=False)
                    if isinstance(meta, dict):
                        epoch = meta.get("epoch")
                        best_loss = meta.get("best_loss", meta.get("loss"))
                        break
                except Exception:
                    pass

    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)
    model.eval()
    return model, weight_file, epoch, best_loss


def run_pgnet_v34_preprocessing(
    model: nn.Module,
    X: np.ndarray,
    device: torch.device,
    batch_size: int = 512,
) -> Tuple[np.ndarray, float]:
    """Run PolyGaussNet V3.4 inference returning `pred_clean` (denoised + baseline-removed)."""
    tensor_x = torch.from_numpy(np.ascontiguousarray(X, dtype=np.float32)).unsqueeze(1)
    if device.type == "cuda":
        try:
            tensor_x = tensor_x.pin_memory()
        except RuntimeError:
            pass

    with torch.no_grad():
        warm = tensor_x[: min(4, len(tensor_x))].to(device)
        _ = model(warm)
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()

    out_chunks: List[torch.Tensor] = []
    t0 = time.perf_counter()
    with torch.no_grad():
        for i in range(0, len(tensor_x), batch_size):
            bx = tensor_x[i : i + batch_size].to(device, non_blocking=(device.type == "cuda"))
            pred_clean, _, _, _ = model(bx)
            out_chunks.append(pred_clean.squeeze(1).cpu())
        if device.type == "cuda":
            torch.cuda.synchronize()
        elif device.type == "mps":
            torch.mps.synchronize()
    elapsed = time.perf_counter() - t0
    out_np = torch.cat(out_chunks, dim=0).numpy().astype(np.float32)
    return np.nan_to_num(out_np, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32), elapsed


def prepare_and_cache_preprocessed_splits(
    dataset: Dict[str, Any],
    cache_dir: str,
    pgnet_model: nn.Module,
    device: torch.device,
    pgnet_batch_size: int,
    n_jobs: int,
    cache_train: bool,
    use_existing_cache: bool,
    master_log: DualLogger,
) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, float]]]:
    """Compute or load preprocessed (train, val, test) splits for raw, conventional, and pgnet_v34."""
    ds_name = dataset["name"]
    os.makedirs(cache_dir, exist_ok=True)

    # Save common labels, wavenumbers, class_names, and MLROD split indices for Program 2
    np.save(os.path.join(cache_dir, f"{ds_name}_y_train.npy"), dataset["y_train"])
    np.save(os.path.join(cache_dir, f"{ds_name}_y_val.npy"), dataset["y_val"])
    np.save(os.path.join(cache_dir, f"{ds_name}_y_test.npy"), dataset["y_test"])
    np.save(os.path.join(cache_dir, f"{ds_name}_wavenumbers.npy"), dataset["wavenumbers"])
    np.save(
        os.path.join(cache_dir, f"{ds_name}_class_names.npy"),
        np.array(dataset["class_names"], dtype=object),
    )

    if ds_name == "mlrod":
        s_info = dataset["split_info"]
        np.savez_compressed(
            os.path.join(cache_dir, "mlrod_split_indices.npz"),
            global_train_indices=s_info["global_train_indices"],
            global_val_indices=s_info["global_val_indices"],
            global_test_indices=s_info["global_test_indices"],
            kept_original_class_ids=np.array(s_info["kept_original_class_ids"], dtype=np.int64),
            class_names=np.array(dataset["class_names"], dtype=object),
        )

    splits_by_prep: Dict[str, Dict[str, np.ndarray]] = {}
    timings_by_prep: Dict[str, Dict[str, float]] = {}

    raw_splits = {
        "train": dataset["X_train"],
        "val": dataset["X_val"],
        "test": dataset["X_test"],
    }

    for prep in PREPROCESSORS_ORDER:
        splits_by_prep[prep] = {}
        timings_by_prep[prep] = {}
        total_spec = 0
        total_sec = 0.0

        for split_name in ("train", "val", "test"):
            cache_file = os.path.join(cache_dir, f"{ds_name}_{split_name}_{prep}.npy")
            raw_arr = raw_splits[split_name]

            if use_existing_cache and os.path.exists(cache_file):
                cached_arr = np.load(cache_file)
                if cached_arr.shape == raw_arr.shape:
                    splits_by_prep[prep][split_name] = cached_arr.astype(np.float32, copy=False)
                    timings_by_prep[prep][f"{split_name}_sec"] = 0.0
                    master_log.log(
                        f"  [Cache Hit] {ds_name} | {prep:12s} | {split_name:5s} -> {cached_arr.shape}"
                    )
                    continue

            if prep == "raw":
                t0 = time.perf_counter()
                out_arr = np.array(raw_arr, dtype=np.float32, copy=True)
                elapsed = time.perf_counter() - t0
            elif prep == "conventional":
                out_arr, elapsed = run_conventional_preprocessing(
                    raw_arr, dataset["wavenumbers"], n_jobs=n_jobs
                )
            elif prep == "pgnet_v34":
                out_arr, elapsed = run_pgnet_v34_preprocessing(
                    pgnet_model, raw_arr, device=device, batch_size=pgnet_batch_size
                )
            else:
                raise ValueError(f"Unknown preprocessor: {prep}")

            splits_by_prep[prep][split_name] = out_arr
            timings_by_prep[prep][f"{split_name}_sec"] = float(elapsed)
            total_spec += len(out_arr)
            total_sec += elapsed

            # Always cache val and test so local Mac testing needs zero CPU conventional recomputation
            if split_name in ("val", "test") or cache_train:
                np.save(cache_file, out_arr)

            rate = len(out_arr) / max(elapsed, 1e-9)
            master_log.log(
                f"  [Preprocess] {ds_name:11s} | {prep:12s} | {split_name:5s} "
                f"({len(out_arr):6d} x {out_arr.shape[1]:4d}) | {elapsed:7.2f}s ({rate:8.1f} spec/s)"
            )

        timings_by_prep[prep]["total_sec"] = float(total_sec)
        timings_by_prep[prep]["total_spectra"] = int(total_spec)
        timings_by_prep[prep]["throughput_spec_per_s"] = float(
            total_spec / max(total_sec, 1e-9) if total_sec > 0 else 0.0
        )

    return splits_by_prep, timings_by_prep


# =====================================================================
# 5. CLASSIFIER TRAINING WITH EPOCH LOGGING & MATCHED BUDGETS
# =====================================================================
def write_history_header(
    logger: DualLogger,
    dataset_name: str,
    dataset_display: str,
    classifier_name: str,
    prep_name: str,
    shapes: Dict[str, Tuple[int, ...]],
    n_classes: int,
    regime_info: Dict[str, Any],
) -> None:
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "local")
    logger.log("=" * 92)
    logger.log(
        f" Downstream Classification Training — Round 12 ({dataset_name} | {classifier_name} | {prep_name})"
    )
    logger.log("=" * 92)
    logger.log(f"  - Slurm Job ID:         {slurm_job_id}")
    logger.log(f"  - Dataset:              {dataset_display}")
    logger.log(
        f"  - Split Shapes:         train={shapes['train']}, val={shapes['val']}, test={shapes['test']} "
        f"(classes={n_classes})"
    )
    logger.log(f"  - Upfront Norm:         Per-spectrum [0, 1] min-max scaling before preprocessing")
    logger.log(f"  - Preprocessor:         {prep_name}")
    logger.log(f"  - Classifier:           {classifier_name}")
    for k, v in regime_info.items():
        logger.log(f"  - {k + ':':22s}{v}")
    logger.log("=" * 92)
    logger.log("")


def save_history_csv(csv_path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------
# 5A. LinearSpectralClassifier (`GlobalRangeScaler` + `LogisticRegression`)
# ---------------------------------------------------------------------
def train_linear_run(
    dataset_name: str,
    dataset_display: str,
    prep_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test_shape: Tuple[int, ...],
    n_classes: int,
    C: float,
    max_iter: int,
    seed: int,
    n_jobs: int,
    log_dir: str,
    model_dir: str,
) -> Dict[str, Any]:
    run_id = f"{dataset_name}_linear_{prep_name}"
    txt_path = os.path.join(log_dir, f"{run_id}_training_history12.txt")
    csv_path = os.path.join(log_dir, f"{run_id}_training_history12.csv")
    model_path = os.path.join(model_dir, f"{run_id}.joblib")
    meta_path = os.path.join(model_dir, f"{run_id}_meta.json")

    logger = DualLogger(txt_path)
    try:
        regime_info = {
            "Architecture": "GlobalRangeScaler + LogisticRegression (multinomial L2)",
            "Hyperparameters": f"C={C}, solver='lbfgs', max_iter={max_iter}, tol=1e-4, seed={seed}",
            "Convergence Policy": f"Identical L-BFGS solver budget (max_iter={max_iter}) across preprocessors",
        }
        write_history_header(
            logger,
            dataset_name=dataset_name,
            dataset_display=dataset_display,
            classifier_name="linear (LinearSpectralClassifier)",
            prep_name=prep_name,
            shapes={"train": X_train.shape, "val": X_val.shape, "test": X_test_shape},
            n_classes=n_classes,
            regime_info=regime_info,
        )

        clf = LinearSpectralClassifier(
            C=C,
            max_iter=max_iter,
            solver="lbfgs",
            random_state=seed,
            n_jobs=n_jobs,
        )

        logger.log("Starting L-BFGS Optimization...")
        t0 = time.perf_counter()
        clf.fit(X_train, y_train)
        fit_sec = time.perf_counter() - t0

        lr_step = clf.pipeline.named_steps["clf"]
        n_iter_arr = getattr(lr_step, "n_iter_", [0])
        actual_iters = int(np.max(n_iter_arr))

        labels_all = np.arange(n_classes)
        train_probs = clf.predict_proba(X_train)
        val_probs = clf.predict_proba(X_val)

        train_loss = float(log_loss(y_train, train_probs, labels=labels_all))
        val_loss = float(log_loss(y_val, val_probs, labels=labels_all))
        train_acc = float(accuracy_score(y_train, np.argmax(train_probs, axis=1)))
        val_acc = float(accuracy_score(y_val, np.argmax(val_probs, axis=1)))

        logger.log(
            f"Epoch [  1/  1] (L-BFGS iters={actual_iters:4d}/{max_iter}) | "
            f"Train Loss: {train_loss:.5f} | Train Acc: {train_acc:.4f} ({train_acc * 100:6.2f}%) | "
            f"Val Loss: {val_loss:.5f} | Val Acc: {val_acc:.4f} ({val_acc * 100:6.2f}%) *BEST* | "
            f"{fit_sec:.2f}s"
        )

        rows = [
            {
                "epoch": 1,
                "lbfgs_iters": actual_iters,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "best_val_acc": val_acc,
                "is_best": 1,
                "elapsed_sec": fit_sec,
            }
        ]
        save_history_csv(csv_path, rows)

        os.makedirs(model_dir, exist_ok=True)
        atomic_joblib_dump(clf, model_path)

        scaler = clf.pipeline.named_steps["scaler"]
        meta = {
            "run_id": run_id,
            "dataset": dataset_name,
            "classifier": "linear",
            "preprocessor": prep_name,
            "input_dim": int(X_train.shape[1]),
            "n_classes": int(n_classes),
            "epochs_run": 1,
            "lbfgs_iterations": actual_iters,
            "max_iter": max_iter,
            "C": C,
            "seed": seed,
            "scaler_data_min": float(scaler.data_min_),
            "scaler_data_scale": float(scaler.data_scale_),
            "best_epoch": 1,
            "best_val_acc": val_acc,
            "best_val_loss": val_loss,
            "final_train_loss": train_loss,
            "final_train_acc": train_acc,
            "final_val_loss": val_loss,
            "final_val_acc": val_acc,
            "train_time_sec": float(fit_sec),
            "model_path": os.path.relpath(model_path, PROJECT_ROOT),
            "history_txt_path": os.path.relpath(txt_path, PROJECT_ROOT),
            "history_csv_path": os.path.relpath(csv_path, PROJECT_ROOT),
        }
        atomic_json_dump(meta, meta_path)

        logger.log("")
        logger.log("=" * 92)
        logger.log(f"Finished {run_id} in {fit_sec:.2f}s | L-BFGS Iters: {actual_iters}")
        logger.log(f"Final Train Loss: {train_loss:.5f} | Train Acc: {train_acc:.4f}")
        logger.log(f"Final Val Loss:   {val_loss:.5f} | Val Acc:   {val_acc:.4f}")
        logger.log(f"Saved Model:      {model_path}")
        logger.log("=" * 92)
        return meta
    finally:
        logger.close()


# ---------------------------------------------------------------------
# 5B. XGBSpectralClassifier (`XGBClassifier` with matched boosting rounds)
# ---------------------------------------------------------------------
def train_xgb_run(
    dataset_name: str,
    dataset_display: str,
    prep_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test_shape: Tuple[int, ...],
    n_classes: int,
    target_rounds: Optional[int],
    max_rounds: int,
    patience_rounds: int,
    max_depth: int,
    learning_rate: float,
    subsample: float,
    colsample_bytree: float,
    xgb_device: str,
    n_jobs: int,
    seed: int,
    log_dir: str,
    model_dir: str,
) -> Dict[str, Any]:
    if XGBClassifier is None:
        raise ImportError(
            "xgboost is required for XGBSpectralClassifier. Install it with `pip install xgboost`."
        )

    run_id = f"{dataset_name}_xgb_{prep_name}"
    txt_path = os.path.join(log_dir, f"{run_id}_training_history12.txt")
    csv_path = os.path.join(log_dir, f"{run_id}_training_history12.csv")
    model_path = os.path.join(model_dir, f"{run_id}.json")
    meta_path = os.path.join(model_dir, f"{run_id}_meta.json")

    is_discovery_run = target_rounds is None
    n_estimators = int(max_rounds if is_discovery_run else target_rounds)
    early_stopping = int(patience_rounds) if is_discovery_run else None

    logger = DualLogger(txt_path)
    try:
        conv_desc = (
            f"Discovery on 'raw' (max_rounds={max_rounds}, early_stopping_rounds={patience_rounds})"
            if is_discovery_run
            else f"Matched fixed budget from 'raw' (n_estimators={n_estimators} rounds, patience=None)"
        )
        regime_info = {
            "Architecture": "XGBSpectralClassifier (Gradient Boosted Decision Trees, hist)",
            "Hyperparameters": (
                f"max_depth={max_depth}, lr={learning_rate}, subsample={subsample}, "
                f"colsample_bytree={colsample_bytree}, device='{xgb_device}', seed={seed}"
            ),
            "Convergence Policy": conv_desc,
        }
        write_history_header(
            logger,
            dataset_name=dataset_name,
            dataset_display=dataset_display,
            classifier_name="xgb (XGBSpectralClassifier)",
            prep_name=prep_name,
            shapes={"train": X_train.shape, "val": X_val.shape, "test": X_test_shape},
            n_classes=n_classes,
            regime_info=regime_info,
        )

        loss_metric = "mlogloss" if n_classes > 2 else "logloss"
        err_metric = "merror" if n_classes > 2 else "error"
        objective = "multi:softprob" if n_classes > 2 else "binary:logistic"

        # Note: XGBoost early_stopping_rounds monitors the LAST metric in eval_metric,
        # so we place loss_metric (`mlogloss`) last so convergence is determined by val log-loss.
        xgb_kwargs: Dict[str, Any] = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            tree_method="hist",
            device=xgb_device,
            n_jobs=n_jobs,
            random_state=seed,
            objective=objective,
            eval_metric=[err_metric, loss_metric],
        )
        if early_stopping is not None:
            xgb_kwargs["early_stopping_rounds"] = early_stopping

        logger.log(f"Starting XGBoost Training (target up to {n_estimators} rounds)...")
        t0 = time.perf_counter()
        try:
            model = XGBClassifier(**xgb_kwargs)
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_train, y_train), (X_val, y_val)],
                verbose=False,
            )
        except Exception as exc:
            if xgb_device != "cpu":
                logger.log(f"[WARN] XGBoost device='{xgb_device}' failed ({exc}); falling back to 'cpu'.")
                xgb_kwargs["device"] = "cpu"
                model = XGBClassifier(**xgb_kwargs)
                model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_train, y_train), (X_val, y_val)],
                    verbose=False,
                )
            else:
                raise
        fit_sec = time.perf_counter() - t0

        evals_res = model.evals_result()
        train_losses = evals_res["validation_0"][loss_metric]
        train_errs = evals_res["validation_0"][err_metric]
        val_losses = evals_res["validation_1"][loss_metric]
        val_errs = evals_res["validation_1"][err_metric]

        rounds_run = len(val_losses)
        sec_per_round = fit_sec / max(rounds_run, 1)

        best_val_acc = -1.0
        best_acc_round = 1
        best_val_loss = float("inf")
        best_loss_round = 1
        rows: List[Dict[str, Any]] = []

        for r_idx in range(rounds_run):
            round_num = r_idx + 1
            tr_loss = float(train_losses[r_idx])
            tr_acc = float(1.0 - train_errs[r_idx])
            va_loss = float(val_losses[r_idx])
            va_acc = float(1.0 - val_errs[r_idx])

            is_best_acc = va_acc > best_val_acc
            if is_best_acc:
                best_val_acc = va_acc
                best_acc_round = round_num

            is_best_loss = va_loss < best_val_loss
            if is_best_loss:
                best_val_loss = va_loss
                best_loss_round = round_num

            tag = " *BEST_ACC*" if is_best_acc else (" *BEST_LOSS*" if is_best_loss else "")
            logger.log(
                f"Epoch [{round_num:4d}/{rounds_run:4d}] | "
                f"Train Loss: {tr_loss:.5f} | Train Acc: {tr_acc:.4f} ({tr_acc * 100:6.2f}%) | "
                f"Val Loss: {va_loss:.5f} | Val Acc: {va_acc:.4f} ({va_acc * 100:6.2f}%){tag} | "
                f"~{sec_per_round:.2f}s"
            )
            rows.append(
                {
                    "epoch": round_num,
                    "train_loss": tr_loss,
                    "train_acc": tr_acc,
                    "val_loss": va_loss,
                    "val_acc": va_acc,
                    "best_val_acc": best_val_acc,
                    "best_val_loss": best_val_loss,
                    "is_best_acc": int(is_best_acc),
                    "is_best_loss": int(is_best_loss),
                    "elapsed_sec": round(round_num * sec_per_round, 4),
                }
            )

        save_history_csv(csv_path, rows)

        os.makedirs(model_dir, exist_ok=True)
        tmp_model_path = model_path + ".tmp.json"
        model.get_booster().save_model(tmp_model_path)
        os.replace(tmp_model_path, model_path)

        meta = {
            "run_id": run_id,
            "dataset": dataset_name,
            "classifier": "xgb",
            "preprocessor": prep_name,
            "input_dim": int(X_train.shape[1]),
            "n_classes": int(n_classes),
            "is_discovery_run": is_discovery_run,
            "epochs_run": int(rounds_run),
            "target_rounds": int(n_estimators),
            "patience_rounds": early_stopping,
            "best_epoch": int(best_acc_round),
            "best_val_acc": float(best_val_acc),
            "best_loss_epoch": int(best_loss_round),
            "best_val_loss": float(best_val_loss),
            "final_train_loss": float(train_losses[-1]),
            "final_train_acc": float(1.0 - train_errs[-1]),
            "final_val_loss": float(val_losses[-1]),
            "final_val_acc": float(1.0 - val_errs[-1]),
            "train_time_sec": float(fit_sec),
            "hyperparameters": {
                "max_depth": max_depth,
                "learning_rate": learning_rate,
                "subsample": subsample,
                "colsample_bytree": colsample_bytree,
                "seed": seed,
            },
            "model_path": os.path.relpath(model_path, PROJECT_ROOT),
            "history_txt_path": os.path.relpath(txt_path, PROJECT_ROOT),
            "history_csv_path": os.path.relpath(csv_path, PROJECT_ROOT),
        }
        atomic_json_dump(meta, meta_path)

        logger.log("")
        logger.log("=" * 92)
        logger.log(
            f"Finished {run_id} in {fit_sec:.2f}s | Total Boosting Rounds Run: {rounds_run}"
        )
        logger.log(
            f"Best Val Acc:  {best_val_acc:.4f} (Round {best_acc_round}) | "
            f"Best Val Loss: {best_val_loss:.5f} (Round {best_loss_round})"
        )
        logger.log(
            f"Final Round ({rounds_run}) -> Val Loss: {val_losses[-1]:.5f} | "
            f"Val Acc: {1.0 - val_errs[-1]:.4f}"
        )
        logger.log(f"Saved Model:   {model_path}")
        logger.log("=" * 92)
        return meta
    finally:
        logger.close()


# ---------------------------------------------------------------------
# 5C. BacteriaResNetClassifier (`ResNet1D` with matched epochs)
# ---------------------------------------------------------------------
def _eval_resnet_loss_and_acc(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=(device.type == "cuda"))
            yb = yb.to(device, non_blocking=(device.type == "cuda"))
            logits = model(xb)
            loss = criterion(logits, yb)
            batch_n = int(yb.size(0))
            total_loss += float(loss.item()) * batch_n
            preds = logits.argmax(dim=1)
            correct += int((preds == yb).sum().item())
            total += batch_n
    return total_loss / max(total, 1), correct / max(total, 1)


def train_resnet1d_run(
    dataset_name: str,
    dataset_display: str,
    prep_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test_shape: Tuple[int, ...],
    n_classes: int,
    target_epochs: Optional[int],
    max_epochs: int,
    patience: int,
    batch_size: int,
    lr: float,
    betas: Tuple[float, float],
    device: torch.device,
    seed: int,
    log_dir: str,
    model_dir: str,
) -> Dict[str, Any]:
    run_id = f"{dataset_name}_resnet1d_{prep_name}"
    txt_path = os.path.join(log_dir, f"{run_id}_training_history12.txt")
    csv_path = os.path.join(log_dir, f"{run_id}_training_history12.csv")
    best_model_path = os.path.join(model_dir, f"{run_id}_best.pth")
    last_model_path = os.path.join(model_dir, f"{run_id}_last.pth")
    meta_path = os.path.join(model_dir, f"{run_id}_meta.json")

    is_discovery_run = target_epochs is None
    epochs_limit = int(max_epochs if is_discovery_run else target_epochs)
    active_patience = int(patience) if is_discovery_run else None

    logger = DualLogger(txt_path)
    try:
        conv_desc = (
            f"Discovery on 'raw' (max_epochs={max_epochs}, patience={patience} on val_acc)"
            if is_discovery_run
            else f"Matched fixed budget from 'raw' (epochs={epochs_limit}, patience=None)"
        )
        regime_info = {
            "Architecture": "ResNet1D (Ho et al. 2019: 6 stages x 2 blocks, channels=64->100, kernel=5)",
            "Hyperparameters": (
                f"Adam(lr={lr}, betas={betas}), batch_size={batch_size}, "
                f"device='{device}', seed={seed}"
            ),
            "Convergence Policy": conv_desc,
        }
        write_history_header(
            logger,
            dataset_name=dataset_name,
            dataset_display=dataset_display,
            classifier_name="resnet1d (BacteriaResNetClassifier)",
            prep_name=prep_name,
            shapes={"train": X_train.shape, "val": X_val.shape, "test": X_test_shape},
            n_classes=n_classes,
            regime_info=regime_info,
        )

        # Deterministic seeding for identical initial weights & batch shuffling order
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        input_dim = int(X_train.shape[1])
        model = ResNet1D(
            hidden_sizes=(100, 100, 100, 100, 100, 100),
            num_blocks=(2, 2, 2, 2, 2, 2),
            input_dim=input_dim,
            in_channels=64,
            n_classes=n_classes,
        ).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        logger.log(f"ResNet1D Parameters: {n_params:,} total")

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas)
        criterion = nn.CrossEntropyLoss()

        train_ds = TensorDataset(
            torch.from_numpy(np.ascontiguousarray(X_train, dtype=np.float32)).unsqueeze(1),
            torch.from_numpy(np.ascontiguousarray(y_train, dtype=np.int64)),
        )
        val_ds = TensorDataset(
            torch.from_numpy(np.ascontiguousarray(X_val, dtype=np.float32)).unsqueeze(1),
            torch.from_numpy(np.ascontiguousarray(y_val, dtype=np.int64)),
        )

        g_loader = torch.Generator()
        g_loader.manual_seed(seed)
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            generator=g_loader,
            pin_memory=(device.type == "cuda"),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size * 2,
            shuffle=False,
            pin_memory=(device.type == "cuda"),
        )

        best_val_acc = -1.0
        best_val_loss_at_best_acc = float("inf")
        best_epoch = 1
        best_state = copy.deepcopy(model.state_dict())
        no_improve = 0
        rows: List[Dict[str, Any]] = []

        logger.log(f"Starting ResNet1D Training (up to {epochs_limit} epochs)...")
        t_start = time.perf_counter()
        epochs_run = 0

        for epoch in range(1, epochs_limit + 1):
            ep_t0 = time.perf_counter()
            model.train()
            running_loss = 0.0
            running_correct = 0
            running_total = 0

            for xb, yb in train_loader:
                xb = xb.to(device, non_blocking=(device.type == "cuda"))
                yb = yb.to(device, non_blocking=(device.type == "cuda"))
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

                batch_n = int(yb.size(0))
                running_loss += float(loss.item()) * batch_n
                running_correct += int((logits.argmax(dim=1) == yb).sum().item())
                running_total += batch_n

            train_loss = running_loss / max(running_total, 1)
            train_acc = running_correct / max(running_total, 1)
            val_loss, val_acc = _eval_resnet_loss_and_acc(model, val_loader, criterion, device)
            ep_sec = time.perf_counter() - ep_t0
            spec_per_s = running_total / max(ep_sec, 1e-9)
            epochs_run = epoch

            is_best = val_acc > best_val_acc
            if is_best:
                best_val_acc = val_acc
                best_val_loss_at_best_acc = val_loss
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                no_improve = 0
                best_tag = " *BEST*"
            else:
                no_improve += 1
                best_tag = f" (patience {no_improve}/{active_patience})" if active_patience else ""

            logger.log(
                f"Epoch [{epoch:3d}/{epochs_limit:3d}] | "
                f"Train Loss: {train_loss:.5f} | Train Acc: {train_acc:.4f} ({train_acc * 100:6.2f}%) | "
                f"Val Loss: {val_loss:.5f} | Val Acc: {val_acc:.4f} ({val_acc * 100:6.2f}%){best_tag} | "
                f"{ep_sec:.2f}s ({spec_per_s:,.0f} spec/s)"
            )

            rows.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "best_val_acc": best_val_acc,
                    "best_epoch": best_epoch,
                    "is_best": int(is_best),
                    "lr": lr,
                    "epoch_sec": round(ep_sec, 4),
                    "spec_per_s": round(spec_per_s, 1),
                }
            )

            if active_patience is not None and no_improve >= active_patience:
                logger.log(
                    f"Early stopping triggered at epoch {epoch} "
                    f"(best val_acc={best_val_acc:.4f} at epoch {best_epoch})."
                )
                break

        total_train_sec = time.perf_counter() - t_start
        last_state = copy.deepcopy(model.state_dict())
        save_history_csv(csv_path, rows)

        os.makedirs(model_dir, exist_ok=True)
        common_ckpt_meta = {
            "run_id": run_id,
            "dataset": dataset_name,
            "classifier": "resnet1d",
            "preprocessor": prep_name,
            "input_dim": input_dim,
            "n_classes": int(n_classes),
            "hidden_sizes": [100, 100, 100, 100, 100, 100],
            "num_blocks": [2, 2, 2, 2, 2, 2],
            "in_channels": 64,
            "epochs_run": int(epochs_run),
            "best_epoch": int(best_epoch),
            "best_val_acc": float(best_val_acc),
            "best_val_loss": float(best_val_loss_at_best_acc),
            "final_val_acc": float(rows[-1]["val_acc"]),
            "final_val_loss": float(rows[-1]["val_loss"]),
        }

        atomic_torch_save(
            {
                **common_ckpt_meta,
                "checkpoint_type": "best_val_acc",
                "epoch": int(best_epoch),
                "model_state_dict": best_state,
            },
            best_model_path,
        )
        atomic_torch_save(
            {
                **common_ckpt_meta,
                "checkpoint_type": "last_epoch",
                "epoch": int(epochs_run),
                "model_state_dict": last_state,
            },
            last_model_path,
        )

        meta = {
            **common_ckpt_meta,
            "is_discovery_run": is_discovery_run,
            "target_epochs": int(epochs_limit),
            "patience": active_patience,
            "parameters": int(n_params),
            "final_train_loss": float(rows[-1]["train_loss"]),
            "final_train_acc": float(rows[-1]["train_acc"]),
            "train_time_sec": float(total_train_sec),
            "hyperparameters": {
                "lr": lr,
                "betas": list(betas),
                "batch_size": batch_size,
                "seed": seed,
            },
            "best_model_path": os.path.relpath(best_model_path, PROJECT_ROOT),
            "last_model_path": os.path.relpath(last_model_path, PROJECT_ROOT),
            "history_txt_path": os.path.relpath(txt_path, PROJECT_ROOT),
            "history_csv_path": os.path.relpath(csv_path, PROJECT_ROOT),
        }
        atomic_json_dump(meta, meta_path)

        logger.log("")
        logger.log("=" * 92)
        logger.log(f"Finished {run_id} in {total_train_sec:.2f}s | Epochs Run: {epochs_run}")
        logger.log(
            f"Best Val Acc:  {best_val_acc:.4f} (Epoch {best_epoch}, Val Loss: {best_val_loss_at_best_acc:.5f})"
        )
        logger.log(
            f"Final Epoch ({epochs_run}) -> Val Loss: {rows[-1]['val_loss']:.5f} | "
            f"Val Acc: {rows[-1]['val_acc']:.4f}"
        )
        logger.log(f"Saved Best Weights: {best_model_path}")
        logger.log(f"Saved Last Weights: {last_model_path}")
        logger.log("=" * 92)
        return meta
    finally:
        logger.close()


# =====================================================================
# 6. MAIN TRAINING DRIVER
# =====================================================================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Round 12 Downstream Classification Training Suite (18 Runs: 2 Datasets x 3 Preps x 3 Classifiers)"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DATASETS_ORDER),
        default=list(DATASETS_ORDER),
        help="Datasets to train on (default: bacteria_id mlrod).",
    )
    parser.add_argument(
        "--classifiers",
        nargs="+",
        choices=list(CLASSIFIERS_ORDER),
        default=list(CLASSIFIERS_ORDER),
        help="Classifiers to train (default: linear xgb resnet1d).",
    )
    parser.add_argument(
        "--bacteria-dir",
        type=str,
        default=os.path.join("datasets", "bacteria-ID"),
        help="Path to datasets/bacteria-ID directory.",
    )
    parser.add_argument(
        "--mlrod-dir",
        type=str,
        default=os.path.join("datasets", "MLROD"),
        help="Path to datasets/MLROD directory.",
    )
    parser.add_argument(
        "--mlrod-subset",
        type=str,
        choices=["pure12", "nonslab15"],
        default="pure12",
        help="MLROD class filter: 'pure12' (12 pure single minerals) or 'nonslab15'.",
    )
    parser.add_argument(
        "--pgnet-model",
        type=str,
        default=os.path.join("training_round12", "polygaussnet12.pth"),
        help="Path to PolyGaussNet V3.4-8sec model weights.",
    )
    parser.add_argument(
        "--pgnet-ckpt",
        type=str,
        default=os.path.join("training_round12", "training_checkpoint12.pth"),
        help="Fallback checkpoint path for PolyGaussNet V3.4-8sec.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="training_round12",
        help="Root directory under which downstream_logs/, downstream_models/, downstream_cache/ are created.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="PyTorch device ('auto', 'cuda', 'mps', 'cpu').",
    )
    parser.add_argument(
        "--xgb-device",
        type=str,
        default="auto",
        help="XGBoost device ('auto', 'cuda', 'cpu').",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
        help="Number of CPU workers for conventional preprocessing, Linear, and XGBoost.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Master random seed (default: 42).")

    # ResNet1D hyperparameters
    parser.add_argument(
        "--resnet-max-epochs",
        type=int,
        default=50,
        help="Max epochs for 'raw' ResNet1D convergence discovery (default: 50).",
    )
    parser.add_argument(
        "--resnet-patience",
        type=int,
        default=7,
        help="Validation patience epochs for 'raw' ResNet1D convergence discovery (default: 7).",
    )
    parser.add_argument(
        "--resnet-batch-size",
        type=int,
        default=64,
        help="Batch size for ResNet1D training (default: 64).",
    )
    parser.add_argument(
        "--resnet-lr",
        type=float,
        default=1e-3,
        help="Adam learning rate for ResNet1D (default: 1e-3).",
    )

    # XGBoost hyperparameters
    parser.add_argument(
        "--xgb-max-rounds",
        type=int,
        default=500,
        help="Max boosting rounds for 'raw' XGBoost convergence discovery (default: 500).",
    )
    parser.add_argument(
        "--xgb-patience",
        type=int,
        default=30,
        help="Early stopping rounds for 'raw' XGBoost convergence discovery (default: 30).",
    )
    parser.add_argument("--xgb-max-depth", type=int, default=6, help="XGBoost max_depth (default: 6).")
    parser.add_argument("--xgb-lr", type=float, default=0.05, help="XGBoost learning_rate (default: 0.05).")

    # Linear hyperparameters
    parser.add_argument("--linear-C", type=float, default=1.0, help="LogisticRegression C (default: 1.0).")
    parser.add_argument(
        "--linear-max-iter",
        type=int,
        default=2000,
        help="LogisticRegression max_iter (default: 2000).",
    )

    # Caching & smoke testing
    parser.add_argument(
        "--pgnet-batch-size",
        type=int,
        default=512,
        help="Batch size for PolyGaussNet V3.4 inference (default: 512).",
    )
    parser.add_argument(
        "--cache-train",
        action="store_true",
        help="Also save preprocessed training splits in downstream_cache/ (default caches val & test only).",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Reuse existing preprocessed .npy splits in downstream_cache/ if present.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Re-train all runs from scratch even if completed model weights and _meta.json already exist.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a fast stratified mini-subset (3 max epochs / 10 XGB rounds) to verify pipeline end-to-end.",
    )
    return parser.parse_args()


def _try_load_completed_run(
    ds_key: str, clf_key: str, prep_key: str, model_dir: str, log_dir: str
) -> Optional[Dict[str, Any]]:
    """Return saved run metadata if this run already finished and all artifacts exist on disk."""
    run_id = f"{ds_key}_{clf_key}_{prep_key}"
    meta_path = os.path.join(model_dir, f"{run_id}_meta.json")
    txt_path = os.path.join(log_dir, f"{run_id}_training_history12.txt")
    if not (os.path.exists(meta_path) and os.path.exists(txt_path)):
        return None
    if clf_key == "linear":
        req_models = [os.path.join(model_dir, f"{run_id}.joblib")]
    elif clf_key == "xgb":
        req_models = [os.path.join(model_dir, f"{run_id}.json")]
    elif clf_key == "resnet1d":
        req_models = [
            os.path.join(model_dir, f"{run_id}_best.pth"),
            os.path.join(model_dir, f"{run_id}_last.pth"),
        ]
    else:
        return None
    if not all(os.path.exists(p) for p in req_models):
        return None
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def select_devices(device_arg: str, xgb_device_arg: str) -> Tuple[torch.device, str]:
    if device_arg == "auto":
        if torch.cuda.is_available():
            pt_dev = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            pt_dev = torch.device("mps")
        else:
            pt_dev = torch.device("cpu")
    else:
        pt_dev = torch.device(device_arg)

    if xgb_device_arg == "auto":
        xgb_dev = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        xgb_dev = xgb_device_arg

    return pt_dev, xgb_dev


def main() -> int:
    args = parse_args()
    if args.smoke_test:
        args.resnet_max_epochs = min(args.resnet_max_epochs, 3)
        args.resnet_patience = min(args.resnet_patience, 2)
        args.xgb_max_rounds = min(args.xgb_max_rounds, 8)
        args.xgb_patience = min(args.xgb_patience, 3)
        args.linear_max_iter = min(args.linear_max_iter, 100)

    out_root = resolve_path(args.output_root)
    log_dir = os.path.join(out_root, "downstream_logs")
    model_dir = os.path.join(out_root, "downstream_models")
    cache_dir = os.path.join(out_root, "downstream_cache")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    summary_txt_path = os.path.join(log_dir, "training_summary12.txt")
    manifest_json_path = os.path.join(log_dir, "training_manifest12.json")
    master_log = DualLogger(summary_txt_path, append=not args.no_resume)

    stop_requested = {"flag": False, "sig": None}

    def _handle_sigterm(signum: int, _frame: Any) -> None:
        stop_requested["flag"] = True
        stop_requested["sig"] = signum
        print(
            f"\n[SIGNAL {signum}] Slurm SIGTERM received; finishing active run cleanly before exiting...",
            flush=True,
        )

    prev_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, _handle_sigterm)

    t_global = time.perf_counter()
    try:
        pt_device, xgb_device = select_devices(args.device, args.xgb_device)
        if pt_device.type == "cuda":
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        gpu_name = torch.cuda.get_device_name(0) if pt_device.type == "cuda" else pt_device.type.upper()

        master_log.log("=" * 96)
        master_log.log(" ROUND 12 DOWNSTREAM CLASSIFICATION TRAINING SUITE (18 RUNS)")
        master_log.log("=" * 96)
        master_log.log(f"  - Slurm Job ID:       {os.environ.get('SLURM_JOB_ID', 'local')}")
        master_log.log(f"  - PyTorch Device:     {pt_device} ({gpu_name})")
        master_log.log(f"  - XGBoost Device:     {xgb_device}")
        master_log.log(f"  - CPU Workers:        {args.n_jobs}")
        master_log.log(f"  - Datasets:           {args.datasets} (MLROD subset={args.mlrod_subset})")
        master_log.log(f"  - Preprocessors:      {list(PREPROCESSORS_ORDER)}")
        master_log.log(f"  - Classifiers:        {args.classifiers}")
        master_log.log(f"  - Upfront Norm:       Per-spectrum [0, 1] min-max scaling on ALL spectra")
        master_log.log(f"  - Resume Completed:   {not args.no_resume}")
        master_log.log(f"  - Smoke Test Mode:    {args.smoke_test}")
        master_log.log(f"  - Logs Dir:           {log_dir}")
        master_log.log(f"  - Models Dir:         {model_dir}")
        master_log.log(f"  - Cache Dir:          {cache_dir}")
        master_log.log("=" * 96)

        # Load PolyGaussNet V3.4-8sec
        pgnet_model, pgnet_weight_path, pgnet_epoch, pgnet_loss = load_pgnet_v34_model(
            args.pgnet_model, args.pgnet_ckpt, device=pt_device
        )
        pgnet_params = sum(p.numel() for p in pgnet_model.parameters())
        master_log.log(
            f"\nLoaded PolyGaussNet V3.4-8sec: {pgnet_weight_path} "
            f"({pgnet_params:,} params, epoch={pgnet_epoch}, train_loss={pgnet_loss})"
        )

        manifest: Dict[str, Any] = {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID", "local"),
            "pytorch_device": str(pt_device),
            "gpu_name": gpu_name,
            "xgb_device": xgb_device,
            "seed": args.seed,
            "smoke_test": args.smoke_test,
            "upfront_normalization": "per_spectrum_min_max_0_1",
            "pgnet_v34": {
                "weight_path": os.path.relpath(pgnet_weight_path, PROJECT_ROOT),
                "parameters": pgnet_params,
                "epoch": pgnet_epoch,
                "train_loss": pgnet_loss,
            },
            "datasets": {},
            "matched_budgets": {},
            "runs": {},
        }

        all_run_metas: List[Dict[str, Any]] = []

        for ds_key in args.datasets:
            master_log.log("\n" + "-" * 96)
            master_log.log(f"STEP 1: Loading & Normalizing Dataset '{ds_key}'")
            master_log.log("-" * 96)

            if ds_key == "bacteria_id":
                dataset = load_bacteria_id_dataset(
                    args.bacteria_dir, seed=args.seed, smoke_test=args.smoke_test
                )
            elif ds_key == "mlrod":
                dataset = load_mlrod_dataset(
                    args.mlrod_dir,
                    subset_mode=args.mlrod_subset,
                    split_seed=args.seed,
                    smoke_test=args.smoke_test,
                )
            else:
                raise ValueError(f"Unsupported dataset: {ds_key}")

            X_tr_raw = dataset["X_train"]
            X_va_raw = dataset["X_val"]
            X_te_raw = dataset["X_test"]
            y_train = dataset["y_train"]
            y_val = dataset["y_val"]
            y_test = dataset["y_test"]
            n_classes = len(dataset["class_names"])

            master_log.log(f"  Display Name: {dataset['display_name']}")
            master_log.log(
                f"  Shapes:       train={X_tr_raw.shape}, val={X_va_raw.shape}, test={X_te_raw.shape} "
                f"| classes={n_classes}"
            )
            master_log.log(
                f"  Range Check:  train=[{X_tr_raw.min():.4f}, {X_tr_raw.max():.4f}], "
                f"val=[{X_va_raw.min():.4f}, {X_va_raw.max():.4f}], "
                f"test=[{X_te_raw.min():.4f}, {X_te_raw.max():.4f}]"
            )

            master_log.log(f"\nSTEP 2: Running Preprocessors for '{ds_key}' (raw, conventional, pgnet_v34)")
            splits_by_prep, timings_by_prep = prepare_and_cache_preprocessed_splits(
                dataset=dataset,
                cache_dir=cache_dir,
                pgnet_model=pgnet_model,
                device=pt_device,
                pgnet_batch_size=args.pgnet_batch_size,
                n_jobs=args.n_jobs,
                cache_train=args.cache_train,
                use_existing_cache=args.use_cache,
                master_log=master_log,
            )

            manifest["datasets"][ds_key] = {
                "display_name": dataset["display_name"],
                "n_classes": n_classes,
                "class_names": dataset["class_names"],
                "wavenumber_bins": int(X_tr_raw.shape[1]),
                "wavenumber_min": float(dataset["wavenumbers"].min()),
                "wavenumber_max": float(dataset["wavenumbers"].max()),
                "train_samples": int(len(y_train)),
                "val_samples": int(len(y_val)),
                "test_samples": int(len(y_test)),
                "preprocessing_timings": timings_by_prep,
            }
            manifest["matched_budgets"][ds_key] = {}

            master_log.log(
                f"\nSTEP 3: Training Classifiers on '{ds_key}' "
                f"(Raw sets convergence budget -> Conventional & PGNet V3.4 match exact epochs/rounds)"
            )

            for clf_key in args.classifiers:
                discovered_budget: Optional[int] = None

                for prep_key in PREPROCESSORS_ORDER:
                    if stop_requested["flag"]:
                        master_log.log(
                            f"\n[PREEMPTED] Exiting cleanly before starting {ds_key}_{clf_key}_{prep_key} "
                            f"(signal {stop_requested['sig']}). Resubmit job to resume remaining runs."
                        )
                        return 0

                    existing_meta = (
                        None
                        if args.no_resume
                        else _try_load_completed_run(ds_key, clf_key, prep_key, model_dir, log_dir)
                    )
                    if existing_meta is not None:
                        run_meta = existing_meta
                        master_log.log(
                            f"\n>>> [Resume Hit] Skipping completed run: {run_meta['run_id']} "
                            f"(epochs_run={run_meta['epochs_run']}, best_val_acc={run_meta['best_val_acc'] * 100:.2f}%)"
                        )
                    else:
                        X_tr = splits_by_prep[prep_key]["train"]
                        X_va = splits_by_prep[prep_key]["val"]
                        X_te_shape = splits_by_prep[prep_key]["test"].shape

                        master_log.log(
                            f"\n>>> Training Run: dataset={ds_key} | classifier={clf_key} | "
                            f"preprocessor={prep_key} | target_budget={discovered_budget or 'DISCOVER_FROM_RAW'}"
                        )

                        if clf_key == "linear":
                            run_meta = train_linear_run(
                                dataset_name=ds_key,
                                dataset_display=dataset["display_name"],
                                prep_name=prep_key,
                                X_train=X_tr,
                                y_train=y_train,
                                X_val=X_va,
                                y_val=y_val,
                                X_test_shape=X_te_shape,
                                n_classes=n_classes,
                                C=args.linear_C,
                                max_iter=args.linear_max_iter,
                                seed=args.seed,
                                n_jobs=args.n_jobs,
                                log_dir=log_dir,
                                model_dir=model_dir,
                            )
                        elif clf_key == "xgb":
                            run_meta = train_xgb_run(
                                dataset_name=ds_key,
                                dataset_display=dataset["display_name"],
                                prep_name=prep_key,
                                X_train=X_tr,
                                y_train=y_train,
                                X_val=X_va,
                                y_val=y_val,
                                X_test_shape=X_te_shape,
                                n_classes=n_classes,
                                target_rounds=discovered_budget,
                                max_rounds=args.xgb_max_rounds,
                                patience_rounds=args.xgb_patience,
                                max_depth=args.xgb_max_depth,
                                learning_rate=args.xgb_lr,
                                subsample=0.8,
                                colsample_bytree=0.8,
                                xgb_device=xgb_device,
                                n_jobs=args.n_jobs,
                                seed=args.seed,
                                log_dir=log_dir,
                                model_dir=model_dir,
                            )
                        elif clf_key == "resnet1d":
                            run_meta = train_resnet1d_run(
                                dataset_name=ds_key,
                                dataset_display=dataset["display_name"],
                                prep_name=prep_key,
                                X_train=X_tr,
                                y_train=y_train,
                                X_val=X_va,
                                y_val=y_val,
                                X_test_shape=X_te_shape,
                                n_classes=n_classes,
                                target_epochs=discovered_budget,
                                max_epochs=args.resnet_max_epochs,
                                patience=args.resnet_patience,
                                batch_size=args.resnet_batch_size,
                                lr=args.resnet_lr,
                                betas=(0.5, 0.999),
                                device=pt_device,
                                seed=args.seed,
                                log_dir=log_dir,
                                model_dir=model_dir,
                            )
                        else:
                            raise ValueError(f"Unsupported classifier: {clf_key}")

                    if prep_key == "raw":
                        if clf_key == "linear":
                            discovered_budget = args.linear_max_iter
                            manifest["matched_budgets"][ds_key]["linear"] = {
                                "mode": "fixed_lbfgs_max_iter",
                                "max_iter": args.linear_max_iter,
                                "raw_lbfgs_iterations": run_meta.get("lbfgs_iterations", 0),
                            }
                        elif clf_key == "xgb":
                            discovered_budget = int(run_meta["epochs_run"])
                            manifest["matched_budgets"][ds_key]["xgb"] = {
                                "mode": "matched_boosting_rounds_from_raw",
                                "epochs_run": discovered_budget,
                                "raw_best_acc_round": run_meta["best_epoch"],
                                "raw_best_loss_round": run_meta.get("best_loss_epoch", run_meta["best_epoch"]),
                                "patience_rounds": args.xgb_patience,
                            }
                        elif clf_key == "resnet1d":
                            discovered_budget = int(run_meta["epochs_run"])
                            manifest["matched_budgets"][ds_key]["resnet1d"] = {
                                "mode": "matched_epochs_from_raw",
                                "epochs_run": discovered_budget,
                                "raw_best_epoch": run_meta["best_epoch"],
                                "patience": args.resnet_patience,
                            }

                    manifest["runs"][run_meta["run_id"]] = run_meta
                    all_run_metas.append(run_meta)

                    # Save manifest atomically after every run so progress is preserved
                    atomic_json_dump(manifest, manifest_json_path)

        # Print final summary table of all 18 training runs
        total_elapsed = time.perf_counter() - t_global
        manifest["total_elapsed_sec"] = float(total_elapsed)
        manifest["all_18_completed"] = len(all_run_metas) == (
            len(args.datasets) * len(args.classifiers) * len(PREPROCESSORS_ORDER)
        )
        atomic_json_dump(manifest, manifest_json_path)

        master_log.log("\n" + "=" * 96)
        master_log.log(" TRAINING SUMMARY ACROSS ALL COMPLETED RUNS")
        master_log.log("=" * 96)
        header = (
            f"{'Run ID':36s} | {'Epochs/Rnds':>11s} | {'Best Ep':>7s} | "
            f"{'Best Val Acc':>12s} | {'Final Val Acc':>13s} | {'Final Val Loss':>14s} | {'Time (s)':>8s}"
        )
        master_log.log(header)
        master_log.log("-" * len(header))
        for m in all_run_metas:
            master_log.log(
                f"{m['run_id']:36s} | {m['epochs_run']:11d} | {m['best_epoch']:7d} | "
                f"{m['best_val_acc'] * 100:11.2f}% | {m['final_val_acc'] * 100:12.2f}% | "
                f"{m['final_val_loss']:14.5f} | {m['train_time_sec']:8.2f}"
            )
        master_log.log("-" * len(header))
        master_log.log(f"Total Suite Wall-Clock Time: {total_elapsed:.2f}s ({total_elapsed / 60.0:.2f} min)")
        master_log.log(f"Saved Training Manifest:     {manifest_json_path}")
        master_log.log(f"Saved Training Summary:      {summary_txt_path}")
        master_log.log("=" * 96)
        return 0
    finally:
        signal.signal(signal.SIGTERM, prev_sigterm)
        master_log.close()


if __name__ == "__main__":
    raise SystemExit(main())
