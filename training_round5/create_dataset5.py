"""
DATASET GENERATION PIPELINE - ROUND 5
--------------------------------------------------------------------------------
Generates the scaled multi-resolution Raman dataset (1,966,080 spectra total):
- 5 Base Bin Sizes: 800, 1016, 1200, 1400, 1600
- 3 Stretching Modes:
    * 1.0x (Original resolution)
    * 1.5x (Stretched resolution)
    * 2.0x (Double resolution)
- 8 files per bin size, 16,384 spectra per file:
    * 131,072 spectra per bin folder
    * 655,360 spectra per stretching mode
    * 1,966,080 spectra total

Directory Structure:
training_data5/
├── 1x/
│   ├── 800/
│   ├── 1016/
│   ├── 1200/
│   ├── 1400/
│   └── 1600/
├── 1.5x/
│   ├── 800/
│   ├── 1016/
│   ├── 1200/
│   ├── 1400/
│   └── 1600/
└── 2x/
    ├── 800/
    ├── 1016/
    ├── 1200/
    ├── 1400/
    └── 1600/
--------------------------------------------------------------------------------
"""

import os
import sys
import gc
import time
import shutil
import argparse
from multiprocessing import Pool, cpu_count
import numpy as np
from scipy.ndimage import zoom

# Configure paths so imports and outputs resolve whether running from project root or inside training_round5
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
for path in [PROJECT_ROOT, SCRIPT_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

from spectra_generator import generate_spectra

# =====================================================================
# 1. CONFIGURATION & NEW DATASET PARAMETERS (ROUND 5)
# =====================================================================
DEFAULT_BASE_DIR = os.path.join(SCRIPT_DIR, "training_data5")
DEFAULT_BIN_SIZES = [800, 1016, 1200, 1400, 1600]
DEFAULT_MODES = ["1x", "1.5x", "2x"]
DEFAULT_FILES_PER_BIN = 8
DEFAULT_SAMPLES_PER_FILE = 16384
DEFAULT_WORKERS = 4
DEFAULT_DTYPE = "float32"  # float32 saves 50% disk space and matches PyTorch defaults

# Base parameters from Round 5 Specification
GENERATION_PARAMS_TEMPLATE = {
    "amplitude_range": [0.001, 0.2],
    "width_range": [1, 100],
    "degree_range": [1, 16],
    "offset_range": [0.0, 1.0],
    "max_coeff": 1.0,
    "min_peak_ratio": 2.0,
    "std_range": [1, 10],
    "probability_cosmic": 1 / 24000,
    "intensity_range_cosmic": [5.0, 20.0],
    "domain_mapping": [-1.0, 1.0],
    "min_value": 0,
    "normalize": True
}


def stretch_matrix(arr: np.ndarray, factor: float) -> np.ndarray:
    """
    Linearly interpolates (stretches) a 2D matrix (batch_size, bins) along the spectral axis.
    Uses scipy.ndimage.zoom with order=1 for fast, zero-dependency, C-accelerated interpolation.
    """
    if factor == 1.0:
        return arr
    return zoom(arr, (1.0, factor), order=1)


def generate_and_save_chunk(args):
    """
    Multiprocessing worker task:
    1. Generates 1 base batch of 16,384 spectra at `bin_size`.
    2. Stretches and saves into 1x, 1.5x, and 2x folders.
    3. Handles resume logic by skipping if files already exist.
    """
    (
        file_idx,
        bin_size,
        samples_count,
        out_paths_by_mode,  # dict: {"1x": path, "1.5x": path, "2x": path}
        dtype_str
    ) = args

    pid = os.getpid()
    np_dtype = np.float32 if dtype_str == "float32" else np.float64

    # Check if all destination files for this chunk already exist (resume capability)
    all_exist = all(os.path.exists(p) and os.path.getsize(p) > 1024 for p in out_paths_by_mode.values())
    if all_exist:
        print(f"[Worker {pid:5d}] Chunk {file_idx:02d} for bin {bin_size} already exists across all modes. Skipping.")
        return bin_size, file_idx, 0.0, True

    # Build exact generation parameters for this bin size
    # wavenum_range = [0, bin_size - 1] -> exactly bin_size bins
    # num_peaks_range = [0, bin_size // 5]
    gen_params = dict(GENERATION_PARAMS_TEMPLATE)
    gen_params["wavenum_range"] = [0, bin_size - 1]
    gen_params["num_peaks_range"] = [0, int(bin_size // 5)]

    # Independent RNG per worker process
    seed = int.from_bytes(os.urandom(4), byteorder="little") ^ pid
    worker_rng = np.random.default_rng(seed)

    start_time = time.time()
    print(f"[Worker {pid:5d}] Starting chunk {file_idx:02d} | Bins: {bin_size} ({samples_count:,} spectra)...")

    # 1. Generate base 1x spectra
    pure, noise_cosmic, full = generate_spectra(
        batch_size=samples_count,
        rng=worker_rng,
        **gen_params
    )

    if np_dtype != np.float64:
        pure = pure.astype(np_dtype)
        noise_cosmic = noise_cosmic.astype(np_dtype)
        full = full.astype(np_dtype)

    # 2. Process and save for each stretching mode
    # 1x: Base resolution
    if "1x" in out_paths_by_mode:
        np.savez_compressed(
            out_paths_by_mode["1x"],
            pure_matrix=pure,
            pure_noise_cosmic_matrix=noise_cosmic,
            full_matrix=full
        )

    # 1.5x: Stretched resolution (factor = 1.5)
    if "1.5x" in out_paths_by_mode:
        pure_15 = stretch_matrix(pure, 1.5)
        noise_15 = stretch_matrix(noise_cosmic, 1.5)
        full_15 = stretch_matrix(full, 1.5)
        np.savez_compressed(
            out_paths_by_mode["1.5x"],
            pure_matrix=pure_15,
            pure_noise_cosmic_matrix=noise_15,
            full_matrix=full_15
        )
        del pure_15, noise_15, full_15

    # 2.0x: Double resolution (factor = 2.0)
    if "2x" in out_paths_by_mode:
        pure_20 = stretch_matrix(pure, 2.0)
        noise_20 = stretch_matrix(noise_cosmic, 2.0)
        full_20 = stretch_matrix(full, 2.0)
        np.savez_compressed(
            out_paths_by_mode["2x"],
            pure_matrix=pure_20,
            pure_noise_cosmic_matrix=noise_20,
            full_matrix=full_20
        )
        del pure_20, noise_20, full_20

    elapsed = time.time() - start_time
    total_mb = sum(os.path.getsize(p) for p in out_paths_by_mode.values()) / (1024 * 1024)
    print(f"[Worker {pid:5d}] Finished chunk {file_idx:02d} | Bins: {bin_size} in {elapsed:.1f}s ({total_mb:.1f} MB written across modes)")

    del pure, noise_cosmic, full
    gc.collect()
    return bin_size, file_idx, elapsed, False


# =====================================================================
# 2. MAIN CLI DISPATCHER
# =====================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Multi-Resolution Stretched Raman Spectra Dataset (Round 5).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--output-dir", type=str, default=DEFAULT_BASE_DIR,
                        help="Target output directory for training_data5")
    parser.add_argument("--bin-sizes", type=int, nargs="+", default=DEFAULT_BIN_SIZES,
                        help="List of base bin numbers to generate")
    parser.add_argument("--modes", type=str, nargs="+", default=DEFAULT_MODES,
                        help="Stretching modes to generate")
    parser.add_argument("--files-per-bin", type=int, default=DEFAULT_FILES_PER_BIN,
                        help="Number of files/chunks per bin folder")
    parser.add_argument("--samples-per-file", type=int, default=DEFAULT_SAMPLES_PER_FILE,
                        help="Number of spectra per file chunk")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help="Number of parallel multiprocessing worker processes")
    parser.add_argument("--dtype", type=str, choices=["float32", "float64"], default=DEFAULT_DTYPE,
                        help="Data type for saved matrices (float32 recommended for disk space and PyTorch)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print directory plan, sample counts, and disk estimation without generating")
    return parser.parse_args()


def main():
    args = parse_args()

    # Calculate total dataset metrics
    num_bins = len(args.bin_sizes)
    num_modes = len(args.modes)
    spectra_per_bin_mode = args.files_per_bin * args.samples_per_file
    spectra_per_mode = num_bins * spectra_per_bin_mode
    total_spectra = num_modes * spectra_per_mode
    total_files = num_modes * num_bins * args.files_per_bin

    # Estimated sizes (float32: ~180 MB per chunk across 3 modes; float64: ~360 MB)
    est_gb = (total_spectra * 1805 * 3 * (4 if args.dtype == "float32" else 8) * 0.45) / (1024 ** 3)

    print("=" * 80)
    print("RAMAN SYNTHETIC DATASET GENERATION PIPELINE - ROUND 5")
    print("=" * 80)
    print(f"Output Directory:     {args.output_dir}")
    print(f"Stretching Modes:     {args.modes}")
    print(f"Base Bin Sizes:       {args.bin_sizes}")
    print(f"Files per Bin:        {args.files_per_bin}")
    print(f"Spectra per File:     {args.samples_per_file:,}")
    print(f"Spectra per Mode:     {spectra_per_mode:,}")
    print(f"Total Spectra:        {total_spectra:,}")
    print(f"Total Files:          {total_files} files ({num_bins * args.files_per_bin} base chunks)")
    print(f"Data Type:            {args.dtype}")
    print(f"Estimated Disk Space: ~{est_gb:.1f} GB")
    print(f"Parallel Workers:     {args.workers}")
    print("=" * 80)

    # 1. Create directory hierarchy
    for mode in args.modes:
        for b in args.bin_sizes:
            dir_path = os.path.join(args.output_dir, mode, str(b))
            os.makedirs(dir_path, exist_ok=True)

    if args.dry_run:
        print("\n[DRY RUN] Directories successfully initialized. Exiting without generating spectra.")
        return

    # 2. Build task list
    # Each base chunk (bin_size, chunk_idx) generates 1 base batch and writes to all 3 modes
    tasks = []
    for b in args.bin_sizes:
        for file_idx in range(1, args.files_per_bin + 1):
            out_paths_by_mode = {}
            for mode in args.modes:
                fname = f"spectra_chunk_{file_idx:02d}.npz"
                out_paths_by_mode[mode] = os.path.join(args.output_dir, mode, str(b), fname)

            tasks.append((
                file_idx,
                b,
                args.samples_per_file,
                out_paths_by_mode,
                args.dtype
            ))

    print(f"\nLaunching {len(tasks)} generation tasks across {args.workers} workers...\n")
    start_all = time.time()

    with Pool(processes=args.workers) as pool:
        results = pool.map(generate_and_save_chunk, tasks)

    total_time = time.time() - start_all
    skipped_count = sum(1 for r in results if r[3])
    completed_count = len(results) - skipped_count

    print("\n" + "=" * 80)
    print("DATASET GENERATION COMPLETE")
    print("=" * 80)
    print(f"Completed Tasks:  {completed_count}")
    print(f"Skipped Tasks:    {skipped_count} (already existed)")
    print(f"Total Time:       {total_time / 60:.2f} minutes ({total_time:.1f}s)")
    print(f"Output Location:  {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
