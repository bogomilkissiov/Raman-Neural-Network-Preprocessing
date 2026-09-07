"""
DATASET GENERATION PIPELINE - ROUND 5 (OPTION 1: NATIVE DIRECT GENERATION)
--------------------------------------------------------------------------------
Generates the scaled multi-resolution Raman dataset (1,966,080 spectra total):
- 5 Base Bin Sizes: 800, 1016, 1200, 1400, 1600
- 3 Optical Resolution Modes:
    * 1.0x (Original resolution):  800,  1016, 1200, 1400, 1600 bins
    * 1.5x (1.5x optical sampling): 1200, 1524, 1800, 2100, 2400 bins
    * 2.0x (2.0x optical sampling): 1600, 2032, 2400, 2800, 3200 bins
- 8 files per bin size, 16,384 spectra per file:
    * 131,072 spectra per bin folder
    * 655,360 spectra per stretching mode
    * 1,966,080 spectra total

PHYSICAL SCALING PRINCIPLES (OPTION 1):
1. Peak widths scale with optical sampling factor: [1.0 * factor, 100.0 * factor].
2. Chemical peak count is invariant to spectrometer grating: num_peaks_range = [0, base_bins // 5].
3. Detector noise remains authentic, pixel-independent white noise across all resolutions.
4. Cosmic ray events per spectrum remain invariant by scaling per-bin probability:
   probability_cosmic = base_probability / factor.
5. Single-pixel cosmic ray hits remain authentic single-pixel delta spikes (never blurred).

DIRECTORY STRUCTURE:
training_data5/
├── 1x/
│   ├── 800/
│   ├── 1016/
│   ├── 1200/
│   ├── 1400/
│   └── 1600/
├── 1.5x/
│   ├── 1200/
│   ├── 1524/
│   ├── 1800/
│   ├── 2100/
│   └── 2400/
└── 2x/
    ├── 1600/
    ├── 2032/
    ├── 2400/
    ├── 2800/
    └── 3200/
--------------------------------------------------------------------------------
"""

import os
import sys
import gc
import time
import argparse
from multiprocessing import Pool, cpu_count
import numpy as np

# Configure paths so imports resolve whether running from project root or inside training_round5
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
MODE_FACTORS = {
    "1x": 1.0,
    "1.5x": 1.5,
    "2x": 2.0
}
DEFAULT_FILES_PER_BIN = 8
DEFAULT_SAMPLES_PER_FILE = 16384
DEFAULT_WORKERS = 4
DEFAULT_DTYPE = "float32"  # float32 saves 50% disk space and matches PyTorch defaults

BASE_PROBABILITY_COSMIC = 1.0 / 24000.0

# Base parameters from Round 5 Specification (width_range and wavenum_range are scaled per mode)
GENERATION_PARAMS_TEMPLATE = {
    "amplitude_range": [0.001, 0.2],
    "degree_range": [1, 16],
    "offset_range": [0.0, 1.0],
    "max_coeff": 1.0,
    "min_peak_ratio": 2.0,
    "std_range": [1, 10],
    "intensity_range_cosmic": [5.0, 20.0],
    "domain_mapping": [-1.0, 1.0],
    "min_value": 0,
    "normalize": True
}


def get_target_bins(base_bin: int, factor: float) -> int:
    """Computes the target integer bin count for a given base resolution and factor."""
    return int(round(base_bin * factor))


def build_params_for_mode(base_bin: int, factor: float) -> tuple[int, dict]:
    """
    Builds physically scaled generation parameters for a given mode and base bin count:
    - Target bins: base_bin * factor
    - Wavenumber range: [0, target_bins - 1]
    - Peak count: [0, base_bin // 5] (chemical complexity invariant to optical grating)
    - Peak width: [1.0 * factor, 100.0 * factor] (peaks span more detector pixels)
    - Cosmic ray probability: BASE_PROBABILITY_COSMIC / factor (event rate per spectrum invariant)
    """
    target_bins = get_target_bins(base_bin, factor)
    scaled_params = dict(GENERATION_PARAMS_TEMPLATE)
    scaled_params["wavenum_range"] = [0, target_bins - 1]
    scaled_params["num_peaks_range"] = [0, int(base_bin // 5)]
    scaled_params["width_range"] = [round(1.0 * factor, 1), round(100.0 * factor, 1)]
    scaled_params["probability_cosmic"] = BASE_PROBABILITY_COSMIC / factor
    return target_bins, scaled_params


def generate_and_save_chunk(args):
    """
    Multiprocessing worker task:
    1. Checks if chunk file already exists (resume capability).
    2. Generates 1 batch directly at target resolution using scaled physical parameters.
    3. Saves compressed .npz file (pure_matrix, pure_noise_cosmic_matrix, full_matrix).
    """
    (
        mode,
        factor,
        base_bin,
        target_bins,
        file_idx,
        samples_count,
        out_filepath,
        dtype_str
    ) = args

    pid = os.getpid()
    np_dtype = np.float32 if dtype_str == "float32" else np.float64

    # 1. Resume check: skip if valid file already exists
    if os.path.exists(out_filepath) and os.path.getsize(out_filepath) > 1024:
        print(f"[Worker {pid:5d}] Chunk {file_idx:02d} for {mode}/{target_bins} already exists. Skipping.")
        return mode, target_bins, file_idx, 0.0, True

    # 2. Build parameters for this specific target resolution
    _, gen_params = build_params_for_mode(base_bin, factor)

    # 3. Independent RNG per worker process
    seed = int.from_bytes(os.urandom(4), byteorder="little") ^ pid
    worker_rng = np.random.default_rng(seed)

    start_time = time.time()
    print(f"[Worker {pid:5d}] Generating {mode} | Target Bins: {target_bins} (Base {base_bin}) | Chunk {file_idx:02d} ({samples_count:,} spectra)...")

    # 4. Generate native spectra directly (authentic white noise & single-pixel cosmic rays)
    pure, noise_cosmic, full = generate_spectra(
        batch_size=samples_count,
        rng=worker_rng,
        **gen_params
    )

    if np_dtype != np.float64:
        pure = pure.astype(np_dtype)
        noise_cosmic = noise_cosmic.astype(np_dtype)
        full = full.astype(np_dtype)

    # 5. Save directly to disk
    np.savez_compressed(
        out_filepath,
        pure_matrix=pure,
        pure_noise_cosmic_matrix=noise_cosmic,
        full_matrix=full
    )

    elapsed = time.time() - start_time
    file_mb = os.path.getsize(out_filepath) / (1024 * 1024)
    print(f"[Worker {pid:5d}] Finished {mode}/{target_bins} chunk {file_idx:02d} in {elapsed:.1f}s ({file_mb:.1f} MB written)")

    del pure, noise_cosmic, full
    gc.collect()
    return mode, target_bins, file_idx, elapsed, False


# =====================================================================
# 2. MAIN CLI DISPATCHER
# =====================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Multi-Resolution Raman Spectra Dataset via Native Direct Generation (Option 1).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--output-dir", type=str, default=DEFAULT_BASE_DIR,
                        help="Target output directory for training_data5")
    parser.add_argument("--bin-sizes", type=int, nargs="+", default=DEFAULT_BIN_SIZES,
                        help="List of base bin numbers to generate")
    parser.add_argument("--modes", type=str, nargs="+", default=["1x", "1.5x", "2x"],
                        choices=["1x", "1.5x", "2x"],
                        help="Resolution modes to generate")
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

    num_base_bins = len(args.bin_sizes)
    num_modes = len(args.modes)
    spectra_per_bin_mode = args.files_per_bin * args.samples_per_file
    spectra_per_mode = num_base_bins * spectra_per_bin_mode
    total_spectra = num_modes * spectra_per_mode
    total_files = num_modes * num_base_bins * args.files_per_bin

    print("=" * 85)
    print("RAMAN SYNTHETIC DATASET GENERATION PIPELINE - ROUND 5 (OPTION 1)")
    print("=" * 85)
    print(f"Output Directory:     {args.output_dir}")
    print(f"Optical Modes:        {args.modes}")
    print(f"Base Bin Sizes:       {args.bin_sizes}")
    print(f"Files per Bin:        {args.files_per_bin}")
    print(f"Spectra per File:     {args.samples_per_file:,}")
    print(f"Spectra per Mode:     {spectra_per_mode:,}")
    print(f"Total Target Spectra: {total_spectra:,}")
    print(f"Total Target Files:   {total_files} files")
    print(f"Data Type:            {args.dtype}")
    print(f"Parallel Workers:     {args.workers}")
    print("=" * 85)
    print("\nResolution & Physical Parameter Mapping:")
    for mode in args.modes:
        factor = MODE_FACTORS[mode]
        p_cosmic = BASE_PROBABILITY_COSMIC / factor
        w_range = [round(1.0 * factor, 1), round(100.0 * factor, 1)]
        target_bins_list = [get_target_bins(b, factor) for b in args.bin_sizes]
        print(f"  Mode {mode:>4s} (factor={factor:.1f}x): Target Bins = {target_bins_list}")
        print(f"         Width Range = {w_range} | P(cosmic) = {p_cosmic:.6f} (1/{int(round(1.0/p_cosmic))})")
    print("=" * 85)

    # 1. Create directory hierarchy
    # Each mode has subfolders named after the exact target bin numbers
    for mode in args.modes:
        factor = MODE_FACTORS[mode]
        for b in args.bin_sizes:
            target_bins = get_target_bins(b, factor)
            dir_path = os.path.join(args.output_dir, mode, str(target_bins))
            os.makedirs(dir_path, exist_ok=True)

    if args.dry_run:
        print("\n[DRY RUN] Directories successfully initialized. Exiting without generating spectra.")
        return

    # 2. Build task list
    tasks = []
    for mode in args.modes:
        factor = MODE_FACTORS[mode]
        for b in args.bin_sizes:
            target_bins = get_target_bins(b, factor)
            for file_idx in range(1, args.files_per_bin + 1):
                fname = f"spectra_chunk_{file_idx:02d}.npz"
                out_filepath = os.path.join(args.output_dir, mode, str(target_bins), fname)
                tasks.append((
                    mode,
                    factor,
                    b,
                    target_bins,
                    file_idx,
                    args.samples_per_file,
                    out_filepath,
                    args.dtype
                ))

    print(f"\nLaunching {len(tasks)} generation tasks across {args.workers} workers...\n")
    start_all = time.time()

    with Pool(processes=args.workers) as pool:
        results = pool.map(generate_and_save_chunk, tasks)

    total_time = time.time() - start_all
    skipped_count = sum(1 for r in results if r[4])
    completed_count = len(results) - skipped_count

    print("\n" + "=" * 85)
    print("DATASET GENERATION COMPLETE")
    print("=" * 85)
    print(f"Completed Tasks:  {completed_count}")
    print(f"Skipped Tasks:    {skipped_count} (already existed and valid)")
    print(f"Total Time:       {total_time / 60:.2f} minutes ({total_time:.1f}s)")
    print(f"Output Location:  {args.output_dir}")
    print("=" * 85)


if __name__ == "__main__":
    main()
