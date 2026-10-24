"""
DATASET GENERATION PIPELINE - ROUND 6 (1x MODE MULTI-BIN DATASETS)
--------------------------------------------------------------------------------
Generates variable-bin Raman datasets (1.0x native optical sampling):
- 5 Variable Bin Sizes: 800, 1016, 1200, 1400, 1600 bins

MODULES:
1. Training Dataset (training_data6/):
   - 8 chunk files per bin size, 16,384 spectra per file:
     * 131,072 spectra per bin size
     * 655,360 spectra total across 40 files
   - Automatic pre-check: if training dataset is already complete, skips generation entirely.

2. Testing Dataset (testing_data6/):
   - 1 chunk file per bin size, 4,096 spectra per file (configurable):
     * 4,096 spectra per bin size
     * 20,480 spectra total across 5 files
   - Dedicated held-out test split generated with identical physics and independent RNG.
   - Automatic pre-check: skips if testing dataset already exists.

PHYSICAL SCALING PRINCIPLES:
1. Optical sampling factor fixed at 1.0x across all bin sizes.
2. Refined peak width range: [1.0, 20.0] bins (sharp Raman bands, distinct from baseline).
3. Chemical peak count varies with spectral window: num_peaks_range = [0, base_bins // 5].
4. Peak amplitude range: [0.005, 0.2] (prevents sub-noise micro-peaks).
5. Authentic, pixel-independent detector white noise across all resolutions.
6. Invariant cosmic ray event probability: probability_cosmic = 1.0 / 24000.0.
7. Single-pixel cosmic ray hits remain authentic delta spikes.

DIRECTORY STRUCTURE:
training_round6/
├── training_data6/
│   ├── 800/  (spectra_chunk_01.npz .. spectra_chunk_08.npz)
│   ├── 1016/
│   ├── 1200/
│   ├── 1400/
│   └── 1600/
└── testing_data6/
    ├── 800/  (spectra_chunk_01.npz)
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
import argparse
from multiprocessing import Pool, cpu_count
import numpy as np

# Configure paths so imports resolve whether running from project root or inside training_round6
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
for path in [PROJECT_ROOT, SCRIPT_DIR]:
    if path not in sys.path:
        sys.path.insert(0, path)

from spectra_generator import generate_spectra

# =====================================================================
# 1. CONFIGURATION & DATASET PARAMETERS (ROUND 6)
# =====================================================================
DEFAULT_TRAIN_DIR = os.path.join(SCRIPT_DIR, "training_data6")
DEFAULT_TEST_DIR = os.path.join(SCRIPT_DIR, "testing_data6")
DEFAULT_BIN_SIZES = [800, 1016, 1200, 1400, 1600]
MODE_FACTOR = 1.0

# Training split defaults
DEFAULT_TRAIN_FILES_PER_BIN = 8
DEFAULT_TRAIN_SAMPLES_PER_FILE = 16384

# Testing split defaults
DEFAULT_TEST_FILES_PER_BIN = 1
DEFAULT_TEST_SAMPLES_PER_FILE = 4096

DEFAULT_WORKERS = max(1, min(cpu_count() - 1, 8))
DEFAULT_DTYPE = "float32"

BASE_PROBABILITY_COSMIC = 1.0 / 24000.0

# Physical generation template (includes refined amplitude and width bounds)
GENERATION_PARAMS_TEMPLATE = {
    "amplitude_range": [0.005, 0.2],
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


def build_params_for_bin(bins: int) -> dict:
    """
    Builds simulation parameters for a specific bin size at 1x optical mode:
    - Wavenumber range: [0, bins - 1]
    - Peak count range: [0, bins // 5]
    - Peak width range: [1.0, 20.0]
    - Peak amplitude range: [0.005, 0.2]
    - Cosmic ray probability: 1 / 24000
    """
    params = dict(GENERATION_PARAMS_TEMPLATE)
    params["wavenum_range"] = [0, bins - 1]
    params["num_peaks_range"] = [0, int(bins // 5)]
    params["width_range"] = [1.0, 20.0]
    params["probability_cosmic"] = BASE_PROBABILITY_COSMIC
    return params


def verify_dataset_status(base_dir: str, bin_sizes: list[int], files_per_bin: int) -> tuple[bool, int, int]:
    """
    Checks if all expected chunk files exist and are valid (>1024 bytes).
    Returns: (is_complete, existing_valid_files, total_expected_files)
    """
    total_expected = len(bin_sizes) * files_per_bin
    existing_valid = 0

    for b in bin_sizes:
        bin_dir = os.path.join(base_dir, str(b))
        if not os.path.exists(bin_dir):
            continue
        for file_idx in range(1, files_per_bin + 1):
            out_filename = f"spectra_chunk_{file_idx:02d}.npz"
            out_filepath = os.path.join(bin_dir, out_filename)
            if os.path.exists(out_filepath) and os.path.getsize(out_filepath) > 1024:
                existing_valid += 1

    is_complete = (existing_valid == total_expected)
    return is_complete, existing_valid, total_expected


def generate_and_save_chunk(args):
    """
    Multiprocessing worker task:
    1. Checks if chunk file already exists (resume capability).
    2. Generates 1 batch directly at target resolution using independent RNG.
    3. Saves compressed .npz file (pure_matrix, pure_noise_cosmic_matrix, full_matrix).
    """
    (
        split_name,
        bins,
        file_idx,
        samples_count,
        out_filepath,
        dtype_str,
        force_regen
    ) = args

    pid = os.getpid()
    np_dtype = np.float32 if dtype_str == "float32" else np.float64

    # 1. Resume check: skip if valid file already exists (unless force is requested)
    if not force_regen and os.path.exists(out_filepath) and os.path.getsize(out_filepath) > 1024:
        print(f"[{split_name} | Worker {pid:5d}] Chunk {file_idx:02d} for {bins} bins already exists. Skipping.")
        return split_name, bins, file_idx, 0.0, True

    # 2. Build parameters for this specific bin resolution
    gen_params = build_params_for_bin(bins)

    # 3. Independent RNG per worker process
    seed = int.from_bytes(os.urandom(4), byteorder="little") ^ pid
    worker_rng = np.random.default_rng(seed)

    start_time = time.time()
    print(f"[{split_name} | Worker {pid:5d}] Generating {bins} bins | Chunk {file_idx:02d} ({samples_count:,} spectra)...")

    # 4. Generate native spectra directly
    pure, noise_cosmic, full = generate_spectra(
        batch_size=samples_count,
        rng=worker_rng,
        **gen_params
    )

    if np_dtype != np.float64:
        pure = pure.astype(np_dtype)
        noise_cosmic = noise_cosmic.astype(np_dtype)
        full = full.astype(np_dtype)

    # 5. Save compressed arrays directly to disk
    os.makedirs(os.path.dirname(out_filepath), exist_ok=True)
    np.savez_compressed(
        out_filepath,
        pure_matrix=pure,
        pure_noise_cosmic_matrix=noise_cosmic,
        full_matrix=full
    )

    elapsed = time.time() - start_time
    file_mb = os.path.getsize(out_filepath) / (1024 * 1024)
    print(f"[{split_name} | Worker {pid:5d}] Finished {bins} bins chunk {file_idx:02d} in {elapsed:.1f}s ({file_mb:.1f} MB written)")

    del pure, noise_cosmic, full
    gc.collect()
    return split_name, bins, file_idx, elapsed, False


# =====================================================================
# 2. MAIN CLI DISPATCHER
# =====================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Variable-Bin Raman Spectra Datasets (Training & Testing, 1x Mode - Round 6).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Directories
    parser.add_argument("--train-dir", type=str, default=DEFAULT_TRAIN_DIR,
                        help="Target output directory for training dataset")
    parser.add_argument("--test-dir", type=str, default=DEFAULT_TEST_DIR,
                        help="Target output directory for testing dataset")

    # Resolution & Shared Parameters
    parser.add_argument("--bin-sizes", type=int, nargs="+", default=DEFAULT_BIN_SIZES,
                        help="List of bin numbers to generate")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                        help="Number of parallel multiprocessing worker processes")
    parser.add_argument("--dtype", type=str, choices=["float32", "float64"], default=DEFAULT_DTYPE,
                        help="Data type for saved matrices")

    # Training Split Parameters
    parser.add_argument("--train-files-per-bin", type=int, default=DEFAULT_TRAIN_FILES_PER_BIN,
                        help="Number of chunk files per bin size for training")
    parser.add_argument("--train-samples-per-file", type=int, default=DEFAULT_TRAIN_SAMPLES_PER_FILE,
                        help="Number of spectra per chunk file for training")

    # Testing Split Parameters
    parser.add_argument("--test-files-per-bin", type=int, default=DEFAULT_TEST_FILES_PER_BIN,
                        help="Number of chunk files per bin size for testing")
    parser.add_argument("--test-samples-per-file", type=int, default=DEFAULT_TEST_SAMPLES_PER_FILE,
                        help="Number of spectra per chunk file for testing")

    # Selective Execution Flags
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip generating the training dataset")
    parser.add_argument("--skip-test", action="store_true",
                        help="Skip generating the testing dataset")
    parser.add_argument("--force", action="store_true",
                        help="Force regeneration of existing chunk files")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print directory plan and status checks without generating")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 80)
    print("RAMAN SYNTHETIC DATASET GENERATION PIPELINE - ROUND 6")
    print("=" * 80)
    print(f"Optical Mode:         1.0x (Native sampling)")
    print(f"Bin Sizes:            {args.bin_sizes}")
    print(f"Peak Width Range:     [1.0, 20.0] bins")
    print(f"Peak Amplitude Range: [0.005, 0.2]")
    print(f"Data Type:            {args.dtype}")
    print(f"Parallel Workers:     {args.workers}")
    print("=" * 80)

    tasks = []

    # -----------------------------------------------------------------
    # MODULE 1: TRAINING DATASET CHECK & TASK PLANNING
    # -----------------------------------------------------------------
    train_complete, train_existing, train_total = verify_dataset_status(
        args.train_dir, args.bin_sizes, args.train_files_per_bin
    )

    print(f"\n[1/2] Training Dataset Status ('{os.path.basename(args.train_dir)}'):")
    print(f"      Path:             {args.train_dir}")
    print(f"      Files Per Bin:    {args.train_files_per_bin} ({args.train_samples_per_file:,} spectra/file)")
    print(f"      Target Spectra:   {len(args.bin_sizes) * args.train_files_per_bin * args.train_samples_per_file:,}")
    print(f"      Existing Chunks:  {train_existing} / {train_total}")

    if args.skip_train:
        print("      Status:           SKIPPED (--skip-train specified)")
    elif train_complete and not args.force:
        print(f"      Status:           ALREADY COMPLETE ({train_existing}/{train_total} valid chunks found). SKIPPING.")
    else:
        if args.force:
            print("      Status:           REGENERATING ALL (--force specified)")
        else:
            print(f"      Status:           INCOMPLETE ({train_total - train_existing} chunks needed). Scheduling generation...")

        for b in args.bin_sizes:
            bin_dir = os.path.join(args.train_dir, str(b))
            for file_idx in range(1, args.train_files_per_bin + 1):
                out_filename = f"spectra_chunk_{file_idx:02d}.npz"
                out_filepath = os.path.join(bin_dir, out_filename)
                if args.force or not (os.path.exists(out_filepath) and os.path.getsize(out_filepath) > 1024):
                    tasks.append((
                        "Train",
                        b,
                        file_idx,
                        args.train_samples_per_file,
                        out_filepath,
                        args.dtype,
                        args.force
                    ))

    # -----------------------------------------------------------------
    # MODULE 2: TESTING DATASET CHECK & TASK PLANNING
    # -----------------------------------------------------------------
    test_complete, test_existing, test_total = verify_dataset_status(
        args.test_dir, args.bin_sizes, args.test_files_per_bin
    )

    print(f"\n[2/2] Testing Dataset Status ('{os.path.basename(args.test_dir)}'):")
    print(f"      Path:             {args.test_dir}")
    print(f"      Files Per Bin:    {args.test_files_per_bin} ({args.test_samples_per_file:,} spectra/file)")
    print(f"      Target Spectra:   {len(args.bin_sizes) * args.test_files_per_bin * args.test_samples_per_file:,}")
    print(f"      Existing Chunks:  {test_existing} / {test_total}")

    if args.skip_test:
        print("      Status:           SKIPPED (--skip-test specified)")
    elif test_complete and not args.force:
        print(f"      Status:           ALREADY COMPLETE ({test_existing}/{test_total} valid chunks found). SKIPPING.")
    else:
        if args.force:
            print("      Status:           REGENERATING ALL (--force specified)")
        else:
            print(f"      Status:           INCOMPLETE ({test_total - test_existing} chunks needed). Scheduling generation...")

        for b in args.bin_sizes:
            bin_dir = os.path.join(args.test_dir, str(b))
            for file_idx in range(1, args.test_files_per_bin + 1):
                out_filename = f"spectra_chunk_{file_idx:02d}.npz"
                out_filepath = os.path.join(bin_dir, out_filename)
                if args.force or not (os.path.exists(out_filepath) and os.path.getsize(out_filepath) > 1024):
                    tasks.append((
                        "Test",
                        b,
                        file_idx,
                        args.test_samples_per_file,
                        out_filepath,
                        args.dtype,
                        args.force
                    ))

    print("\n" + "-" * 80)
    print(f"Total Chunks Scheduled to Generate: {len(tasks)}")
    print("-" * 80)

    if not tasks:
        print("\nAll requested datasets already exist and are complete! Nothing to generate.")
        print("(Use '--force' if you explicitly wish to overwrite existing files.)\n")
        return

    if args.dry_run:
        print("\n[DRY RUN] Plan complete. No files were generated.")
        return

    # -----------------------------------------------------------------
    # 3. EXECUTE GENERATION VIA MULTIPROCESSING POOL
    # -----------------------------------------------------------------
    start_total = time.time()
    print(f"\nLaunching {args.workers} worker processes for {len(tasks)} tasks...")

    with Pool(processes=args.workers) as pool:
        results = pool.map(generate_and_save_chunk, tasks)

    total_time = time.time() - start_total
    num_generated = sum(1 for r in results if not r[4])
    num_skipped = sum(1 for r in results if r[4])

    print("\n" + "=" * 80)
    print("DATASET GENERATION SUMMARY - ROUND 6")
    print("=" * 80)
    print(f"Total Execution Time: {total_time:.2f}s ({total_time / 60.0:.2f} minutes)")
    print(f"Newly Generated:      {num_generated} chunk files")
    print(f"Skipped (Existed):    {num_skipped} chunk files")
    print(f"Training Directory:   {os.path.abspath(args.train_dir)}")
    print(f"Testing Directory:    {os.path.abspath(args.test_dir)}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
