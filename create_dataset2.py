import os
import sys
import gc
import time
import numpy as np
from multiprocessing import Pool

# Add parent directory if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from spectra_generator import generate_spectra

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
num_files = 16
samples_per_file = 16384
num_workers = 2
base_dir = "generated_spectra"

# Parameters for spectra generation
GENERATION_PARAMS = {
    "wavenum_range": [0, 1015],
    "num_peaks_range": [10, 100],
    "amplitude_range": [0.009, 0.1],
    "width_range": [2, 30],
    "degree_range": [1, 7],
    "offset_range": [0.0, 2.0],
    "max_coeff": 1.0,
    "min_peak_ratio": 2.5,
    "std_range": [0.3, 0.7],
    "probability_cosmic": 1 / 12000,
    "intensity_range_cosmic": [3.0, 10.0],
    "domain_mapping": [-1.0, 1.0],
    "min_value": 0}

def generate_and_save_chunk(args):
    """
    Worker task function for multiprocessing.
    Generates a batch of spectra and saves it directly to a compressed .npz file.
    """
    file_idx, total_files, samples_count, output_filepath, gen_params = args
    pid = os.getpid()
    
    # Initialize a process-distinct random number generator with unique seed
    seed = int.from_bytes(os.urandom(4), byteorder="little") ^ pid
    worker_rng = np.random.default_rng(seed)
    
    start_time = time.time()
    print(f"[Process {pid}] Starting chunk {file_idx}/{total_files} ({samples_count:,} samples) -> {output_filepath}")
    
    pure, noise_cosmic, full = generate_spectra(
        batch_size=samples_count,
        rng=worker_rng,
        **gen_params)
    
    # Save immediately to disk
    np.savez_compressed(
        output_filepath,
        pure_matrix=pure,
        pure_noise_cosmic_matrix=noise_cosmic,
        full_matrix=full)
    
    file_size_mb = os.path.getsize(output_filepath) / (1024 * 1024)
    elapsed = time.time() - start_time
    print(f"[Process {pid}] Finished chunk {file_idx}/{total_files} ({file_size_mb:.2f} MB) in {elapsed:.2f}s! Memory cleared.")
    
    # Explicitly release arrays and force garbage collection
    del pure, noise_cosmic, full
    gc.collect()
    
    return output_filepath, elapsed

def main():
    # -------------------------------------------------------------------------
    # 1. Output directory creation (auto-increment if exists)
    # -------------------------------------------------------------------------
    output_dir = base_dir
    counter = 1
    while os.path.exists(output_dir):
        output_dir = f"{base_dir}({counter})"
        counter += 1
    os.makedirs(output_dir, exist_ok=True)
    
    total_samples = num_files * samples_per_file
    print("=" * 70)
    print("DATASET GENERATION PIPELINE")
    print("=" * 70)
    print(f"Output Directory:    {output_dir}")
    print(f"Total Chunks:        {num_files}")
    print(f"Samples per Chunk:   {samples_per_file:,}")
    print(f"Total Dataset Size:  {total_samples:,} spectra")
    print(f"Parallel Workers:    {num_workers}")
    print("=" * 70 + "\n")

    # -------------------------------------------------------------------------
    # 2. Prepare task arguments
    # -------------------------------------------------------------------------
    tasks = []
    for i in range(num_files):
        filename = os.path.join(output_dir, f"dataset_part_{i+1}.npz")
        tasks.append((i + 1, num_files, samples_per_file, filename, GENERATION_PARAMS))
    
    overall_start = time.time()
    
    # -------------------------------------------------------------------------
    # 3. Multiprocessing Pool Execution (2 Workers)
    # -------------------------------------------------------------------------
    with Pool(processes=num_workers) as pool:
        results = pool.map(generate_and_save_chunk, tasks)
        
    overall_elapsed = time.time() - overall_start
    print("\n" + "=" * 70)
    print("GENERATION COMPLETE")
    print("=" * 70)
    print(f"All {num_files} files ({total_samples:,} total samples) successfully generated!")
    print(f"Total Time Elapsed:  {overall_elapsed:.2f} seconds ({overall_elapsed / 60:.2f} minutes)")
    print(f"Average Throughput:  {total_samples / overall_elapsed:.1f} spectra/second")
    print(f"Saved Directory:     {output_dir}")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()