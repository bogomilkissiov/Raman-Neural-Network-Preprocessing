import os
import gc
import time
import numpy as np
from multiprocessing import Pool, cpu_count
from spectra_generator import generate_spectra

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
num_files = 10
samples_per_file = 20000
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
    "min_value": 0
}

def generate_and_save_chunk(args):
    """
    Worker task function for multiprocessing.
    Generates a batch of spectra and saves it to a compressed .npz file.
    """
    file_idx, total_files, samples_count, output_filepath, gen_params = args
    pid = os.getpid()
    
    # Initialize a process-distinct random number generator to ensure unique seeds
    seed = int.from_bytes(os.urandom(4), byteorder="little") ^ pid
    worker_rng = np.random.default_rng(seed)
    
    start_time = time.time()
    print(f"[Process {pid}] Starting generation for file {file_idx}/{total_files} ({samples_count} samples) -> {output_filepath}")
    
    pure, noise_cosmic, full = generate_spectra(
        batch_size=samples_count,
        rng=worker_rng,
        **gen_params
    )
    
    # Save directly to disk
    np.savez_compressed(
        output_filepath,
        pure_matrix=pure,
        pure_noise_cosmic_matrix=noise_cosmic,
        full_matrix=full
    )
    
    elapsed = time.time() - start_time
    print(f"[Process {pid}] Saved {output_filepath} in {elapsed:.2f}s! (Memory cleared)")
    
    del pure, noise_cosmic, full
    gc.collect()
    
    return output_filepath, elapsed

def main():
    # -------------------------------------------------------------------------
    # 1. Output directory creation
    # -------------------------------------------------------------------------
    output_dir = base_dir
    counter = 1
    while os.path.exists(output_dir):
        output_dir = f"{base_dir}({counter})"
        counter += 1
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n--- Saving dataset to newly created folder: {output_dir} ---")
    
    # -------------------------------------------------------------------------
    # 2. Worker / Core Allocation
    # -------------------------------------------------------------------------
    # Detect available CPU cores (e.g. up to 128+ cores on DGX / server nodes)
    total_cpus = cpu_count()
    workers = min(num_files, total_cpus)
    print(f"Detected {total_cpus} CPU cores. Launching {workers} worker processes in parallel.\n")

    # -------------------------------------------------------------------------
    # 3. Prepare task arguments
    # -------------------------------------------------------------------------
    tasks = []
    for i in range(num_files):
        filename = os.path.join(output_dir, f"dataset_part_{i+1}.npz")
        tasks.append((i + 1, num_files, samples_per_file, filename, GENERATION_PARAMS))
    
    overall_start = time.time()
    
    # -------------------------------------------------------------------------
    # 4. Multiprocessing Pool Execution
    # -------------------------------------------------------------------------
    with Pool(processes=workers) as pool:
        results = pool.map(generate_and_save_chunk, tasks)
        
    overall_elapsed = time.time() - overall_start
    print(f"\n=======================================================")
    print(f"All {num_files} files ({num_files * samples_per_file:,} total samples) successfully generated!")
    print(f"Total time elapsed: {overall_elapsed:.2f} seconds ({overall_elapsed/60:.2f} minutes)")
    print(f"Output directory: {output_dir}")
    print(f"=======================================================\n")

if __name__ == "__main__":
    main()
