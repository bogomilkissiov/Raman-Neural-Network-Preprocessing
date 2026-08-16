import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import gc
import numpy as np
from spectra_generator import generate_spectra

num_files = 1
samples_per_file = 1000
base_dir = "generated_spectra"
output_dir = base_dir
counter = 1
while os.path.exists(output_dir):
    output_dir = f"{base_dir}({counter})"
    counter += 1
os.makedirs(output_dir)
print(f"\n--- Saving dataset to newly created folder: {output_dir} ---\n")

for i in range(num_files):
    filename = os.path.join(output_dir, f"dataset_part_{i+1}.npz")
    print(f"Generating file {i+1}/{num_files}: {filename}...")
    
    pure, noise_cosmic, full = generate_spectra(
        batch_size=samples_per_file,
        wavenum_range=[0, 1200],          
        num_peaks_range=[10, 100],
        amplitude_range=[0.009, 0.1],
        width_range=[2, 30],
        degree_range=[1, 7],
        offset_range=[0.0, 2.0],
        max_coeff=1.0,
        min_peak_ratio=2.5,
        std_range=[0.3, 0.7],
        probability_cosmic=1/12000,
        intensity_range_cosmic=[3.0, 10.0]
    )
    
    # Save immediately
    np.savez_compressed(
        filename,
        pure_matrix=pure,
        pure_noise_cosmic_matrix=noise_cosmic,
        full_matrix=full
    )
    print(f"Saved {filename}! Memory cleared.")

    # Explicitly free memory and collect garbage after each file
    del pure, noise_cosmic, full
    gc.collect()
