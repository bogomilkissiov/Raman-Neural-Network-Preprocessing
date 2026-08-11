import spectra_generator as g
# pyrefly: ignore [missing-import]
from astropy.modeling.models import Gaussian1D, Lorentz1D, Voigt1D, Moffat1D, Polynomial1D
# pyrefly: ignore [missing-import]
from astropy.modeling import Fittable1DModel
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # want to test generate_baseline and generate_baseline_batch
    wavenumber_range = [0, 1200]
    degree = 7
    offset_range = [0, 1]
    max_coef = 1
    domain_mapping = [-1,1]
    rng = np.random.default_rng(seed=13)
    x = np.arange(0, 1201, 1)

    # Single baseline generation
    print("Generating single test baseline...")
    baseline = g.generate_baseline(wavenumber_range, degree, offset_range, max_coef, domain_mapping, rng)
    y = baseline(x)

    plt.figure(figsize=(10, 5), dpi=120)
    plt.plot(x, y, label=f"Baseline (Polynomial Degree {degree})", color="#ff7f0e")
    plt.title(f"Test Generated Baseline (Degree {degree})")
    plt.xlabel("Wavenumber (cm⁻¹)")
    plt.ylabel("Intensity")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()
    print("\n")

    # Batch baseline generation
    print("Testing batch baseline generation...")
    batch_size = 100
    degree_range = [1,7]
    batch_baselines = g.generate_baseline_batch(batch_size, wavenumber_range, degree_range, offset_range, max_coef, domain_mapping, rng)
    random_baseline = batch_baselines[80]
    y2 = random_baseline(x)

    plt.figure(figsize=(10, 5), dpi=120)
    plt.plot(x, y2, label=f"Random Baseline", color="#ff7f0e")
    plt.title(f"Test Random Baseline")
    plt.xlabel("Wavenumber (cm⁻¹)")
    plt.ylabel("Intensity")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()
    print("\n")


    

