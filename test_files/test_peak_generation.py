import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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

    # Want to test pick_profile, get_voigt_amplitude, init_peak, 
    # generate_single_pure_spectrum, generate_pure_spectra_batch

    rng = np.random.default_rng(seed=13)

    # Test pick_profile
    profile = g.pick_profile(rng)
    print(f"pick_profile output: {profile}")

    print("\n")

    # Test get_voigt_amplitude
    amp = g.get_voigt_amplitude(amplitude=1, fwhm_L=1, fwhm_G=1)
    test_voigt = Voigt1D(amplitude_L=amp, x_0=0, fwhm_L=1, fwhm_G=1)
    test_amp = test_voigt(0)
    print(f"test_get_voigt_amplitude output: 1.0")
    print(f"Calculated amplitude: {amp}")
    print(f"Test amplitude: {test_amp}")

    print("\n")

    # Test init_peak
    print(f"Testing init_peak...")

    amplitude_range = [0,1]
    center_range = [0, 1200]
    width_range = [2, 30]
    x = np.arange(0, 1201, 1)
    single_peak_vec, peak_amp = g.init_peak(profile, amplitude_range, center_range, width_range, x, rng)

    plt.figure(figsize=(10, 5), dpi=120)
    plt.plot(x, single_peak_vec, label=f"Single Peak ({profile.__name__})", color="#1f77b4")
    plt.title(f"Test Initialized Peak: {profile.__name__} (amp: {peak_amp:.4f})")
    plt.xlabel("Wavenumber (cm⁻¹)")
    plt.ylabel("Intensity")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()
    print("\n")

    # Test generate_single_pure_spectrum
    print(f"Testing generate_single_pure_spectrum...")
    num_peaks_range = [5, 10]
    single_spectrum_vec, min_peak_amplitude = g.generate_single_pure_spectrum(center_range, num_peaks_range, amplitude_range, width_range, rng, x=x)

    plt.figure(figsize=(10, 5), dpi=120)
    plt.plot(x, single_spectrum_vec, label="Single Pure Spectrum (Vector)", color="#1f77b4")
    plt.title(f"Test Generated Single Spectrum (min amp: {min_peak_amplitude:.4f})")
    plt.xlabel("Wavenumber (cm⁻¹)")
    plt.ylabel("Intensity")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()
    print("\n")

    # Test generate_pure_spectra_batch
    print(f"Testing generate_pure_spectra_batch...")
    batch_spectra, batch_min_amps = g.generate_pure_spectra_batch(100, center_range, num_peaks_range, amplitude_range, width_range, rng)
    random_choice = batch_spectra[69]

    plt.figure(figsize=(10, 5), dpi=120)
    plt.plot(x, random_choice, label="Random Choice (Vector)", color="#1f77b4")
    plt.title(f"Test Random Choice (min amp: {batch_min_amps[69]:.4f})")
    plt.xlabel("Wavenumber (cm⁻¹)")
    plt.ylabel("Intensity")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()
    print("\n")

    print(f"Test complete!")
