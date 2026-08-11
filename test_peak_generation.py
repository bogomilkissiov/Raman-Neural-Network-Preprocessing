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
    single_peak = g.init_peak(profile, amplitude_range, center_range, width_range, rng)
    x = np.arange(0, 1201, 1)
    y = single_peak(x)

    plt.figure(figsize=(10, 5), dpi=120)
    plt.plot(x, y, label=f"Single Peak ({single_peak.__class__.__name__})", color="#1f77b4")
    plt.title(f"Test Initialized Peak: {single_peak.__class__.__name__}")
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
    single_spectrum, single_spectrum_list = g.generate_single_pure_spectrum(center_range, num_peaks_range, amplitude_range, width_range, rng)
    y = single_spectrum(x)

    plt.figure(figsize=(10, 5), dpi=120)
    plt.plot(x, y, label=f"Single Spectrum ({single_spectrum.__class__.__name__})", color="#1f77b4")
    plt.title(f"Test Generated Single Spectrum: {single_spectrum.__class__.__name__}")
    plt.xlabel("Wavenumber (cm⁻¹)")
    plt.ylabel("Intensity")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()
    print("\n")

    # Test generate_pure_spectra_batch
    print(f"Testing generate_pure_spectra_batch...")
    batch_spectrum, batch_spectrum_list = g.generate_pure_spectra_batch(100, center_range, num_peaks_range, amplitude_range, width_range, rng)
    random_choice = batch_spectrum[69]
    y = random_choice(x)

    plt.figure(figsize=(10, 5), dpi=120)
    plt.plot(x, y, label=f"Random Choice ({random_choice.__class__.__name__})", color="#1f77b4")
    plt.title(f"Test Random Choice: {random_choice.__class__.__name__}")
    plt.xlabel("Wavenumber (cm⁻¹)")
    plt.ylabel("Intensity")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()
    print("\n")

    print(f"Test complete!")
