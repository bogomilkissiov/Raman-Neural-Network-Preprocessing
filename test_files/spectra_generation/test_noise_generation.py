import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import spectra_generator as g
# pyrefly: ignore [missing-import]
from astropy.modeling.models import Gaussian1D, Lorentz1D, Voigt1D, Moffat1D, Polynomial1D
# pyrefly: ignore [missing-import]
from astropy.modeling import Fittable1DModel
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt

if __name__ == '__main__':
    # Want to test get_min_peak_amplitude, gaussian_noise_vector, gaussian_noise_batch, add_cosmic_rays

    rng = np.random.default_rng(seed=13)
    
    amplitude_range = [0,1]
    center_range = [0, 1200]
    width_range = [2, 30]
    num_peaks_range = [5, 10]
    batch_spectra, batch_min_amps = g.generate_pure_spectra_batch(100, center_range, num_peaks_range, amplitude_range, width_range, rng)
    x = np.arange(0, 1201, 1)
    random_choice_min_amp = batch_min_amps[69]
    random_choice = batch_spectra[69]
    
    print("Testing min_peak_amplitude:")
    min_peak_amplitude = g.get_min_peak_amplitude(random_choice_min_amp)
    
    print(f"min_peak_amplitude: {min_peak_amplitude}")

    print("\n")

    # Test gaussian_noise_vector
    print("Testing gaussian_noise_vector:")
    noise_vector = g.gaussian_noise_vector(random_choice_min_amp, bins=1201, min_peak_ratio=2.0, std_range=[0.1, 0.5], rng=rng)
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), dpi=120, sharex=True)

    axes[0].plot(x, random_choice, label="Random Choice", color="#1f77b4")
    axes[0].set_title("Clean Spectrum")
    axes[0].set_ylabel("Intensity")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    axes[1].plot(x, noise_vector, label="Noise Vector", color="#ff7f0e")
    axes[1].set_title("Gaussian Noise Vector")
    axes[1].set_ylabel("Intensity")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()

    axes[2].plot(x, random_choice + noise_vector, label="Spectrum + Noise", color="#2ca02c")
    axes[2].set_title("Spectrum + Noise")
    axes[2].set_xlabel("Wavenumber (cm⁻¹)")
    axes[2].set_ylabel("Intensity")
    axes[2].grid(True, linestyle="--", alpha=0.5)
    axes[2].legend()

    plt.tight_layout()
    plt.show()
    print("Plot successfully made.")
    print("\n")

    # Test gaussian_noise_batch
    print("Testing gaussian_noise_batch:")
    noise_batch = g.gaussian_noise_batch(batch_min_amps, bins=1201, min_peak_ratio=2.0, std_range=[0.1, 0.5], rng=rng)
    pure_intensities = batch_spectra
    mix = pure_intensities + noise_batch

    random_index = np.random.randint(0, 100)
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), dpi=120, sharex=True)

    axes[0].plot(x, pure_intensities[random_index], label="Pure Spectrum", color="#1f77b4")
    axes[0].set_title("Clean Spectrum")
    axes[0].set_ylabel("Intensity")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    axes[1].plot(x, noise_batch[random_index], label="Noise Vector", color="#ff7f0e")
    axes[1].set_title("Gaussian Noise Vector")
    axes[1].set_ylabel("Intensity")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()

    axes[2].plot(x, mix[random_index], label="Spectrum + Noise", color="#2ca02c")
    axes[2].set_title("Spectrum + Noise")
    axes[2].set_xlabel("Wavenumber (cm⁻¹)")
    axes[2].set_ylabel("Intensity")
    axes[2].grid(True, linestyle="--", alpha=0.5)
    axes[2].legend()

    plt.tight_layout()
    plt.show()
    print("Plot successfully made.")
    print("\n")

    # Test add_cosmic_rays
    print("Testing add_cosmic_rays:")
    rays_and_noise = g.add_cosmic_rays(noise_batch, 1/120, intensity_range=[5,10], rng=rng)

    random_index = np.random.randint(0, 100)
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), dpi=120, sharex=True)

    axes[0].plot(x, noise_batch[random_index], label="Noise Vector", color="#1f77b4")
    axes[0].set_title("Clean Spectrum")
    axes[0].set_ylabel("Intensity")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    axes[1].plot(x, rays_and_noise[random_index] - noise_batch[random_index], label="Cosmic Rays", color="#ff7f0e")
    axes[1].set_title("Cosmic Rays")
    axes[1].set_ylabel("Intensity")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()

    axes[2].plot(x, rays_and_noise[random_index], label="Spectrum + Noise", color="#2ca02c")
    axes[2].set_title("Spectrum + Noise")
    axes[2].set_xlabel("Wavenumber (cm⁻¹)")
    axes[2].set_ylabel("Intensity")
    axes[2].grid(True, linestyle="--", alpha=0.5)
    axes[2].legend()

    plt.tight_layout()
    plt.show()
    print("Plot successfully made.")
    print("\n")
