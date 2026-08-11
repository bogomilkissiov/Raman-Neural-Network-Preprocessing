import numpy as np
import torch
from spectra_class import spectra

# for parallel spectra generation
import os
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat

# SciPy for signal processing, noise, and fitting functions
from scipy import signal
from scipy.special import voigt_profile  # Alternative direct math implementation for Voigt profiles

# Astropy for 1D peak models (Gaussian, Lorentzian, Voigt, Moffat)
from astropy.modeling.models import Gaussian1D, Lorentz1D, Voigt1D, Moffat1D, Polynomial1D
from astropy.modeling import Fittable1DModel

#--------------------------------------------------------------------------------#
# For now, parameters assume each intensity bin is ~one wavenumber (cm^-1) apart #
#--------------------------------------------------------------------------------#
# generation pipeline: peaks + baseline --> noise + cosmic rays --> add together

# peak generation (gaussian, lorentzian, voigt, moffat)
def pick_profile(rng : np.random.Generator = None):
    "helper function for picking a peak type"
    if rng is None:
        rng = np.random.default_rng()

    profiles = [Gaussian1D, Lorentz1D, Voigt1D, Moffat1D]
    return rng.choice(profiles)

def get_voigt_amplitude(amplitude : float, fwhm_L : float, fwhm_G : float) -> float:
    "Takes in a desired general amplitude and calculates necessary amplitude_L for Voigt1D."
    v_unit = Voigt1D(x_0=0, amplitude_L=1.0, fwhm_L=fwhm_L, fwhm_G=fwhm_G)
    unit_height = v_unit(0.0)
    amplitude_L = amplitude / unit_height
    return amplitude_L

def init_peak(peak_type : type[Fittable1DModel], amplitude_range : list, center_range : list, width_range : list, rng : np.random.Generator = None) -> Fittable1DModel:
    """helper function to take in a peak type + parameter range and return initialized peak.
    Input range lists should be [min, max].
    """
    if rng is None:
        rng = np.random.default_rng()

    if peak_type == Gaussian1D:
        amplitude = rng.uniform(low=amplitude_range[0], high=amplitude_range[1])
        center = rng.uniform(low=center_range[0], high=center_range[1])
        width = rng.uniform(low=width_range[0], high=width_range[1])
        return Gaussian1D(amplitude=amplitude, mean=center, stddev=width)
    
    elif peak_type == Lorentz1D:
        amplitude = rng.uniform(low=amplitude_range[0], high=amplitude_range[1])
        center = rng.uniform(low=center_range[0], high=center_range[1])
        width = rng.uniform(low=width_range[0], high=width_range[1])
        return Lorentz1D(amplitude=amplitude, x_0=center, fwhm=width)
    
    elif peak_type == Voigt1D:
        amplitude = rng.uniform(low=amplitude_range[0], high=amplitude_range[1])
        center = rng.uniform(low=center_range[0], high=center_range[1])
        widthL = rng.uniform(low=width_range[0], high=width_range[1])
        widthG = rng.uniform(low=width_range[0], high=width_range[1])
        amplitude_L = get_voigt_amplitude(amplitude, widthL, widthG)
        return Voigt1D(x_0=center, amplitude_L=amplitude_L, fwhm_L=widthL, fwhm_G=widthG)
    
    elif peak_type == Moffat1D:
        amplitude = rng.uniform(low=amplitude_range[0], high=amplitude_range[1])
        center = rng.uniform(low=center_range[0], high=center_range[1])
        gamma = rng.uniform(low=width_range[0], high=width_range[1])
        alpha = rng.uniform(low=width_range[0], high=width_range[1])
        return Moffat1D(amplitude=amplitude, x_0=center, gamma=gamma, alpha=alpha)

def generate_single_pure_spectrum(wavenumber_range : list, num_peaks_range : list, amplitude_range : list, width_range : list, rng : np.random.Generator = None) -> tuple[Fittable1DModel, list[Fittable1DModel]]:
    """
    This function will generate a peak-only spectra based on the parameters you pass.
    wavnumber_range: list of 2 values [start, end]
    num_peaks_range: list of 2 values [min_peaks, max_peaks]
    amplitude_range: list of 2 values [min_amplitude, max_amplitude]
    width_range: list of 2 values [min_width, max_width]
    rng: random number generator
    Returns: tuple of (sum(pure_spectra), pure_spectra) where pure_spectra is list of individual peaks
    """
    if rng is None:
        rng = np.random.default_rng()

    num_peaks = rng.integers(low=num_peaks_range[0], high=num_peaks_range[1])
    pure_spectra = []

    for i in range(num_peaks):
        peak_type = pick_profile(rng)
        peak = init_peak(peak_type, amplitude_range, wavenumber_range, width_range, rng)
        pure_spectra.append(peak)

    return sum(pure_spectra[1:], start=pure_spectra[0]), pure_spectra

def generate_pure_spectra_batch(batch_size : int, wavenumber_range : list, num_peaks_range : list, amplitude_range : list, width_range : list, rng : np.random.Generator = None, n_jobs : int = -1) -> tuple[list[Fittable1DModel], list[list[Fittable1DModel]]]:
    """
    This function will generate a batch of peak-only spectra based on the parameters you pass, running in parallel.
    batch_size: number of spectra to generate
    wavenumber_range: list of 2 values [start, end]
    num_peaks_range: list of 2 values [min_peaks, max_peaks]
    amplitude_range: list of 2 values [min_amplitude, max_amplitude]
    width_range: list of 2 values [min_width, max_width]
    rng: random number generator
    n_jobs: number of parallel processes to run (-1 for all cores)
    """

    if rng is None:
        rng = np.random.default_rng()

    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1

    # Generate distinct random generators for each process
    seeds = rng.integers(0, 2**31 - 1, size=batch_size)
    rngs = [np.random.default_rng(seed) for seed in seeds]
    
    # ProcessPoolExecutor.map can take multiple iterables. 
    # repeat() passes the static arguments infinitely, while rngs dictates the iteration length.
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        results = list(executor.map(
            generate_single_pure_spectrum,
            repeat(wavenumber_range),
            repeat(num_peaks_range),
            repeat(amplitude_range),
            repeat(width_range),
            rngs
        ))
    batch_spectra, batch_spectra_lists = zip(*results)
    return batch_spectra, batch_spectra_lists

# polynomial baseline generation (up to 7th order)
def generate_baseline(wavenumber_range: list, degree: int, offset_range: list = [0.0, 0.5], max_coeff: float = 1.0, domain_mapping: list = [-1.0, 1.0], rng: np.random.Generator = None) -> Polynomial1D:
    
    if rng is None:
        rng = np.random.default_rng()
        
    # 1. Sample random coefficients for the domain mapping window
    coeffs = {f"c{i}": rng.uniform(-max_coeff, max_coeff) for i in range(degree + 1)}
    
    # 2. Build Astropy model with domain mapping [w_min, w_max] -> domain_mapping
    poly = Polynomial1D(degree=degree, domain=wavenumber_range, window=domain_mapping, **coeffs)
    
    # 3. Find the minimum across the wavenumber range
    num_pts = max(200, int(wavenumber_range[1] - wavenumber_range[0]))
    sample_x = np.linspace(wavenumber_range[0], wavenumber_range[1], num_pts)
    min_val = np.min(poly(sample_x))
    
    # 4. Shift the polynomial so its absolute minimum falls within offset_range
    desired_min = rng.uniform(offset_range[0], offset_range[1])
    poly.c0 = poly.c0.value - min_val + desired_min
        
    return poly

def generate_baseline_batch(batch_size, wavenumber_range: list, degree_range: list = [1, 7], offset_range: list = [0.0, 0.5], max_coeff: float = 1.0, domain_mapping: list = [-1.0, 1.0], rng: np.random.Generator = None, n_jobs: int = -1) -> list[Polynomial1D]:
    """
    This function will generate a batch of baseline-only spectra based on the parameters you pass, running in parallel.
    batch_size: number of spectra to generate
    wavenumber_range: list of 2 values [start, end]
    degree_range: list of 2 values [min_degree, max_degree] to randomly select degrees from
    offset_range: list of 2 values [min_offset, max_offset]
    max_coeff: maximum coefficient value
    domain_mapping: domain mapping to the baseline [-1.0, 1.0]
    rng: random number generator
    n_jobs: number of parallel processes to run (-1 for all cores)
    """

    if rng is None:
        rng = np.random.default_rng()

    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1

    # Pre-select random degrees for each baseline in the batch
    degrees = rng.integers(low=degree_range[0], high=degree_range[1] + 1, size=batch_size)

    # Generate distinct random generators for each process
    seeds = rng.integers(0, 2**31 - 1, size=batch_size)
    rngs = [np.random.default_rng(seed) for seed in seeds]
    
    # ProcessPoolExecutor.map can take multiple iterables. 
    # repeat() passes the static arguments infinitely, while degrees and rngs dictate the iteration length.
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        batch_spectra = list(executor.map(
            generate_baseline,
            repeat(wavenumber_range),
            degrees,
            repeat(offset_range),
            repeat(max_coeff),
            repeat(domain_mapping),
            rngs
        ))

    return batch_spectra

# noise and cosmic ray generation
def get_min_peak_amplitude(pure_spectra_list : list[Fittable1DModel]) -> float:
    """Returns the smallest peak height among a list of peaks from generate_single_pure_spectrum."""
    
    peak_heights = []
    
    for peak in pure_spectra_list:
        if isinstance(peak, Voigt1D):
            # Voigt1D has no single peak parameter; evaluate at its center x_0
            center = peak.x_0.value
            peak_heights.append(peak(center))
        elif isinstance(peak, (Gaussian1D, Lorentz1D, Moffat1D)):
            # Access the amplitude Parameter object and extract its float value
            peak_heights.append(peak.amplitude.value)
            
    return min(peak_heights)

def gaussian_noise_vector(pure_spectra_list, bins : int=1200, min_peak_ratio : float=2.0, std_range: list = [0.01, 0.05], rng: np.random.Generator = None) -> np.ndarray:
    """
    Creates a gaussian noise vector based off the pure spectra astropy functions.
    Variance of the noise is random but also scaled off of the min peak amplitude of the input pure spectra.
    std_range: tuple of 2 values [min_std, max_std] in the context of the minimal peak amplitude of the input pure spectra
    """
    if rng is None:
        rng = np.random.default_rng()
    
    a = get_min_peak_amplitude(pure_spectra_list)
    
    # Sample a random standard deviation (noise level)
    noise_std = rng.uniform(std_range[0], std_range[1])
    
    # Generate Gaussian white noise centered at 0
    gaussian_noise = rng.normal(loc=0.0, scale=noise_std * a * min_peak_ratio, size=bins)
    
    return gaussian_noise

def gaussian_noise_batch(pure_spectra_batch_list, bins : int=1200, min_peak_ratio : float=2.0, std_range: list = [0.01, 0.05], rng: np.random.Generator = None, n_jobs: int = -1) -> np.ndarray:
    """
    Creates gaussian noise matrix based off the pure spectra astropy functions.
    The gaussian noise vectors generated by gaussian_noise_vector are rows in the returned matrix.
    """
    if rng is None:
        rng = np.random.default_rng()

    if n_jobs == -1:
        n_jobs = os.cpu_count() or 1

    # Generate distinct random generators for each process
    seeds = rng.integers(0, 2**31 - 1, size=len(pure_spectra_batch_list))
    rngs = [np.random.default_rng(seed) for seed in seeds]
    
    # ProcessPoolExecutor.map can take multiple iterables. 
    # repeat() passes the static arguments infinitely, while rngs dictates the iteration length.
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        batch_noise = list(executor.map(
            gaussian_noise_vector,
            pure_spectra_batch_list,
            repeat(bins),
            repeat(min_peak_ratio),
            repeat(std_range),
            rngs
        ))

    return np.array(batch_noise)

def add_cosmic_rays(batch : np.ndarray, probability : float, intensity_range : list = [10.0, 100.0], rng : np.random.Generator = None) -> np.ndarray:
    """
    Takes in a matrix of intensity (row) vectors and uses probability to randomly
    add cosmic rays to the matrix.
    """
    if rng is None:
        rng = np.random.default_rng()
    result = batch.copy()
    
    # 1. Generate boolean mask
    mask = rng.random(size=batch.shape) < probability
    num_hits = np.count_nonzero(mask)
    
    # 2. Sample intensities only for the locations that triggered True
    if num_hits > 0:
        ray_intensities = rng.uniform(intensity_range[0], intensity_range[1], size=num_hits)
        result[mask] += ray_intensities
    return result

def generate_spectra(
    batch_size: int,
    wavenum_range: list,
    num_peaks_range: list,
    amplitude_range: list,
    width_range: list,
    degree_range: list,
    offset_range: list,
    max_coeff: float,
    min_peak_ratio: float,
    std_range: list,
    probability_cosmic: float,
    intensity_range_cosmic: list,
    domain_mapping: list = [-1.0, 1.0],
    rng: np.random.Generator = None,
    n_jobs: int = -1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates a full batch of synthetic spectra, including baseline, noise, and cosmic rays,
    and returns them directly as numpy arrays.
    """
    if rng is None:
        rng = np.random.default_rng()
        
    # 1. Generate x vector (wavenumbers) with step size of 1
    x = np.arange(wavenum_range[0], wavenum_range[1] + 1, 1)
    bins = len(x)
    
    # 2. Generate pure spectra and baseline batches (Astropy models)
    pure_models, pure_models_lists = generate_pure_spectra_batch(
        batch_size=batch_size,
        wavenumber_range=wavenum_range,
        num_peaks_range=num_peaks_range,
        amplitude_range=amplitude_range,
        width_range=width_range,
        rng=rng,
        n_jobs=n_jobs
    )
    
    baseline_models = generate_baseline_batch(
        batch_size=batch_size,
        wavenumber_range=wavenum_range,
        degree_range=degree_range,
        offset_range=offset_range,
        max_coeff=max_coeff,
        domain_mapping=domain_mapping,
        rng=rng,
        n_jobs=n_jobs
    )
    
    # 3. Evaluate models on the x vector to get intensity matrices
    pure_intensities = np.array([model(x) for model in pure_models])
    baseline_intensities = np.array([model(x) for model in baseline_models])
    
    # 4. Add them together
    clean_spectra_matrix = pure_intensities + baseline_intensities
    
    # 5. Generate noise and cosmic rays
    noise_matrix = gaussian_noise_batch(
        pure_spectra_batch_list=pure_models_lists,
        min_peak_ratio=min_peak_ratio,
        bins=bins,
        std_range=std_range,
        rng=rng,
        n_jobs=n_jobs
    )
    
    noise_and_cosmic_matrix = add_cosmic_rays(
        batch=noise_matrix,
        probability=probability_cosmic,
        intensity_range=intensity_range_cosmic,
        rng=rng
    )
    
    # 6. Sum components for different outputs
    pure_noise_cosmic_matrix = pure_intensities + noise_and_cosmic_matrix
    final_matrix = clean_spectra_matrix + noise_and_cosmic_matrix
    
    # 7. Return matrices directly
    return pure_intensities, pure_noise_cosmic_matrix, final_matrix


# POTENTIALLY MIGHT WANT TO UPDATE THE BASELINE BATCH GENERATION FUNCTION TO
# RANDOMIZE THE DOMAIN_RANGE INSTEAD OF HARDCODING IT...

# adding peaks together right away in generate_single_pure spectrum
# might be causing issues in the get_min_peak_amplitude function...
# might have to update the functions so that the pure spectra stay as a function of lists
# until necessary or so that the pure spectra generation functions output a tuple containing
# min amplitude (peak) and the multifunction astropy oobject...
# idk...
