import numpy as np
from astropy.modeling.models import Gaussian1D, Lorentz1D, Voigt1D, Moffat1D, Polynomial1D
from astropy.modeling import Fittable1DModel

#--------------------------------------------------------------------------------#
# For now, parameters assume each intensity bin is ~one wavenumber (cm^-1) apart #
#--------------------------------------------------------------------------------#
# generation pipeline: peaks + baseline --> noise + cosmic rays --> add together

# spectra are generated as ROW vectors!!!

# peak generation (gaussian, lorentzian, voigt, moffat)
def pick_profile(rng: np.random.Generator = None):
    """Helper function for picking a peak profile type."""
    if rng is None:
        rng = np.random.default_rng()

    profiles = [Gaussian1D, Lorentz1D, Voigt1D, Moffat1D]
    return rng.choice(profiles)

def get_voigt_amplitude(amplitude: float, fwhm_L: float, fwhm_G: float) -> float:
    """Takes in a desired general amplitude and calculates necessary amplitude_L for Voigt1D."""
    v_unit = Voigt1D(x_0=0, amplitude_L=1.0, fwhm_L=fwhm_L, fwhm_G=fwhm_G)
    unit_height = v_unit(0.0)
    amplitude_L = amplitude / unit_height
    return amplitude_L

def init_peak(
    peak_type: type[Fittable1DModel], 
    amplitude_range: list, 
    center_range: list, 
    width_range: list, 
    x: np.ndarray,
    rng: np.random.Generator = None) -> tuple[np.ndarray, float]:
    """
    Initializes an Astropy peak model, samples it immediately onto x, and returns (peak_vector, amplitude).
    Input range lists should be [min, max].
    """
    if rng is None:
        rng = np.random.default_rng()

    amplitude = rng.uniform(low=amplitude_range[0], high=amplitude_range[1])
    center = rng.uniform(low=center_range[0], high=center_range[1])

    if peak_type == Gaussian1D:
        width = rng.uniform(low=width_range[0], high=width_range[1])
        model = Gaussian1D(amplitude=amplitude, mean=center, stddev=width)
    elif peak_type == Lorentz1D:
        width = rng.uniform(low=width_range[0], high=width_range[1])
        model = Lorentz1D(amplitude=amplitude, x_0=center, fwhm=width)
    elif peak_type == Voigt1D:
        widthL = rng.uniform(low=width_range[0], high=width_range[1])
        widthG = rng.uniform(low=width_range[0], high=width_range[1])
        amplitude_L = get_voigt_amplitude(amplitude, widthL, widthG)
        model = Voigt1D(x_0=center, amplitude_L=amplitude_L, fwhm_L=widthL, fwhm_G=widthG)
    elif peak_type == Moffat1D:
        gamma = rng.uniform(low=width_range[0], high=width_range[1])
        alpha = rng.uniform(low=width_range[0], high=width_range[1])
        model = Moffat1D(amplitude=amplitude, x_0=center, gamma=gamma, alpha=alpha)
    else:
        raise ValueError(f"Unsupported peak type: {peak_type}")

    # Immediately sample model onto x vector into float64 array
    return model(x), amplitude

def generate_single_pure_spectrum(
    wavenumber_range: list, 
    num_peaks_range: list, 
    amplitude_range: list, 
    width_range: list, 
    rng: np.random.Generator = None,
    x: np.ndarray = None) -> tuple[np.ndarray, float]:
    """
    Generates a pure peak-only spectrum row vector by summing immediately sampled peak vectors.
    wavenumber_range: list of 2 values [start, end]
    num_peaks_range: list of 2 values [min_peaks, max_peaks]
    amplitude_range: list of 2 values [min_amplitude, max_amplitude]
    width_range: list of 2 values [min_width, max_width]
    rng: random number generator
    x: optional precomputed wavenumber array np.arange(start, end + 1, 1)
    Returns: tuple of (spectrum_vector, min_peak_amplitude)
    """
    if rng is None:
        rng = np.random.default_rng()

    if x is None:
        x = np.arange(wavenumber_range[0], wavenumber_range[1] + 1, 1, dtype=np.float64)

    num_peaks = rng.integers(low=num_peaks_range[0], high=num_peaks_range[1])
    spectrum_vec = np.zeros_like(x, dtype=np.float64)
    min_peak_amplitude = float('inf')

    for _ in range(num_peaks):
        peak_type = pick_profile(rng)
        peak_vec, amp = init_peak(peak_type, amplitude_range, wavenumber_range, width_range, x, rng)
        spectrum_vec += peak_vec
        if amp < min_peak_amplitude:
            min_peak_amplitude = amp

    return spectrum_vec, min_peak_amplitude

def generate_pure_spectra_batch(
    batch_size: int, 
    wavenumber_range: list, 
    num_peaks_range: list, 
    amplitude_range: list, 
    width_range: list, 
    rng: np.random.Generator = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Generates a batch of pure peak spectra directly as a 2D NumPy array.
    batch_size: number of spectra to generate
    wavenumber_range: list of 2 values [start, end]
    num_peaks_range: list of 2 values [min_peaks, max_peaks]
    amplitude_range: list of 2 values [min_amplitude, max_amplitude]
    width_range: list of 2 values [min_width, max_width]
    rng: random number generator
    Returns:
        batch_spectra: np.ndarray of shape (batch_size, bins)
        min_amplitudes: np.ndarray of shape (batch_size,) containing min peak amplitude per spectrum
    """
    if rng is None:
        rng = np.random.default_rng()

    x = np.arange(wavenumber_range[0], wavenumber_range[1] + 1, 1, dtype=np.float64)
    bins = len(x)

    batch_spectra = np.empty((batch_size, bins), dtype=np.float64)
    min_amplitudes = np.empty(batch_size, dtype=np.float64)
    
    for i in range(batch_size):
        single_spectrum_vec, min_amp = generate_single_pure_spectrum(
            wavenumber_range, num_peaks_range, amplitude_range, width_range, rng, x=x
        )
        batch_spectra[i] = single_spectrum_vec
        min_amplitudes[i] = min_amp

    return batch_spectra, min_amplitudes

# polynomial baseline generation (up to 7th order)
def generate_baseline(
    wavenumber_range: list, 
    degree: int, 
    offset_range: list = [0.0, 0.5], 
    max_coeff: float = 1.0, 
    domain_mapping: list = [-1.0, 1.0], 
    rng: np.random.Generator = None) -> Polynomial1D:
    """
    Generates a random Polynomial1D baseline with minimum shifted within offset_range.
    """
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

def generate_baseline_batch(
    batch_size: int, 
    wavenumber_range: list, 
    degree_range: list = [1, 7], 
    offset_range: list = [0.0, 0.5], 
    max_coeff: float = 1.0, 
    domain_mapping: list = [-1.0, 1.0], 
    rng: np.random.Generator = None,
    x: np.ndarray = None) -> np.ndarray:
    """
    Generates a batch of baseline-only spectra evaluated directly as a 2D NumPy array.
    batch_size: number of spectra to generate
    wavenumber_range: list of 2 values [start, end]
    degree_range: list of 2 values [min_degree, max_degree]
    offset_range: list of 2 values [min_offset, max_offset]
    max_coeff: maximum coefficient value
    domain_mapping: domain mapping to the baseline [-1.0, 1.0]
    rng: random number generator
    x: optional precomputed wavenumber array np.arange(start, end + 1, 1)
    Returns:
        batch_spectra: np.ndarray of shape (batch_size, bins)
    """
    if rng is None:
        rng = np.random.default_rng()

    if x is None:
        x = np.arange(wavenumber_range[0], wavenumber_range[1] + 1, 1, dtype=np.float64)
    bins = len(x)

    degrees = rng.integers(low=degree_range[0], high=degree_range[1] + 1, size=batch_size)
    batch_spectra = np.empty((batch_size, bins), dtype=np.float64)
    
    for i, degree_item in enumerate(degrees):
        baseline_poly = generate_baseline(
            wavenumber_range, degree_item, offset_range, max_coeff, domain_mapping, rng
        )
        batch_spectra[i] = baseline_poly(x)

    return batch_spectra

# noise and cosmic ray generation
def get_min_peak_amplitude(amplitudes) -> float:
    """
    Returns the smallest peak amplitude from a float scalar or array/list of amplitudes.
    """
    if isinstance(amplitudes, (int, float, np.floating)):
        return float(amplitudes)
    return float(np.min(amplitudes))

def gaussian_noise_vector(
    min_peak_amplitude: float, 
    bins: int = 1200, 
    min_peak_ratio: float = 2.0, 
    std_range: list = [0.01, 0.05], 
    rng: np.random.Generator = None) -> np.ndarray:
    """
    Creates a gaussian noise vector scaled by the minimum peak amplitude.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    a = get_min_peak_amplitude(min_peak_amplitude)
    noise_std = rng.uniform(std_range[0], std_range[1])
    return rng.normal(loc=0.0, scale=noise_std * a * min_peak_ratio, size=bins)

def gaussian_noise_batch(
    min_amplitudes: np.ndarray, 
    bins: int = 1200, 
    min_peak_ratio: float = 2.0, 
    std_range: list = [0.01, 0.05], 
    rng: np.random.Generator = None) -> np.ndarray:
    """
    Creates a gaussian noise matrix based off minimum peak amplitudes in a single vectorized call.
    min_amplitudes: 1D NumPy array or list of min amplitudes for each spectrum in the batch.
    """
    if rng is None:
        rng = np.random.default_rng()

    min_amps = np.asarray(min_amplitudes, dtype=np.float64)
    batch_size = len(min_amps)
    noise_stds = rng.uniform(std_range[0], std_range[1], size=batch_size)
    scales = noise_stds * min_amps * min_peak_ratio
    return rng.normal(loc=0.0, scale=scales[:, None], size=(batch_size, bins))

def add_cosmic_rays(
    batch: np.ndarray, 
    probability: float, 
    intensity_range: list = [10.0, 100.0], 
    rng: np.random.Generator = None) -> np.ndarray:
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

# Combined spectra generation
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
    min_value: float = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates a full batch of synthetic Raman spectra (pure peaks, baseline, noise, and cosmic rays),
    returning them directly as three NumPy 2D arrays: (pure_list, pure_noise_cosmic_list, full_list).

    Parameters:
    -----------
    batch_size : int
        Number of synthetic spectra to generate in the batch.
    wavenum_range : list [min_wn, max_wn]
        The wavenumber range [start, end] in cm^-1 (e.g., [0, 1200]). Step size is 1.
    num_peaks_range : list [min_peaks, max_peaks]
        Inclusive range for the random number of Raman peaks sampled per spectrum.
    amplitude_range : list [min_amp, max_amp]
        Range of peak amplitudes (intensities/heights) for each sampled peak profile.
    width_range : list [min_width, max_width]
        Range of peak widths (FWHM / standard deviation / gamma) across peak models.
    degree_range : list [min_degree, max_degree]
        Range of polynomial degrees for the synthetic baseline (e.g., [1, 7]).
    offset_range : list [min_offset, max_offset]
        Vertical offset range added to shift the baseline polynomial's absolute minimum.
    max_coeff : float
        Maximum coefficient magnitude sampled for the Chebyshev/polynomial baseline terms.
    min_peak_ratio : float
        Multiplier scaling factor applied to the smallest peak amplitude when calculating noise level.
    std_range : list [min_std, max_std]
        Relative noise standard deviation range (scaled by smallest peak amplitude * min_peak_ratio).
    probability_cosmic : float
        Per-wavenumber bin probability of encountering a cosmic ray spike (e.g., 1/12000).
    intensity_range_cosmic : list [min_spike, max_spike]
        Intensity range added to bins that trigger a cosmic ray spike.
    domain_mapping : list [min_domain, max_domain], default=[-1.0, 1.0]
        Window used to remap the wavenumber domain when evaluating baseline polynomials for numerical stability.
    rng : np.random.Generator, optional
        NumPy random number generator instance for reproducible sampling.
    min_value : float, optional
        If provided, shifts each spectrum in final_matrix along the y-axis so its minimum value equals min_value.

    Returns:
    --------
    pure_list : np.ndarray
        Shape (batch_size, bins): Pure synthesized Raman peaks only.
    pure_noise_cosmic_list : np.ndarray
        Shape (batch_size, bins): Pure peaks + Gaussian noise + Cosmic ray spikes (no baseline).
    full_list : np.ndarray
        Shape (batch_size, bins): Pure peaks + Polynomial baseline + Gaussian noise + Cosmic ray spikes.
    """
    if rng is None:
        rng = np.random.default_rng()
        
    # 1. Generate x vector (wavenumbers) with step size of 1
    x = np.arange(wavenum_range[0], wavenum_range[1] + 1, 1, dtype=np.float64)
    bins = len(x)
    
    # 2. Generate pure spectra and min amplitudes with immediate sampling
    pure_intensities, min_amplitudes = generate_pure_spectra_batch(
        batch_size=batch_size,
        wavenumber_range=wavenum_range,
        num_peaks_range=num_peaks_range,
        amplitude_range=amplitude_range,
        width_range=width_range,
        rng=rng
    )
    
    # 3. Generate baseline batch evaluated directly on x
    baseline_intensities = generate_baseline_batch(
        batch_size=batch_size,
        wavenumber_range=wavenum_range,
        degree_range=degree_range,
        offset_range=offset_range,
        max_coeff=max_coeff,
        domain_mapping=domain_mapping,
        rng=rng,
        x=x
    )
    
    # 4. Add clean spectra
    clean_spectra_matrix = pure_intensities + baseline_intensities
    
    # 5. Generate noise and cosmic rays
    noise_matrix = gaussian_noise_batch(
        min_amplitudes=min_amplitudes,
        bins=bins,
        min_peak_ratio=min_peak_ratio,
        std_range=std_range,
        rng=rng
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
    
    # 7. Shift final_matrix if min_value is specified
    if min_value is not None:
        mins = np.min(final_matrix, axis=1, keepdims=True)
        final_matrix = final_matrix - mins + min_value

    # 8. Return matrices directly
    return pure_intensities, pure_noise_cosmic_matrix, final_matrix
