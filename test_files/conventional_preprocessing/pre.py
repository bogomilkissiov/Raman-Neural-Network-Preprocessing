import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
from spectra_class import spectrum, spectra
from spectra_generator import generate_spectra

import copy
from scipy import sparse
from scipy.linalg import cholesky
from scipy.sparse.linalg import spsolve
from scipy.special import wofz
from scipy.optimize import curve_fit
from skimage.restoration import denoise_wavelet
import peakutils
import BaselineRemoval
from spectra_class import spectrum, spectra
import functools

# Root directory path for data files
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
data_path = os.path.join(ROOT_DIR, "test_data.npz")

# ==========================================
# DECORATOR
# ==========================================
def process_spectra_data(func):
    """
    Decorator that takes `data` (spectrum, list[spectrum], or spectra),
    extracts the 2D wavenumber and intensity matrices, passes them to the 
    underlying function, and then reconstructs/updates the original data objects.
    """
    @functools.wraps(func)
    def wrapper(data, *args, **kwargs):
        if isinstance(data, spectra):
            wavenumbers = data.wavenumber_matrix
            intensities = data.intensity_matrix
        elif isinstance(data, list):
            wavenumbers = np.vstack([s.wavenumber_vector for s in data])
            intensities = np.vstack([s.intensity_vector for s in data])
        elif isinstance(data, spectrum):
            wavenumbers = data.wavenumber_vector.reshape(1, -1)
            intensities = data.intensity_vector.reshape(1, -1)
        else:
            raise TypeError("Unsupported data type. Must be spectrum, list of spectrum, or spectra object.")
            
        new_intensities = func(wavenumbers, intensities, *args, **kwargs)
        
        if isinstance(data, spectra):
            data.intensity_matrix = new_intensities
        elif isinstance(data, list):
            for i, s in enumerate(data):
                s.intensity_vector = new_intensities[i]
        elif isinstance(data, spectrum):
            data.intensity_vector = new_intensities[0]
            
        return data
    return wrapper

# ==========================================
# SPIKE REMOVAL (Cosmic Rays & Dead Pixels)
# ==========================================
def med_z_score(intensities):
    """ Function to calculate modified Z-scores for 1D or 2D array """
    intensity_median = np.median(intensities, axis=-1, keepdims=True)
    intensity_mad = np.median(np.abs(intensities - intensity_median), axis=-1, keepdims=True)
    z_scores = 0.6745*(intensities - intensity_median) / (intensity_mad + 1e-12) 
    return z_scores

def whit_z_score(intensities, absolute=True):
    """
    Extract the differenced series (ie. y[n+1] - y[n])
    and calculate the modified z-score from it for 1D array
    """
    delta_intensity = np.diff(intensities, axis=-1) # n-1 sizing
    # insert at axis=-1
    delta_intensity = np.insert(delta_intensity, 0, 0, axis=-1)
    
    if not absolute:
        return med_z_score(delta_intensity)
    return np.abs(med_z_score(delta_intensity))

def find_adjacent_neighbors(series, item):
    """
    See which mislabeled items in series share a 
    common neighbor index with item or are truly neighbors
    """
    if not item:
        return []
    # see which neighbors have that same point
    common_neighbors = [k for k,v in series.items() if list(item.values())[0] in v]
    
    series_condensed = copy.deepcopy(series)
    for k in common_neighbors:
        del series_condensed[k]
    
    if len(common_neighbors) == 1:
        return common_neighbors
    
    try:
        lb_neighbor = min(common_neighbors)
        lb_neighbor = {lb_neighbor: min(series[lb_neighbor])}
    except Exception:
        lb_neighbor = []
        
    try:
        ub_neighbor = max(common_neighbors)
        ub_neighbor = {ub_neighbor: max(series[ub_neighbor])}
    except Exception:
        ub_neighbor = []
    
    return common_neighbors + find_adjacent_neighbors(series_condensed, lb_neighbor) + find_adjacent_neighbors(series_condensed, ub_neighbor)

def spike_removal(intensity, threshold=10, m=4):
    """ 
    Spike removal at location signifying spikes
    m is ~= window size / 2
    """
    spikes_raw = whit_z_score(intensity, absolute=False)
    spikes_dead = copy.deepcopy(spikes_raw) < -30
    spikes = np.abs(spikes_raw) > threshold
    intensity_spikeless = copy.deepcopy(intensity)
    
    spikes_in_question = {}
    mislabeled_spike_centers = []
    
    for i in np.argwhere(spikes == True).flatten():
        # find all the neighboring valid points (non-spikes)
        sub_intensity_i = np.argwhere(spikes[i-m:i+(m+1)] == 0).flatten() + (i-m)
        spikes_in_question[i] = sub_intensity_i
        
        # dead pixel spikes
        if spikes_dead[i] == True:
            pass
        # if theres only 1 valid point, probably misclassified
        elif len(sub_intensity_i) == 1:
            mislabeled_spike_centers.append({i: sub_intensity_i[0]})
    
    # remove all mislabeled points
    for spike in mislabeled_spike_centers:        
        spikey_wikeys = np.unique(find_adjacent_neighbors(spikes_in_question, spike))
        for k in spikey_wikeys:
            if k in spikes_in_question:
                del spikes_in_question[k]
    
    # average labeled points
    for k, v in spikes_in_question.items():
        if len(v) == 0:
            continue
        intensity_spikeless[k] = copy.deepcopy(np.mean(intensity[v]))
        
    return intensity_spikeless, spikes_raw

@process_spectra_data
def despike_spectra(wavenumbers, intensities, threshold=10, m=4):
    """
    Run the outlier removal step.
    Modifies data in place.
    """
    new_intensities = np.empty_like(intensities)
    for i in range(intensities.shape[0]):
        f_int, _ = spike_removal(intensities[i], threshold=threshold, m=m)
        new_intensities[i] = f_int
    return new_intensities

# ==========================================
# WAVELET DENOISING
# ==========================================
def log10_transform(intensities):
    min_val = np.min(intensities, axis=1, keepdims=True)
    shift = np.where(min_val <= 0, np.abs(min_val) + 1.0, 1.0)
    return np.log10(intensities + shift), shift

@process_spectra_data
def denoise_spectra(wavenumbers, intensities):
    """ Wavelet transform for noise reduction using BayesShrink """
    new_intensities = np.empty_like(intensities)
    log_intensity, shift = log10_transform(intensities)
    for i in range(intensities.shape[0]):
        if np.max(intensities[i]) == np.min(intensities[i]):
            new_intensities[i] = intensities[i]
            continue
        denoised_intensity = denoise_wavelet(log_intensity[i], method='BayesShrink', mode='soft', wavelet_levels=1, wavelet='coif3', rescale_sigma=True)
        new_intensities[i] = (10**denoised_intensity) - shift[i][0]
    return new_intensities

# ==========================================
# BASELINE SUBTRACTION
# ==========================================
def arpls(y, lam=1e4, ratio=0.05, itermax=100):
    """ Asymmetrically reweighted penalized least squares smoothing (Baek et al. 2015) """
    N = len(y)
    D = sparse.eye(N, format='csc')
    D = D[1:] - D[:-1]
    D = D[1:] - D[:-1]

    H = lam * D.T * D
    w = np.ones(N)
    for i in range(itermax):
        W = sparse.diags(w, 0, shape=(N, N))
        WH = sparse.csc_matrix(W + H)
        C = sparse.csc_matrix(cholesky(WH.todense()))
        z = spsolve(C, spsolve(C.T, w * y))
        d = y - z
        dn = d[d < 0]
        m = np.mean(dn)
        s = np.std(dn)
        wt = 1. / (1 + np.exp(2 * (d - (2 * s - m)) / (s + 1e-12)))
        if np.linalg.norm(w - wt) / np.linalg.norm(w) < ratio:
            break
        w = wt
    return z

@process_spectra_data
def remove_baseline(wavenumbers, intensities, order=3, method='combo', l_=100):
    """
    Determines baseline via desired methodologies:
        - 'poly'    : adaptive polynomial fitting (PeakUtils)
        - 'modpoly' : modified polynomial fitting (Zhao et al. 2007)
        - 'airpls'  : airPLS (Zhang et al. 2010)
        - 'arpls'   : arPLS (Baek et al. 2015)
        - 'combo'   : peakutils + ZhangFit
    """
    new_intensities = np.empty_like(intensities)
    for i in range(intensities.shape[0]):
        y = intensities[i]
        
        if np.max(y) == np.min(y):
            new_intensities[i] = y
            continue
            
        if method == 'airpls':
            baseobj = BaselineRemoval.BaselineRemoval(y)
            spectra_corrected = baseobj.ZhangFit(repitition=50)
        elif method == 'modpoly':
            baseobj = BaselineRemoval.BaselineRemoval(y)
            spectra_corrected = baseobj.IModPoly(order)
        elif method == 'arpls':
            baseline = arpls(y)
            spectra_corrected = y - baseline
        elif method == 'combo':
            baseline = peakutils.baseline(y, order)
            spectra_corrected = y - baseline
            baseobj = BaselineRemoval.BaselineRemoval(spectra_corrected)
            spectra_corrected2 = baseobj.ZhangFit(repitition=50, lambda_=l_)
            spectra_corrected = spectra_corrected2
        else:
            baseline = peakutils.baseline(y, order)
            spectra_corrected = y - baseline

        if np.isnan(spectra_corrected).any():
            spectra_corrected = y - np.min(y)

        new_intensities[i] = spectra_corrected
    return new_intensities

# ==========================================
# SILICON BACKGROUND REMOVAL
# ==========================================
def voigt_profile(x, x0, sigma, gamma, height=1, offset=0):
    """ Voigt function: convolution of Cauchy-Lorentz and Gaussian distribution. """
    w = wofz((x - x0 + 1j * gamma) / (sigma * np.sqrt(2)))
    return height * np.real(w) / (sigma * np.sqrt(2*np.pi)) + offset

def silicon_background_2order(x, x01, x02, x03, sigma1, sigma2, sigma3, gamma1, gamma2, gamma3, height1, height2, height3):
    return (voigt_profile(x, x01, sigma1, gamma1, height1, offset=0) +
            voigt_profile(x, x02, sigma2, gamma2, height2, offset=0) +
            voigt_profile(x, x03, sigma3, gamma3, height3, offset=0))

@process_spectra_data
def background_removal_silicon(wavenumbers, intensities):
    """ Single spectrum background removal of silicon optical phonons. """
    new_intensities = np.empty_like(intensities)
    for i in range(intensities.shape[0]):
        wn = wavenumbers[i]
        y = intensities[i]
        mask = (wn >= 800) & (wn <= 1150)
        wn_sub = wn[mask]
        data_sub = y[mask]
        
        if len(wn_sub) == 0:
            new_intensities[i] = y
            continue
        
        params_tot = 12
        lower_bounds = [0 for _ in range(params_tot)]
        upper_bounds = [np.inf for _ in lower_bounds]
        rl = [920, 950, 970]
        ru = [940, 960, 985]
        for j in range(3):     # resonance location
            lower_bounds[j] = rl[j]
            upper_bounds[j] = ru[j]
        for j in range(3):    # HWHM of gaussian (width), HWHM of lorentz (width), height
            lower_bounds[j+3] = 1
            lower_bounds[j+6] = 1
            upper_bounds[j+3] = 50
            upper_bounds[j+6] = 50
            
        try:
            popt, cov = curve_fit(silicon_background_2order, wn_sub, data_sub, bounds=(lower_bounds, upper_bounds), maxfev=100000)
            data_background = silicon_background_2order(wn, *popt)
            new_intensities[i] = y - data_background
        except Exception:
            new_intensities[i] = y
            
    return new_intensities

# ==========================================
# NORMALIZATION
# ==========================================
@process_spectra_data
def normalize_spectra(wavenumbers, intensities):
    """ Normalize spectra by computing the z-score based off of the mean. """
    intensity_mean = np.mean(intensities, axis=1, keepdims=True)
    intensity_std = np.std(intensities, axis=1, keepdims=True)
    z_scores = (intensities - intensity_mean) / (intensity_std + 1e-12)
    return z_scores

# ==========================================
# NON-NEGATIVE / MIN-ZERO SHIFT
# ==========================================
@process_spectra_data
def shift_to_zero(wavenumbers, intensities):
    """ Shifts each spectrum upwards so that its minimum intensity is at y=0. """
    min_vals = np.min(intensities, axis=1, keepdims=True)
    return intensities - min_vals

# ==========================================
# PREPROCESSING PIPELINE
# ==========================================
def preprocess_pipeline(data):
    """
    Takes in a single spectrum object, a list of spectrum objects, or a spectra object.
    Runs them through the preprocessing pipeline modifying the objects in place.
    Pipeline steps: despike -> denoise -> baseline removal -> normalization -> shift_to_zero
    """
    despike_spectra(data)
    denoise_spectra(data)
    remove_baseline(data)
    normalize_spectra(data)
    shift_to_zero(data)
    return data
