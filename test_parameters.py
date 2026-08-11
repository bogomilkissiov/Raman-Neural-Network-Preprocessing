import numpy as np
import matplotlib.pyplot as plt
from astropy.modeling.models import Gaussian1D, Lorentz1D, Voigt1D, Moffat1D
from spectra_generator import generate_spectra


# 1. Define x vector from -900 to +900 with step size of 1
x = np.arange(-900, 901, 1)

# 2. Shared peak parameters
amp = 1.0
center = 0.0
width = 30  # Characteristic scale/width parameter

# 3. Instantiate each peak profile with comparable width/scale parameters
g_peak = Gaussian1D(amplitude=amp, mean=center, stddev=width)
l_peak = Lorentz1D(amplitude=amp, x_0=600, fwhm=width)

# Voigt amplitude scaling to match target peak height amp=1.0 at center
v_unit = Voigt1D(x_0=center, amplitude_L=1.0, fwhm_L=5, fwhm_G=5)
v_amp_L = amp / v_unit(center)
v_peak = Voigt1D(x_0=250, amplitude_L=v_amp_L, fwhm_L=5, fwhm_G=5)

m_peak = Moffat1D(amplitude=amp, x_0=-250, gamma=30, alpha=1)

# 4. Evaluate profiles over x vector
y_g = g_peak(x)
y_l = l_peak(x)
y_v = v_peak(x)
y_m = m_peak(x)

# 5. Plot profiles
plt.figure(figsize=(10, 6), dpi=120)
plt.plot(x, y_g, label="Gaussian1D", color="#1f77b4", linewidth=2)
plt.plot(x, y_l, label="Lorentz1D", color="#ff7f0e", linewidth=2)
plt.plot(x, y_v, label="Voigt1D", color="#2ca02c", linewidth=2)
plt.plot(x, y_m, label="Moffat1D", color="#d62728", linewidth=2)

plt.title("Comparison of Astropy 1D Peak Profiles", fontsize=14, fontweight="bold")
plt.xlabel("Wavenumber Shift (cm⁻¹)", fontsize=12)
plt.ylabel("Intensity / Amplitude", fontsize=12)
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend(fontsize=11)
plt.tight_layout()

plt.savefig("peak_profiles_comparison.png")

# looks like width/fhwm scales can be within a similar range
# min width = 1
# max width = 30 -> Reason: (1200 * 0.025)

# alpha controls how wide the tails are
# min alpha = 1
# max alpha = 30

# gamma controls how wide the peak is (similar to fwhm)
# min gamma = 1
# max gamma = 30

# basically just put width/fhwm/gamma in a similar range...

    


