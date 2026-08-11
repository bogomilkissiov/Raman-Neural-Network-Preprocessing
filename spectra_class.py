import os
import numpy as np

############################
# SPECTRA CLASS DEFINITION #
############################
class spectra:
    def __init__(self, position=None, wavenumber=None, intensity=None):
        """
        position: (x, y) stage position coordinates for the spectrum within a map.
        wavenumber: 1D array/list containing Wavenumbers (cm^-1)
        intensity: 1D array/list containing Intensities
        """
        self._position = np.array(position) if position is not None else np.array([0.0, 0.0])
        self.wavenumber = np.array(wavenumber) if wavenumber is not None else np.array([])
        self.intensity = np.array(intensity) if intensity is not None else np.array([])
    
    @property
    def position(self):
        return self._position

    @property
    def wavenumbers(self):
        return self.wavenumber

    @property
    def intensities(self):
        return self.intensity

    @property
    def x(self):
        return self._position[0] if len(self._position) > 0 else None

    @property
    def y(self):
        return self._position[1] if len(self._position) > 1 else None
