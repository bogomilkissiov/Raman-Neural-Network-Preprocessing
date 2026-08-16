import os
import numpy as np

############################
# SPECTRUM CLASS DEFINITION #
############################
class spectrum:
    def __init__(self, wavenumber_vector, intensity_vector, position_vector=None):
        """
        wavenumber_vector: 1D numpy array containing wavenumbers (cm^-1)
        intensity_vector: 1D numpy array containing intensities
        position_vector: optional (x, y) stage position coordinates for the spectrum within a map.
        """
        self.position_vector = np.array(position_vector) if position_vector is not None else None
        self.wavenumber_vector = np.array(wavenumber_vector)
        self.intensity_vector = np.array(intensity_vector)

    @property
    def wavenumbers(self):
        return self.wavenumber_vector

    @wavenumbers.setter
    def wavenumbers(self, value):
        self.wavenumber_vector = np.array(value)

    @property
    def intensities(self):
        return self.intensity_vector

    @intensities.setter
    def intensities(self, value):
        self.intensity_vector = np.array(value)

    @property
    def x(self):
        return self.position_vector[0] if self.position_vector is not None else None

    @property
    def y(self):
        return self.position_vector[1] if self.position_vector is not None else None


############################
# SPECTRA CLASS DEFINITION #
############################
class spectra:
    def __init__(self, spectrum_list):
        """
        takes in list of spectrum objects and unpacks them into row matrices.
        """
        wavenumbers = []
        intensities = []
        positions = []
        has_positions = True
        
        for s in spectrum_list:
            wavenumbers.append(s.wavenumber_vector)
            intensities.append(s.intensity_vector)
            if s.position_vector is not None:
                positions.append(s.position_vector)
            else:
                has_positions = False
                    
        # np.vstack ensures 1D vectors are placed into rows
        self.wavenumber_matrix = np.vstack(wavenumbers)
        self.intensity_matrix = np.vstack(intensities)
            
        if has_positions and len(spectrum_list) > 0:
            self.position_matrix = np.vstack(positions)
        else:
            self.position_matrix = None

    @classmethod
    def from_matrices(cls, wavenumber_matrix, intensity_matrix, position_matrix=None):
        """
        Fast initialization path that bypasses the list unpacking in __init__.
        """
        obj = cls.__new__(cls)
        obj.wavenumber_matrix = np.atleast_2d(wavenumber_matrix)
        obj.intensity_matrix = np.atleast_2d(intensity_matrix)
        obj.position_matrix = np.atleast_2d(position_matrix) if position_matrix is not None else None
        return obj

    def __add__(self, other):
        """
        Stacks two spectra objects row-wise to form a combined spectra object.
        """
        if not isinstance(other, spectra):
            return NotImplemented

        new_wavenumbers = np.vstack([self.wavenumber_matrix, other.wavenumber_matrix])
        new_intensities = np.vstack([self.intensity_matrix, other.intensity_matrix])

        if self.position_matrix is not None and other.position_matrix is not None:
            new_positions = np.vstack([self.position_matrix, other.position_matrix])
        else:
            new_positions = None

        return spectra.from_matrices(new_wavenumbers, new_intensities, new_positions)

    def __radd__(self, other):
        """
        Enables Python's built-in sum() on collections of spectra objects.
        """
        if other == 0:
            return self
        return self.__add__(other)

    def unpack(self):
        """
        Unpacks the matrices in this `spectra` object back into a list of individual `spectrum` objects.
        """
        spectrum_list = []
        num_spectra = self.wavenumber_matrix.shape[0]
        
        for i in range(num_spectra):
            w_vec = self.wavenumber_matrix[i]
            i_vec = self.intensity_matrix[i]
            p_vec = self.position_matrix[i] if self.position_matrix is not None else None
            spectrum_list.append(spectrum(w_vec, i_vec, p_vec))
            
        return spectrum_list

    @classmethod
    def from_wdf(cls, wdf_filepath: str):
        """Constructs a `spectra` object directly from a wdf file."""
        from file_loaders import wdf_to_spectra
        res = wdf_to_spectra(wdf_filepath, pack=True)
        return res if isinstance(res, cls) else cls(res)

    @classmethod
    def from_spc(cls, spc_filepath: str):
        """Constructs a `spectra` object directly from a spc file."""
        from file_loaders import spc_to_spectra
        res = spc_to_spectra(spc_filepath, pack=True)
        return res if isinstance(res, cls) else cls(res)

    @classmethod
    def from_txt(cls, txt_filepath: str):
        """Constructs a `spectra` object directly from a txt file."""
        from file_loaders import txt_to_spectra
        res = txt_to_spectra(txt_filepath, pack=True)
        return res if isinstance(res, cls) else cls(res)


    @property
    def wavenumbers(self):
        return self.wavenumber_matrix

    @wavenumbers.setter
    def wavenumbers(self, value):
        self.wavenumber_matrix = np.atleast_2d(value)

    @property
    def intensities(self):
        return self.intensity_matrix

    @intensities.setter
    def intensities(self, value):
        self.intensity_matrix = np.atleast_2d(value)

    @property
    def positions(self):
        return self.position_matrix

    @property
    def x(self):
        return self.position_matrix[:, 0] if self.position_matrix is not None else None

    @property
    def y(self):
        return self.position_matrix[:, 1] if self.position_matrix is not None else None