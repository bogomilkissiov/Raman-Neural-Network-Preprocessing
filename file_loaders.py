import numpy as np
from renishawWiRE import WDFReader
from spectra_class import spectrum, spectra

# Fix for renishawWiRE strict origin validation error on certain map scans
def _safe_parse_wmap(self):
    try:
        uid, pos, size = self.block_info["WMAP"]
    except KeyError:
        return
    self.file_obj.seek(pos + 16)
    x_start = self._WDFReader__read_type("float")
    y_start = self._WDFReader__read_type("float")
    unknown1 = self._WDFReader__read_type("float")
    x_pad = self._WDFReader__read_type("float")
    y_pad = self._WDFReader__read_type("float")
    unknown2 = self._WDFReader__read_type("float")
    spectra_w = self._WDFReader__read_type("int32")
    spectra_h = self._WDFReader__read_type("int32")
    self.map_shape = (spectra_w, spectra_h)
    self.map_info = dict(
        x_start=x_start,
        y_start=y_start,
        x_pad=x_pad,
        y_pad=y_pad,
        x_span=spectra_w * x_pad,
        y_span=spectra_h * y_pad,
        x_unit=self.xpos_unit,
        y_unit=self.ypos_unit,
    )

WDFReader._parse_wmap = _safe_parse_wmap

def wdf_to_spectra(wdf_filepath : str, pack : bool=True):
    """
    Parses a Renishaw WDF map file and returns a list of spectrum objects.
    Each spectrum object contains:
      - position_vector: [x, y] stage coordinates
      - wavenumber_vector: 1D array of wavenumbers
      - intensity_vector: 1D array of intensities
    """
    reader = WDFReader(wdf_filepath)
    wavenumbers = reader.xdata
    spectra_data = reader.spectra

    if spectra_data.ndim == 1:
        if wavenumbers.ndim == 1 and len(spectra_data) > len(wavenumbers) and len(spectra_data) % len(wavenumbers) == 0:
            spectra_data = spectra_data.reshape(-1, len(wavenumbers))
        else:
            spectra_data = spectra_data.reshape(1, -1)
    elif spectra_data.ndim > 2:
        spectra_data = spectra_data.reshape(-1, spectra_data.shape[-1])
 
    x_coords = np.ravel(reader.xpos) if reader.xpos is not None else np.zeros(len(spectra_data))
    y_coords = np.ravel(reader.ypos) if reader.ypos is not None else np.zeros(len(spectra_data))
        
    if pack:
        wavenumber_matrix = np.tile(wavenumbers, (len(spectra_data), 1)) if wavenumbers.ndim == 1 else wavenumbers
        position_matrix = np.column_stack((x_coords, y_coords)) if (reader.xpos is not None or reader.ypos is not None) else None
        return spectra.from_matrices(wavenumber_matrix, spectra_data, position_matrix)

    spectrum_list = []
    for i in range(len(spectra_data)):
        pos = [x_coords[i], y_coords[i]]
        spectrum_list.append(spectrum(wavenumbers, spectra_data[i], pos))
     
    return spectrum_list

# uses spcfile library by kogens: https://github.com/kogens/spcfile
def spc_to_spectra(spc_filepath: str, pack : bool=False):
    """
    Parses a SPC map file and returns a list of spectrum objects.
    Each spectrum object contains:
      - position_vector: [x, y] stage coordinates
      - wavenumber_vector: 1D array of wavenumbers
      - intensity_vector: 1D array of intensities
    """
    try:
        from spcfile import SPCFile
    except ImportError:
        raise ImportError("The 'spcfile' package is required to parse .spc files. Install it using 'pip install spcfile'.")

    spc = SPCFile(spc_filepath)
    
    if pack:
        valid_subs = [sub for sub in spc if not np.all(sub.y == 0)]
        if not valid_subs:
            return spectra.from_matrices([], [], None)
        
        wavenumber_matrix = np.vstack([sub.x for sub in valid_subs])
        intensity_matrix = np.vstack([sub.y for sub in valid_subs])
        positions = [[sub.subheader.get('z_value', 0.0), sub.subheader.get('w_value', 0.0)] for sub in valid_subs]
        position_matrix = np.vstack(positions)
        return spectra.from_matrices(wavenumber_matrix, intensity_matrix, position_matrix)

    spectrum_list = []
    # Standard SPC files often lose X,Y map coordinates when exported by Renishaw
    for sub in spc:
        # Skip empty padding spectra that are often added during export
        if np.all(sub.y == 0):
            continue
            
        # We try to extract Z and W if they exist, otherwise default to 0.0
        x_pos = sub.subheader.get('z_value', 0.0)
        y_pos = sub.subheader.get('w_value', 0.0)
        
        pos = [x_pos, y_pos]
        spectrum_list.append(spectrum(sub.x, sub.y, pos))
        
    return spectrum_list

def txt_to_spectra(txt_filepath: str, pack : bool=False):
    """
    Parses a TXT map file and returns a list of spectrum objects.
    The text file is expected to have columns: #X, #Y, #Wave, #Intensity
    Each spectrum object contains:
      - position_vector: [x, y] stage coordinates
      - wavenumber_vector: 1D array of wavenumbers
      - intensity_vector: 1D array of intensities
    """
    data = np.loadtxt(txt_filepath, skiprows=1)
    
    x_coords = data[:, 0]
    y_coords = data[:, 1]
    wavenumbers_all = data[:, 2]
    intensities_all = data[:, 3]
    
    diffs = (np.diff(x_coords) != 0) | (np.diff(y_coords) != 0)
    split_indices = np.where(diffs)[0] + 1
    
    x_splits = np.split(x_coords, split_indices)
    y_splits = np.split(y_coords, split_indices)
    wavenumber_splits = np.split(wavenumbers_all, split_indices)
    intensity_splits = np.split(intensities_all, split_indices)
    
    if pack:
        wavenumber_matrix = np.vstack(wavenumber_splits)
        intensity_matrix = np.vstack(intensity_splits)
        positions = [[x_arr[0], y_arr[0]] for x_arr, y_arr in zip(x_splits, y_splits)]
        position_matrix = np.vstack(positions)
        return spectra.from_matrices(wavenumber_matrix, intensity_matrix, position_matrix)

    spectrum_list = []
    for x_arr, y_arr, wave_arr, int_arr in zip(x_splits, y_splits, wavenumber_splits, intensity_splits):
        pos = [x_arr[0], y_arr[0]]
        spectrum_list.append(spectrum(wave_arr, int_arr, pos))

    return spectrum_list
