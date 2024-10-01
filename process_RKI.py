import os                       # Module for interacting with the operating system
import sys                      # Module for accessing command-line arguments and system functions
import numpy as np              # Numerical computing library for arrays and mathematical functions
import h5py                     # Library for working with HDF5 files
from scipy.signal import savgol_filter       # For smoothing data using the Savitzky-Golay filter
from scipy.stats import binned_statistic     # For binning data into histograms
from scipy import sparse                    # Sparse matrix library (not used in this script)
from scipy.linalg import norm               # For calculating vector norms (not used in this script)
from tqdm import tqdm            # For displaying progress bars in loops

class SpectrumObject:
    def __init__(self, mz=None, intensity=None):
        # Initialize the SpectrumObject with m/z and intensity arrays
        self.mz = mz
        self.intensity = intensity

    @staticmethod
    def tof2mass(ML1, ML2, ML3, TOF):
        # Convert Time-of-Flight (TOF) data to mass-to-charge ratio (m/z) values using calibration parameters
        A = ML3
        B = np.sqrt(1e12 / ML1)
        C = ML2 - TOF

        if A == 0:
            # Use linear calibration formula if A is zero
            return (C * C) / (B * B)
        else:
            # Use quadratic calibration formula otherwise
            return ((-B + np.sqrt((B * B) - (4 * A * C))) / (2 * A)) ** 2

    @classmethod
    def from_bruker(cls, acqu_file, fid_file):
        # Read a spectrum from Bruker format files ('acqu' and 'fid')
        with open(acqu_file, "rb") as f:
            # Read acquisition parameters from the 'acqu' file
            lines = [line.decode("utf-8", errors="replace").rstrip() for line in f]

        # Initialize variables for calibration parameters
        TD = DELAY = DW = ML1 = ML2 = ML3 = BYTORDA = None

        # Parse necessary parameters from the 'acqu' file
        for l in lines:
            if l.startswith("##$TD"):
                TD = int(l.split("= ")[1])          # Time domain (number of data points)
            if l.startswith("##$DELAY"):
                DELAY = int(l.split("= ")[1])       # Delay before acquisition starts
            if l.startswith("##$DW"):
                DW = float(l.split("= ")[1])        # Dwell time between data points
            if l.startswith("##$ML1"):
                ML1 = float(l.split("= ")[1])       # Calibration parameter ML1
            if l.startswith("##$ML2"):
                ML2 = float(l.split("= ")[1])       # Calibration parameter ML2
            if l.startswith("##$ML3"):
                ML3 = float(l.split("= ")[1])       # Calibration parameter ML3
            if l.startswith("##$BYTORDA"):
                BYTORDA = int(l.split("= ")[1])     # Byte order indicator (0: little-endian, 1: big-endian)

        # Ensure all necessary calibration parameters were found
        if None in [TD, DELAY, DW, ML1, ML2, ML3, BYTORDA]:
            raise ValueError("Missing calibration parameters in 'acqu' file.")

        # Read intensity values from the 'fid' file with correct byte order
        intensity = np.fromfile(fid_file, dtype={0: "<i", 1: ">i"}[BYTORDA])

        if len(intensity) < TD:
            # Adjust TD if the intensity data is shorter than expected
            TD = len(intensity)

        # Calculate Time-of-Flight (TOF) values for each data point
        TOF = DELAY + np.arange(TD) * DW

        # Convert TOF to mass-to-charge (m/z) values using the calibration function
        mass = cls.tof2mass(ML1, ML2, ML3, TOF)

        # Set any negative intensity values to zero (since they are not physically meaningful)
        intensity[intensity < 0] = 0

        # Return a SpectrumObject instance with the calculated m/z and intensity values
        return cls(mz=mass, intensity=intensity)

def preprocess_spectrum(spectrum):
    # Preprocess a spectrum by applying several steps: variance stabilization, smoothing, baseline correction, and normalization

    # Variance Stabilization: Apply square root transformation to stabilize the variance across intensity values
    spectrum.intensity = np.sqrt(spectrum.intensity)

    # Smoothing: Apply a Savitzky-Golay filter to smooth the intensity values
    # Window length is 11 points, and polynomial order is 3
    spectrum.intensity = savgol_filter(spectrum.intensity, window_length=11, polyorder=3)

    # Baseline Correction: Remove background noise using a simple rolling ball algorithm (moving average)
    def rolling_ball(y, window_size):
        # Compute the moving average (baseline) over the specified window size
        baseline = np.convolve(y, np.ones(window_size) / window_size, mode='same')
        return baseline

    baseline = rolling_ball(spectrum.intensity, window_size=100)
    # Subtract the estimated baseline from the intensity values
    spectrum.intensity = spectrum.intensity - baseline
    # Set any negative intensity values (after baseline correction) to zero
    spectrum.intensity[spectrum.intensity < 0] = 0

    # Normalization: Scale the intensity values so that the total area under the curve sums to 1 (Total Ion Current normalization)
    total_intensity = np.sum(spectrum.intensity)
    if total_intensity > 0:
        spectrum.intensity = spectrum.intensity / total_intensity

    # Return the preprocessed spectrum
    return spectrum

def bin_spectrum(spectrum, mz_start=2000, mz_end=20000, bin_size=50):
    # Bin the spectrum into fixed-width m/z bins
    # Default m/z range is from 2000 to 20000 with a bin size of 3 units

    # Create bin edges from mz_start to mz_end with specified bin_size
    bins = np.arange(mz_start, mz_end + bin_size, bin_size)
    # Compute the sum of intensities within each bin
    binned_intensity, _, _ = binned_statistic(
        spectrum.mz, spectrum.intensity, statistic='sum', bins=bins
    )
    # Calculate the center of each bin
    bin_centers = bins[:-1] + bin_size / 2
    # Return the bin centers and the corresponding binned intensities
    return bin_centers, binned_intensity

def RKI_raw_to_h5(RKI_ROOT, outfile):
    # Read raw spectra from the RKI dataset, preprocess them, and save to an HDF5 file

    spectra_list = []         # List to store SpectrumObject instances
    labels = []               # List to store species labels corresponding to each spectrum
    species_labels_set = set()  # Set to store unique species labels
    locs = []                 # List to store the file paths (locations) of each spectrum

    print("Reading raw spectra...")
    # Walk through the directory tree starting from RKI_ROOT
    for root, dirs, files in os.walk(RKI_ROOT):
        for dir_name in tqdm(dirs):
            # Construct the full path to the current directory
            dir_path = os.path.join(root, dir_name)
            # Define paths to the 'acqu' and 'fid' files within the directory
            acqu_file = os.path.join(dir_path, "acqu")
            fid_file = os.path.join(dir_path, "fid")
            # Check if both files exist in the directory
            if os.path.exists(acqu_file) and os.path.exists(fid_file):
                # Read the spectrum from the Bruker files
                spectrum = SpectrumObject.from_bruker(acqu_file, fid_file)
                # Append the spectrum to the list of spectra
                spectra_list.append(spectrum)
                # Normalize the path to ensure consistent separators
                normalized_path = os.path.normpath(dir_path)
                # Split the path into components based on the OS-specific separator
                path_components = normalized_path.split(os.sep)
                # Find the index of 'RKI_ROOT' in the path components
                try:
                    root_idx = path_components.index('RKI_ROOT')
                    # Extract the species label from the path components
                    # Adjust the indices based on your directory structure
                    # Example: species is located at root_idx + 2 (two levels below 'RKI_ROOT')
                    species = path_components[root_idx + 2]
                except (ValueError, IndexError):
                    # Handle cases where 'RKI_ROOT' is not found or index is out of range
                    species = 'Unknown'
                     
                # Extract the species label from the directory structure
                # Assumes that the parent directory name is the species label
                # species = os.path.basename(os.path.dirname(dir_path))
                labels.append(species)              # Add species label to the list
                species_labels_set.add(species)     # Add species label to the set of unique labels

                # Store the file path (location) of the spectrum
                locs.append(dir_path)

    # Create a sorted list of unique species labels
    species_labels = sorted(species_labels_set)
    # Create a mapping from species labels to integer indices
    species_to_idx = {species: idx for idx, species in enumerate(species_labels)}
    # Convert species labels to their corresponding indices for all spectra
    labels_idx = [species_to_idx[label] for label in labels]

    print("Preprocessing spectra...")
    preprocessed_spectra = []
    # Loop over all spectra and apply preprocessing
    for spectrum in tqdm(spectra_list):
        # Preprocess the spectrum (variance stabilization, smoothing, baseline correction, normalization)
        spectrum = preprocess_spectrum(spectrum)
        # Bin the spectrum into fixed-width m/z bins
        mz_bins, intensity_bins = bin_spectrum(spectrum)
        # Append the binned m/z values and intensities to the list
        preprocessed_spectra.append((mz_bins, intensity_bins))

    print("Saving data to HDF5...")
    with h5py.File(outfile, 'w') as h5f:
        # Create datasets in the HDF5 file for m/z values, intensities, labels, locations, and species labels

        # Save the binned m/z values for each spectrum
        h5f.create_dataset('mz', data=[mz for mz, _ in preprocessed_spectra])
        # Save the binned intensity values for each spectrum
        h5f.create_dataset('intensity', data=[intensity for _, intensity in preprocessed_spectra])
        # Save the integer labels corresponding to species for each spectrum
        h5f.create_dataset('labels', data=labels_idx)
        # Define the variable-length string data type for h5py
        dt = h5py.string_dtype(encoding='utf-8')
        # Save the file paths (locations) of each spectrum
        h5f.create_dataset('locs', data=locs, dtype=dt)
        # Save the list of unique species labels
        h5f.create_dataset('species_labels', data=species_labels, dtype=dt)
    print(f"Data saved to {outfile}")

def main():
    # Entry point of the script

    # Get command-line arguments
    RKI_root = sys.argv[1]       # The root directory containing the RKI dataset
    output_file = sys.argv[2]    # The output HDF5 file to save the preprocessed data

    # Call the function to process raw data and save it to the HDF5 file
    RKI_raw_to_h5(RKI_root, output_file)

if __name__ == '__main__':
    main()
