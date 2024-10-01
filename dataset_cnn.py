import h5py
import torch
from torch.utils.data import Dataset
import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage import grey_opening

class MALDIDataset(Dataset):
    def __init__(self, h5_file_path, transform=None):
        """
        Custom Dataset for MALDI spectra with enhanced preprocessing.

        Parameters:
        - h5_file_path: str
            Path to the HDF5 file containing the preprocessed data.
        - transform: callable, optional
            Optional transform to be applied on a sample.
        """
        self.h5_file_path = h5_file_path
        self.transform = transform

        # Load data from the HDF5 file
        with h5py.File(self.h5_file_path, 'r') as h5f:
            self.intensity_data = h5f['intensity'][:]  # Shape: (num_samples, num_bins)
            self.labels = h5f['labels'][:]
            self.species_labels = h5f['species_labels'][:]

        # Decode species labels if they are stored as bytes
        if isinstance(self.species_labels[0], bytes):
            self.species_labels = [label.decode('utf-8') for label in self.species_labels]

        # Map label indices to species names
        self.label_to_species = {i: self.species_labels[i] for i in range(len(self.species_labels))}

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        intensity = self.intensity_data[idx]
        label = self.labels[idx]

        # Apply baseline correction
        intensity = self.baseline_correction(intensity)

        # Apply smoothing with Savitzky-Golay filter
        intensity = savgol_filter(intensity, window_length=11, polyorder=3)

        # Normalize intensity data
        intensity = (intensity - np.mean(intensity)) / (np.std(intensity) + 1e-8)

        # Convert to PyTorch tensors
        intensity = torch.tensor(intensity, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        # Apply transforms if any
        if self.transform:
            intensity = self.transform(intensity, label)

        # Add a channel dimension for CNN input (batch_size, channels, sequence_length)
        intensity = intensity.unsqueeze(0)  # Shape: (1, sequence_length)

        return intensity, label

    def baseline_correction(self, spectrum):
        """
        Apply baseline correction to the spectrum using morphological opening.

        Parameters:
        - spectrum: ndarray
            The input spectrum.

        Returns:
        - corrected_spectrum: ndarray
            The baseline-corrected spectrum.
        """
        # Apply morphological opening to estimate the baseline
        size = max(3, int(len(spectrum) * 0.01))  # Ensure size is at least 3
        baseline = grey_opening(spectrum, size=size)
        corrected_spectrum = spectrum - baseline
        return corrected_spectrum
