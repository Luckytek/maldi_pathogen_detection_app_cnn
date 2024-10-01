import h5py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from model_cnn import MALDIResNet
import argparse
from scipy.signal import savgol_filter
from scipy.ndimage import grey_opening

def baseline_correction(spectrum):
    """
    Apply baseline correction to the spectrum using morphological opening.

    Parameters:
    - spectrum: ndarray
        The input spectrum.

    Returns:
    - corrected_spectrum: ndarray
        The baseline-corrected spectrum.
    """
    size = max(3, int(len(spectrum) * 0.01))  # Ensure size is at least 3
    baseline = grey_opening(spectrum, size=size)
    corrected_spectrum = spectrum - baseline
    return corrected_spectrum

def preprocess_spectrum(intensity):
    """
    Preprocess the spectrum by applying baseline correction, smoothing, and normalization.

    Parameters:
    - intensity: ndarray
        The raw intensity values of the spectrum.

    Returns:
    - intensity: Tensor
        The preprocessed spectrum as a PyTorch tensor.
    """
    # Apply baseline correction
    intensity = baseline_correction(intensity)

    # Apply smoothing with Savitzky-Golay filter
    intensity = savgol_filter(intensity, window_length=11, polyorder=3)

    # Normalize intensity data
    intensity = (intensity - np.mean(intensity)) / (np.std(intensity) + 1e-8)

    # Convert to PyTorch tensor and add batch and channel dimensions
    intensity_tensor = torch.tensor(intensity, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, sequence_length)

    return intensity_tensor

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Test MALDI CNN Model')
    parser.add_argument('--h5_file', type=str, default='preprocessed_data.h5', help='Path to the HDF5 file')
    parser.add_argument('--model_path', type=str, default='cnn_model.pth', help='Path to the trained model file')
    args = parser.parse_args()

    h5_file_path = args.h5_file
    model_path = args.model_path

    print("Loading preprocessed data...")
    # Load the preprocessed data
    with h5py.File(h5_file_path, 'r') as h5f:
        intensity_data = h5f['intensity'][:]  # Shape: (num_samples, num_bins)
        labels = h5f['labels'][:]
        species_labels = h5f['species_labels'][:]
        # Load m/z bin centers if available
        if 'mz_bins' in h5f:
            mz_bins = h5f['mz_bins'][:]  # Shape: (num_bins,)
        else:
            mz_bins = None

    # Decode species labels if they are stored as bytes
    if isinstance(species_labels[0], bytes):
        species_labels = [label.decode('utf-8') for label in species_labels]

    # Map label indices to species names
    label_to_species = {i: species_labels[i] for i in range(len(species_labels))}
    print(f"Species labels mapping loaded.")

    # Display available spectra indices
    num_spectra = len(labels)
    print(f"Total number of spectra: {num_spectra}")
    print(f"Available indices: 0 to {num_spectra - 1}")

    # Prompt the user to select a spectrum index
    spectrum_index = int(input(f"Enter the index of the spectrum to analyze (0 to {num_spectra - 1}): "))
    if spectrum_index < 0 or spectrum_index >= num_spectra:
        print("Invalid index. Exiting.")
        return

    print("Loading the selected spectrum...")
    # Get the selected spectrum and label
    intensity = intensity_data[spectrum_index]
    true_label = labels[spectrum_index]
    true_species = label_to_species[true_label]

    print(f"True label for index {spectrum_index}: {true_label}")
    print(f"True species for index {spectrum_index}: {true_species}")

    print("Displaying the spectrum plot...")
    plt.figure(figsize=(10, 6))
    if mz_bins is not None:
        plt.plot(mz_bins, intensity)
        plt.xlabel('m/z')
    else:
        plt.plot(intensity)
        plt.xlabel('Bin Index')
    plt.ylabel('Intensity')
    plt.title(f'Spectrum Index: {spectrum_index}, True Species: {true_species}')
    plt.show(block=True)
    print("Spectrum plot displayed. Proceeding to preprocess the spectrum...")

    print("Starting spectrum preprocessing...")
    # Preprocess the spectrum
    intensity_tensor = preprocess_spectrum(intensity)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load the model
    num_classes = len(species_labels)
    model = MALDIResNet(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    print("Model loaded successfully.")

    # Move data to the device
    intensity_tensor = intensity_tensor.to(device)

    print("Making the prediction...")
    with torch.no_grad():
        outputs = model(intensity_tensor)  # Shape: (1, num_classes)
        _, predicted_label = torch.max(outputs, dim=1)
        predicted_label = predicted_label.item()

    predicted_species = label_to_species.get(predicted_label, "Unknown")

    # Display the predicted species
    print(f"Predicted Species Label: {predicted_label}")
    print(f"Predicted Species: {predicted_species}")
    print(f"True Species Label: {true_label}")
    print(f"True Species: {true_species}")

    if predicted_species == true_species:
        print("Prediction is correct!")
    else:
        print("Prediction is incorrect.")

if __name__ == '__main__':
    main()
