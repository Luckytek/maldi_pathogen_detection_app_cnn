import streamlit as st
import h5py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from model_cnn import MALDIResNet
from scipy.signal import savgol_filter
from scipy.ndimage import grey_opening
import os
import gdown  # Import gdown for downloading from Google Drive

# Function to download the data file from Google Drive
def download_data():
    h5_file_path = 'preprocessed_data.h5'
    if not os.path.exists(h5_file_path):
        url = 'https://drive.google.com/uc?id=1c4l6_rZXRhmiwwnO2sqwys2HLU6tJTT8'  # Replace with your actual file ID
        st.write("Downloading preprocessed data from Google Drive...")
        gdown.download(url, h5_file_path, quiet=False)
        st.write("Data file downloaded successfully.")
    else:
        st.write("Data file already exists locally.")

# Call the download function at the beginning of the script
download_data()

# Function for baseline correction
def baseline_correction(spectrum):
    size = max(3, int(len(spectrum) * 0.01))  # Ensure size is at least 3
    baseline = grey_opening(spectrum, size=size)
    corrected_spectrum = spectrum - baseline
    return corrected_spectrum

# Function to preprocess the spectrum
def preprocess_spectrum(intensity):
    # Apply baseline correction
    intensity = baseline_correction(intensity)

    # Apply smoothing with Savitzky-Golay filter
    intensity = savgol_filter(intensity, window_length=11, polyorder=3)

    # Normalize intensity data
    intensity = (intensity - np.mean(intensity)) / (np.std(intensity) + 1e-8)

    # Convert to PyTorch tensor and add batch and channel dimensions
    intensity_tensor = torch.tensor(intensity, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, sequence_length)

    return intensity_tensor

# Modify load_data to load from local file
@st.cache_data
def load_data():
    h5_file_path = 'preprocessed_data.h5'
    # Load data
    with h5py.File(h5_file_path, 'r') as h5f:
        intensity_data = h5f['intensity'][:]  # Shape: (num_samples, num_bins)
        labels = h5f['labels'][:]
        species_labels = h5f['species_labels'][:]
        if 'mz_bins' in h5f:
            mz_bins = h5f['mz_bins'][:]  # Shape: (num_bins,)
        else:
            mz_bins = None

    # Decode species labels if they are stored as bytes
    if isinstance(species_labels[0], bytes):
        species_labels = [label.decode('utf-8') for label in species_labels]

    # Map label indices to species names
    label_to_species = {i: species_labels[i] for i in range(len(species_labels))}

    return intensity_data, labels, label_to_species, mz_bins

@st.cache_resource
def load_model(model_path, num_classes, device):
    model = MALDIResNet(num_classes=num_classes).to(device)
    # Load the state dictionary with weights_only=True (PyTorch 2.0 or later)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    return model

def main():
    st.title("MALDI CNN Model Web App")
    st.write("This web application allows you to test the CNN model on MALDI spectra.")

    # Sidebar inputs
    st.sidebar.header("User Inputs")

    # Model file path
    model_file_default = 'cnn_model.pth'
    model_path = st.sidebar.text_input("Enter the path to the model file:", model_file_default)

    # Check if model file exists
    if not os.path.exists(model_path):
        st.error(f"Model file not found at {model_path}")
        return

    # Load data
    st.write("Loading preprocessed data...")
    intensity_data, labels, label_to_species, mz_bins = load_data()
    st.write("Data loaded successfully.")

    num_spectra = len(labels)
    num_classes = len(label_to_species)
    st.write(f"Total number of spectra: {num_spectra}")

    # Select a spectrum index
    spectrum_index = st.sidebar.number_input(
        f"Select the index of the spectrum to analyze (0 to {num_spectra - 1}):",
        min_value=0,
        max_value=num_spectra - 1,
        value=0,
        step=1
    )

    # Load the selected spectrum and label
    intensity = intensity_data[spectrum_index]
    true_label = labels[spectrum_index]
    true_species = label_to_species[true_label]

    st.write(f"**Spectrum Index:** {spectrum_index}")
    st.write(f"**True Species:** {true_species}")

    # Plot the spectrum
    st.write("### Spectrum Plot")
    fig, ax = plt.subplots(figsize=(10, 6))
    if mz_bins is not None:
        ax.plot(mz_bins, intensity)
        ax.set_xlabel('m/z')
    else:
        ax.plot(intensity)
        ax.set_xlabel('Bin Index')
    ax.set_ylabel('Intensity')
    ax.set_title(f'Spectrum Index: {spectrum_index}, True Species: {true_species}')
    st.pyplot(fig)

    # Preprocess the spectrum
    st.write("Preprocessing the spectrum...")
    intensity_tensor = preprocess_spectrum(intensity)

    # Device selection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    st.write(f"Using device: {device}")

    # Load the model
    st.write("Loading the model...")
    try:
        model = load_model(model_path, num_classes, device)
        st.write("Model loaded successfully.")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return

    # Move data to the device
    intensity_tensor = intensity_tensor.to(device)

    # Make the prediction
    st.write("Making the prediction...")
    with torch.no_grad():
        outputs = model(intensity_tensor)  # Shape: (1, num_classes)
        _, predicted_label = torch.max(outputs, dim=1)
        predicted_label = predicted_label.item()

    predicted_species = label_to_species.get(predicted_label, "Unknown")

    # Display the predicted species
    st.write(f"**Predicted Species:** {predicted_species}")

    # Determine if the prediction is correct
    if predicted_species == true_species:
        st.success("Prediction is correct!")
    else:
        st.error("Prediction is incorrect.")

if __name__ == '__main__':
    main()
