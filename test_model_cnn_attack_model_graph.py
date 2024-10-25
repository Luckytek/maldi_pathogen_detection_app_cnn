# test_model_cnn.py

import h5py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from model_cnn import MALDIResNet
import argparse
from scipy.signal import savgol_filter
from scipy.ndimage import grey_opening

# Import or define FocalLoss
# If losses_cnn.py is accessible, you can import FocalLoss
# from losses_cnn import FocalLoss

# Alternatively, define FocalLoss directly here
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        """
        Focal Loss for multi-class classification.

        Parameters:
        - alpha: Tensor or None
            Weights for each class. If None, all classes are treated equally.
        - gamma: float
            Focusing parameter gamma >= 0.
        - reduction: str
            Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
        """
        super(FocalLoss, self).__init__()
        if alpha is not None:
            if isinstance(alpha, (list, np.ndarray)):
                alpha = torch.tensor(alpha, dtype=torch.float32)
            elif not isinstance(alpha, torch.Tensor):
                raise TypeError('alpha must be a list, numpy array, or torch Tensor')
            self.alpha = alpha
        else:
            self.alpha = None
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Forward pass for the loss function.

        Parameters:
        - inputs: Tensor
            Predicted logits with shape (batch_size, num_classes).
        - targets: Tensor
            Ground truth labels with shape (batch_size).
        """
        # Compute softmax over the classes
        probs = F.softmax(inputs, dim=1)
        # Get the probabilities corresponding to the target classes
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        # Compute the logarithm of pt
        log_pt = torch.log(pt + 1e-8)  # Add epsilon to avoid log(0)
        # Compute the focal loss term
        focal_term = (1 - pt) ** self.gamma
        loss = -focal_term * log_pt

        # Apply class weights if provided
        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            at = self.alpha.gather(0, targets)
            loss = loss * at

        # Reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

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

def fgsm_attack(model, tensor, true_label, epsilon, device, criterion):
    """
    Generates adversarial examples using the Fast Gradient Sign Method (FGSM).

    Parameters:
    - model: PyTorch model
        The trained model.
    - tensor: Tensor
        The input tensor.
    - true_label: int
        The true label of the input.
    - epsilon: float
        The perturbation magnitude.
    - device: torch.device
        The device to perform computations on.
    - criterion: Loss function
        The loss function used for computing gradients.

    Returns:
    - perturbed_tensor: Tensor
        The adversarial example.
    """
    # Set requires_grad attribute of tensor
    tensor.requires_grad = True

    # Forward pass the data through the model
    outputs = model(tensor)

    # Calculate the loss using FocalLoss
    target = torch.tensor([true_label], dtype=torch.long).to(device)
    loss = criterion(outputs, target)

    # Zero all existing gradients
    model.zero_grad()

    # Perform backward pass
    loss.backward()

    # Collect the data gradient
    data_grad = tensor.grad.data

    # Create the perturbed tensor by adjusting each element
    perturbed_tensor = tensor + epsilon * data_grad.sign()

    # Clamp the perturbed tensor to maintain original data range
    perturbed_tensor = torch.clamp(perturbed_tensor, tensor.min(), tensor.max())

    return perturbed_tensor.detach()

def get_top_predictions(probabilities, label_to_species, true_species, top_k=5):
    """
    Get the top K predictions and their probabilities.

    Parameters:
    - probabilities: Tensor
        The probabilities output by the model after softmax.
    - label_to_species: dict
        Mapping from label indices to species names.
    - true_species: str
        The true species name.
    - top_k: int
        Number of top predictions to retrieve.

    Returns:
    - species_names: list
        List of species names in the top K predictions.
    - prediction_probs: list
        List of probabilities corresponding to the species.
    - colors: list
        List of colors for the bars in the plot.
    """
    topk_prob, topk_indices = torch.topk(probabilities, k=top_k, dim=1)
    # Remove the batch dimension
    topk_prob = topk_prob.squeeze(0)
    topk_indices = topk_indices.squeeze(0)

    species_names = []
    prediction_probs = []
    colors = []

    for i in range(top_k):
        predicted_label_topk = topk_indices[i].item()
        predicted_species_topk = label_to_species.get(predicted_label_topk, "Unknown")
        probability = topk_prob[i].item()
        species_names.append(predicted_species_topk)
        prediction_probs.append(probability)
        if predicted_species_topk == true_species:
            colors.append('green')  # Highlight the true species in green
        else:
            colors.append('skyblue')

    # If true species is not in the top predictions, add it
    if true_species not in species_names:
        true_label = None
        for label, species in label_to_species.items():
            if species == true_species:
                true_label = label
                break
        if true_label is not None:
            true_species_probability = probabilities[0, true_label].item()
            species_names.append(true_species)
            prediction_probs.append(true_species_probability)
            colors.append('red')  # Highlight the true species in red

    return species_names, prediction_probs, colors

def plot_predictions(species_names, prediction_probs, colors, title):
    """
    Plot a bar chart of predicted species and their probabilities.

    Parameters:
    - species_names: list
        List of species names.
    - prediction_probs: list
        List of probabilities corresponding to the species.
    - colors: list
        List of colors for the bars in the plot.
    - title: str
        Title of the plot.
    """
    from matplotlib.patches import Patch

    plt.figure(figsize=(10, 6))
    bars = plt.bar(species_names, prediction_probs, color=colors)
    plt.xlabel('Species')
    plt.ylabel('Prediction Probability')
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.ylim([0, 1])

    # Annotate the bars with probability values
    for bar, prob in zip(bars, prediction_probs):
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.01, f"{prob:.4f}", ha='center', va='bottom')

    # Add legend for colors
    legend_elements = [
        Patch(facecolor='skyblue', label='Predicted Species'),
        Patch(facecolor='green', label='True Species (in Top Predictions)'),
        Patch(facecolor='red', label='True Species (Not in Top Predictions)'),
    ]
    plt.legend(handles=legend_elements)

    plt.tight_layout()
    plt.show()

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
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("Model loaded successfully.")

    # Compute class weights based on the labels
    labels_array = np.array(labels)
    class_counts = np.bincount(labels_array)
    total_samples = len(labels_array)
    class_weights = total_samples / (len(class_counts) * class_counts)
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    # Initialize FocalLoss with class_weights and gamma
    criterion = FocalLoss(alpha=class_weights, gamma=2.0)

    # Move data to the device
    intensity_tensor = intensity_tensor.to(device)

    print("Making the prediction on the original spectrum...")
    with torch.no_grad():
        outputs = model(intensity_tensor)  # Shape: (1, num_classes)
        probabilities = torch.softmax(outputs, dim=1)  # Apply softmax to get probabilities
        max_prob, predicted_label = torch.max(probabilities, dim=1)
        predicted_label = predicted_label.item()
        max_prob = max_prob.item()

    predicted_species = label_to_species.get(predicted_label, "Unknown")

    # Display the predicted species
    print(f"Predicted Species Label: {predicted_label}")
    print(f"Predicted Species: {predicted_species}")
    print(f"Prediction Probability: {max_prob:.4f}")
    print(f"True Species Label: {true_label}")
    print(f"True Species: {true_species}")

    if predicted_species == true_species:
        print("Prediction is correct!")
    else:
        print("Prediction is incorrect.")

    # Get top predictions and plot for the original spectrum
    species_names, prediction_probs, colors = get_top_predictions(probabilities, label_to_species, true_species)
    plot_predictions(species_names, prediction_probs, colors, title='Original Spectrum Predictions')

    # Generate adversarial examples with different epsilon values
    epsilons = [0.01, 0.05, 0.1]  # You can adjust these values
    # epsilons = [0.01, 0.05, 0.2]  # You can adjust these values
    adversarial_tensors = []
    adversarial_intensities = []
    adversarial_predictions = []

    for epsilon in epsilons:
        print(f"\nGenerating adversarial example with epsilon = {epsilon}...")
        adv_tensor = fgsm_attack(model, intensity_tensor.clone(), true_label, epsilon, device, criterion)
        adversarial_tensors.append(adv_tensor)

        # Convert tensor back to numpy array for plotting
        adv_intensity = adv_tensor.squeeze().cpu().numpy()
        adversarial_intensities.append(adv_intensity)

        # Make prediction on adversarial example
        with torch.no_grad():
            outputs_adv = model(adv_tensor)
            probabilities_adv = torch.softmax(outputs_adv, dim=1)
            max_prob_adv, predicted_label_adv = torch.max(probabilities_adv, dim=1)
            predicted_label_adv = predicted_label_adv.item()
            max_prob_adv = max_prob_adv.item()

        predicted_species_adv = label_to_species.get(predicted_label_adv, "Unknown")

        print(f"Adversarial Predicted Species Label: {predicted_label_adv}")
        print(f"Adversarial Predicted Species: {predicted_species_adv}")
        print(f"Adversarial Prediction Probability: {max_prob_adv:.4f}")

        if predicted_species_adv == true_species:
            print("Adversarial prediction is correct!")
        else:
            print("Adversarial prediction is incorrect!")

        # Store predictions
        adversarial_predictions.append({
            'epsilon': epsilon,
            'adv_tensor': adv_tensor,
            'probabilities': probabilities_adv,
            'predicted_label': predicted_label_adv,
            'predicted_species': predicted_species_adv,
            'probability': max_prob_adv,
        })

    # Plot the original and adversarial spectra
    plt.figure(figsize=(12, 8))
    num_plots = 1 + len(epsilons)
    plt.subplot(num_plots, 1, 1)
    if mz_bins is not None:
        plt.plot(mz_bins, intensity, label='Original Spectrum')
        plt.xlabel('m/z')
    else:
        plt.plot(intensity, label='Original Spectrum')
        plt.xlabel('Bin Index')
    plt.ylabel('Intensity')
    plt.title(f'Original Spectrum - True Species: {true_species}')
    plt.legend()

    for i, adv_intensity in enumerate(adversarial_intensities):
        plt.subplot(num_plots, 1, i+2)
        if mz_bins is not None:
            plt.plot(mz_bins, adv_intensity, label=f'Adversarial Spectrum (epsilon={epsilons[i]})')
            plt.xlabel('m/z')
        else:
            plt.plot(adv_intensity, label=f'Adversarial Spectrum (epsilon={epsilons[i]})')
            plt.xlabel('Bin Index')
        plt.ylabel('Intensity')
        plt.title(f'Adversarial Spectrum - Epsilon: {epsilons[i]}')
        plt.legend()

    plt.tight_layout()
    plt.show()

    # Display predictions for all spectra
    print("\nSummary of Predictions:")
    print("Original Spectrum:")
    print(f"  Predicted Species: {predicted_species}")
    print(f"  Prediction Probability: {max_prob:.4f}")
    if predicted_species == true_species:
        print("  Prediction is correct!")
    else:
        print("  Prediction is incorrect!")

    for adv_pred in adversarial_predictions:
        print(f"\nAdversarial Spectrum (epsilon = {adv_pred['epsilon']}):")
        print(f"  Predicted Species: {adv_pred['predicted_species']}")
        print(f"  Prediction Probability: {adv_pred['probability']:.4f}")
        if adv_pred['predicted_species'] == true_species:
            print("  Adversarial prediction is correct!")
        else:
            print("  Adversarial prediction is incorrect!")

        # Get top predictions and plot for the adversarial spectrum
        species_names_adv, prediction_probs_adv, colors_adv = get_top_predictions(
            adv_pred['probabilities'], label_to_species, true_species
        )
        plot_title = f'Adversarial Spectrum Predictions (epsilon = {adv_pred["epsilon"]})'
        plot_predictions(species_names_adv, prediction_probs_adv, colors_adv, title=plot_title)

if __name__ == '__main__':
    main()
