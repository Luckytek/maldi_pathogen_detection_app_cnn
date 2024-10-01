import torch
from torch.utils.data import DataLoader, Subset
from dataset_cnn import MALDIDataset
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data.sampler import WeightedRandomSampler

# Define the transform class at the module level
class AugmentForMinorityClasses:
    def __init__(self, minority_classes):
        self.minority_classes = set(minority_classes.tolist())

    def __call__(self, intensity, label):
        if label.item() in self.minority_classes:
            # Apply data augmentation (e.g., adding noise)
            noise = torch.randn_like(intensity) * 0.05  # Adjust noise level as needed
            intensity = intensity + noise
        return intensity

def get_data_loaders(h5_file_path, batch_size=32, validation_split=0.2, shuffle=True):
    """
    Create data loaders for training and validation datasets with stratified splitting,
    class weights calculation, and selective data augmentation for minority classes.

    Parameters:
    - h5_file_path: str
        Path to the HDF5 file containing the preprocessed data.
    - batch_size: int
        Batch size for data loaders.
    - validation_split: float
        Fraction of the dataset to use for validation.
    - shuffle: bool
        Whether to shuffle the dataset before splitting.

    Returns:
    - train_loader: DataLoader
        DataLoader for the training set.
    - val_loader: DataLoader
        DataLoader for the validation set.
    - num_classes: int
        Number of classes in the dataset.
    - class_weights: ndarray
        Array of class weights for the loss function.
    """
    # Create the dataset
    dataset = MALDIDataset(h5_file_path)

    # Get labels for stratification
    labels = dataset.labels

    # Compute class weights using scikit-learn
    classes = np.unique(labels)
    class_weights = compute_class_weight('balanced', classes=classes, y=labels)
    # class_weights remains a NumPy array here

    # Stratified splitting
    strat_split = StratifiedShuffleSplit(
        n_splits=1, test_size=validation_split, random_state=42
    )

    for train_idx, val_idx in strat_split.split(np.zeros(len(labels)), labels):
        train_indices = train_idx
        val_indices = val_idx

    # Create separate dataset instances for training and validation
    train_dataset_full = MALDIDataset(h5_file_path)
    val_dataset_full = MALDIDataset(h5_file_path)

    # Identify minority classes in the training set
    train_labels = labels[train_indices]
    class_counts = np.bincount(train_labels)
    minority_classes = np.where(class_counts < np.percentile(class_counts, 25))[0]  # Bottom 25%

    # Apply transform only to the training dataset
    transform = AugmentForMinorityClasses(minority_classes)
    train_dataset_full.transform = transform  # Only the training dataset has the transform

    # Create subsets
    train_dataset = Subset(train_dataset_full, train_indices)
    val_dataset = Subset(val_dataset_full, val_indices)

    # Compute sample weights for the WeightedRandomSampler
    # Use inverse of class counts to give higher weight to minority classes
    class_sample_counts = np.array([class_counts[t] for t in train_labels])
    sample_weights = 1.0 / class_sample_counts
    sample_weights = sample_weights / sample_weights.sum()  # Normalize

    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Get the number of classes
    num_classes = len(classes)

    return train_loader, val_loader, num_classes, class_weights
