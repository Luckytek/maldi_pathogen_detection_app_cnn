import torch
import torch.nn as nn
import torch.optim as optim
from data_loader_cnn import get_data_loaders
from model_cnn import MALDIResNet
from losses_cnn import FocalLoss  # Import FocalLoss
from sklearn.metrics import accuracy_score, classification_report
from torch.utils.tensorboard import SummaryWriter
import numpy as np

def train_model(
    h5_file_path,
    model_save_path='cnn_model.pth',
    num_epochs=100,  # Increased epochs
    batch_size=32,
    learning_rate=0.0005,  # Slightly increased learning rate
    validation_split=0.2
):
    # Initialize TensorBoard writer
    writer = SummaryWriter()

    # Get data loaders, number of classes, and class weights
    train_loader, val_loader, num_classes, class_weights = get_data_loaders(
        h5_file_path, batch_size, validation_split
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize the model
    model = MALDIResNet(num_classes=num_classes).to(device)

    # Convert class weights to PyTorch tensor and move to device
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

    # Use Focal Loss with class weights
    criterion = FocalLoss(alpha=class_weights, gamma=2.0)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Early stopping parameters
    patience = 10
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        y_true_train = []
        y_pred_train = []

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            # Collect predictions for accuracy calculation
            _, preds = torch.max(outputs, 1)
            y_true_train.extend(labels.cpu().numpy())
            y_pred_train.extend(preds.cpu().numpy())

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = accuracy_score(y_true_train, y_pred_train)

        # Validation
        model.eval()
        running_val_loss = 0.0
        y_true_val = []
        y_pred_val = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)

                running_val_loss += loss.item() * inputs.size(0)

                # Collect predictions for accuracy calculation
                _, preds = torch.max(outputs, 1)
                y_true_val.extend(labels.cpu().numpy())
                y_pred_val.extend(preds.cpu().numpy())

        val_loss = running_val_loss / len(val_loader.dataset)
        val_acc = accuracy_score(y_true_val, y_pred_val)

        # Adjust learning rate
        scheduler.step(val_loss)

        # Logging to TensorBoard
        writer.add_scalar('Loss/train', epoch_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', epoch_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)

        print(f'Epoch [{epoch+1}/{num_epochs}] '
              f'Train Loss: {epoch_loss:.4f} Train Acc: {epoch_acc:.4f} '
              f'Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}')

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            # Save the best model
            torch.save(model.state_dict(), model_save_path)
        else:
            epochs_no_improve += 1

        if epochs_no_improve == patience:
            print("Early stopping!")
            break

    # Close TensorBoard writer
    writer.close()

    print(f'Best model saved to {model_save_path}')

    # Print classification report on validation data
    print('Classification Report on Validation Data:')
    print(classification_report(y_true_val, y_pred_val, zero_division=0))

if __name__ == '__main__':
    # Example usage
    h5_file_path = 'preprocessed_data.h5'  # Replace with your actual file path
    train_model(h5_file_path)
