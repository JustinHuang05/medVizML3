from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import timm  # Make sure to install timm: pip install timm
import matplotlib.pyplot as plt
from tqdm import tqdm  # Make sure to install tqdm: pip install tqdm

# Define the model class
class MLModel(nn.Module):
    def __init__(self, num_classes=6):
        super(MLModel, self).__init__()
        # Use a pre-trained ResNet50 model
        self.base_model = timm.create_model('resnet50', pretrained=True)
        # Replace the final fully connected layer with a new one that includes Dropout for regularization
        self.base_model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.base_model.fc.in_features, num_classes)
        )
    
    def forward(self, x):
        return self.base_model(x)

# Define the training function
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=25, patience=5):
    model.to(device)
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_loader_tqdm = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', unit='batch')
        for inputs, labels in train_loader_tqdm:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            train_loader_tqdm.set_postfix(loss=running_loss / len(train_loader.dataset))
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')
        
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        val_loader_tqdm = tqdm(val_loader, desc=f'Validation {epoch+1}/{num_epochs}', unit='batch')
        with torch.no_grad():
            for inputs, labels in val_loader_tqdm:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                val_loader_tqdm.set_postfix(loss=val_loss / len(val_loader.dataset))
        val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        val_accuracy = correct / total
        print(f'Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}')

        # Check for improvement in validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            # Save the best model
            torch.save(model.state_dict(), 'resnet50_medViz3.pth')
            print('Model saved to resnet50_medViz3.pth')
        else:
            epochs_without_improvement += 1
            # Early stopping condition
            if epochs_without_improvement >= patience:
                print('Early stopping triggered')
                break

        # Step the scheduler
        scheduler.step()

    # Plot and save the loss over epochs
    plt.plot(train_losses, label='Training loss')
    plt.plot(val_losses, label='Validation loss')
    plt.legend()
    plt.title("Loss over epochs")
    output_file = 'loss_over_epochs.png'
    plt.savefig(output_file)
    plt.show()

def main():
    # Define transforms with augmentation for training data
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Transforms for validation data
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Path to the dataset
    dataPath = '/Users/justinhuang/Documents/Developer/ML/medVizML3/medVizData_split'
    base_dir = Path(dataPath)
    train_dir = base_dir / 'train'
    valid_dir = base_dir / 'valid'

    # Load datasets
    train_dataset = ImageFolder(root=train_dir, transform=train_transform)
    val_dataset = ImageFolder(root=valid_dir, transform=val_transform)

    # Define data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # Initialize model, loss function, and optimizer
    model = MLModel(num_classes=6)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Train the model with early stopping and plot the losses
    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=25, patience=5)

if __name__ == "__main__":
    main()
