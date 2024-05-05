import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image



def train_autoencoder(model, train_loader, criterion, optimizer, device, epoch = num_epochs):
    """Function to train the autoencoder model."""
    model.train()  # Set model to training mode
    total_loss = 0.0
    for i, (images, _) in enumerate(train_loader):
        images = images.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, images)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if (i+1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

    return total_loss / len(train_loader)

def evaluate_autoencoder(model, test_loader, criterion, device):
    """Function to evaluate the autoencoder model."""
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    with torch.no_grad():
        for images, _ in test_loader:
            images = images.to(device)
            outputs = model(images)
            loss = criterion(outputs, images)
            total_loss += loss.item() * images.size(0)

    return total_loss / len(test_loader.dataset)

def main():
    # Define training parameters
    num_epochs = '...'  # Number of epochs
    batch_size = '....' # Batch size
    learning_rate = '....'  # Learning rate

    # Check if GPU is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create the autoencoder model
    model = model().to(device)   #Call Model you want to train

    # Define the loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Define data transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])

    # Load the dataset
    train_dataset = ImageFolder(root="path/to/train-data", transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = ImageFolder(root="path/to/validation/data", transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch [{epoch+1}/{num_epochs}]")  # Epoch number

        # Training
        train_loss = train_autoencoder(model, train_loader, criterion, optimizer, device)
        print(f"Training Loss: {train_loss:.4f}")  # Training loss

        # Save reconstructed images
        with torch.no_grad():
            sample_images = next(iter(train_loader))[0][:8].to(device)
            reconstructed = model(sample_images)
            sample_images = torch.cat([sample_images, reconstructed])

        # Evaluate the model
        test_loss = evaluate_autoencoder(model, test_loader, criterion, device)
        print(f"Average Loss on Test Set: {test_loss:.4f}")  # Test loss

        # Save the model
        torch.save(model.state_dict(), f"model{epoch+1}.pth")

    print("Training completed!")

if __name__ == "__main__":
    main()
