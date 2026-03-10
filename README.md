# Convolutional Autoencoder for Image Denoising

## AIM
To develop a convolutional autoencoder for image denoising application.

## Problem Statement and Dataset
The goal of this experiment is to build a Convolutional Autoencoder that can remove noise from images. Gaussian noise is artificially added to MNIST handwritten digit images and the model is trained to reconstruct the original clean image from the noisy input. The MNIST dataset consists of 60,000 training and 10,000 test grayscale images of size 28x28, each representing a digit from 0 to 9. Since the model learns to map noisy inputs back to clean outputs, this is an unsupervised image reconstruction task.

## DESIGN STEPS

### STEP 1: Load and Prepare Dataset
The MNIST dataset is downloaded and loaded using torchvision. Images are converted to tensors and batched using DataLoader with a batch size of 128. A noise function is defined to add Gaussian noise to images and clamp values between 0 and 1.

### STEP 2: Build the Convolutional Autoencoder
The encoder uses two convolutional layers with ReLU activation and MaxPooling to compress the image into a lower-dimensional representation. The decoder uses two transposed convolutional layers to reconstruct the image back to its original 28x28 size, with a Sigmoid activation at the output.

### STEP 3: Train and Evaluate the Model
The model is trained for 5 epochs using MSE Loss and Adam optimizer. Noisy images are fed as input and the loss is computed against the original clean images. After training, the model is evaluated on the test set and original, noisy, and denoised images are visualized side by side.

## PROGRAM

### Name: Ramitha Chowdary S
### Register Number: 212224240130
```python
# Autoencoder for Image Denoising using PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# Install torchsummary
!pip install torchsummary
from torchsummary import summary

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Transform and load MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])

dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(dataset, batch_size=128, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

# Add noise to images
def add_noise(inputs, noise_factor=0.5):
    noisy = inputs + noise_factor * torch.randn_like(inputs)
    return torch.clamp(noisy, 0., 1.)

# Define Convolutional Autoencoder
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=2, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Initialize model, loss function and optimizer
model = DenoisingAutoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Print model summary
summary(model, input_size=(1, 28, 28))

# Train the autoencoder
def train(model, loader, criterion, optimizer, epochs=5):
    train_losses = []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, _ in loader:
            images = images.to(device)
            noisy_images = add_noise(images).to(device)
            optimizer.zero_grad()
            outputs = model(noisy_images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(loader)
        train_losses.append(avg_loss)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.6f}")
    return train_losses

# Run training
train_losses = train(model, train_loader, criterion, optimizer, epochs=5)

# Plot training loss
print('Name: Ramitha Chowdary S')
print('Register Number: 212224240130')
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.show()

# Evaluate and visualize
def visualize_denoising(model, loader, num_images=10):
    model.eval()
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
            noisy_images = add_noise(images).to(device)
            outputs = model(noisy_images)
            break

    images = images.cpu().numpy()
    noisy_images = noisy_images.cpu().numpy()
    outputs = outputs.cpu().numpy()

    print('Name: Ramitha Chowdary S')
    print('Register Number: 212224240130')
    plt.figure(figsize=(18, 6))
    for i in range(num_images):
        ax = plt.subplot(3, num_images, i + 1)
        plt.imshow(images[i].squeeze(), cmap='gray')
        ax.set_title("Original")
        plt.axis("off")

        ax = plt.subplot(3, num_images, i + 1 + num_images)
        plt.imshow(noisy_images[i].squeeze(), cmap='gray')
        ax.set_title("Noisy")
        plt.axis("off")

        ax = plt.subplot(3, num_images, i + 1 + 2 * num_images)
        plt.imshow(outputs[i].squeeze(), cmap='gray')
        ax.set_title("Denoised")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

# Run visualization
visualize_denoising(model, test_loader)
```

## OUTPUT

### Model Summary
<img width="424" height="376" alt="image" src="https://github.com/user-attachments/assets/9de31adf-0a06-4bd2-af52-ba2cfa89187a" />

### Autoencoder Training Loss Convergence Over Epochs
<img width="500" height="413" alt="image" src="https://github.com/user-attachments/assets/f50911a0-fbe2-4b9e-ad2d-1eb073b0c666" />


### Original vs Noisy vs Reconstructed Image
<img width="813" height="393" alt="image" src="https://github.com/user-attachments/assets/dbcfc03f-38ec-4d57-907b-afc20ad1857b" />


## RESULT
The convolutional autoencoder was successfully trained on the MNIST dataset for 5 epochs to perform image denoising. The model learned to reconstruct clean digit images from noisy inputs with consistently decreasing training loss across epochs. The visualization confirms that the denoised outputs closely resemble the original images, demonstrating the model's ability to effectively suppress Gaussian noise while preserving the structure of handwritten digits.
