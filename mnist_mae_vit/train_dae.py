import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from dae import ConvDenoiser

def train_one_epoch(model, dataloader, optimizer, criterion, epoch, device, mask_ratio=0.75):
    model.train()
    train_loss = 0
    for data in dataloader:
        images, _ = data

        ## add random noise to the input images
        noisy_imgs = images + noise_factor * torch.randn(*images.shape)
        # Clip the images to be between 0 and 1
        noisy_imgs = np.clip(noisy_imgs, 0., 1.)

        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        ## forward pass: compute predicted outputs by passing *noisy* images to the model
        outputs, latent = dae_model(noisy_imgs)
        # calculate the loss
        # the "target" is still the original, not-noisy images
        loss = criterion(outputs, images)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update running training loss
        train_loss += loss.item()*images.size(0)
    
    avg_loss = train_loss / len(dataloader)
    print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")
    return avg_loss



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# convert data to torch.FloatTensor
transform = transforms.ToTensor()

# load the training and test datasets
train_data = datasets.MNIST(root='~/.pytorch/MNIST_data/', train=True,
                                   download=True, transform=transform)
test_data = datasets.MNIST(root='~/.pytorch/MNIST_data/', train=False,
                                  download=True, transform=transform)

# Create training and test dataloaders
num_workers = 0
# how many samples per batch to load
batch_size = 20

# prepare data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, num_workers=num_workers)

dae_model = ConvDenoiser()

# specify loss function
criterion = nn.MSELoss()

# specify loss function
optimizer = torch.optim.Adam(dae_model.parameters(), lr=0.001)

# number of epochs to train the model
n_epochs = 20

# for adding noise to images
noise_factor=0.5

for epoch in range(1, n_epochs+1):
    # monitor training loss
    train_one_epoch(dae_model, train_loader, optimizer, criterion, epoch, device)
    with torch.no_grad():
        test_loss = 0.0
        for images, _ in test_loader:
            images = images.to(device)
            outputs, _ = dae_model(images)
            loss = criterion(outputs, images)
            test_loss += loss.item()*images.size(0)
    
    print(f'Test loss: {test_loss/len(test_loader):.4f}')
torch.save(dae_model.state_dict(), 'dae_weights_20_epochs.pth')  # Save DAE weights