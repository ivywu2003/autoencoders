import random
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from dae import ConvDenoiser
print(torch.__version__)
print(np.__version__)



# Function to compute saliency in the latent space
def compute_latent_saliency(dae_model, input_image, original_image, label):
    noisy_img = input_image
    input_image.requires_grad_()  # Ensure input image supports gradient computation

    # Pass the image through the encoder to get the latent representation
    latent = dae_model.encoder(input_image)
    res = dae_model.decoder(latent)
    # Compute gradients with respect to the latent representation
    latent.backward(torch.ones_like(latent))

    # Get gradients w.r.t. the input image
    saliency_map = input_image.grad.data.abs().squeeze().cpu().numpy()

    saliency_map_min = saliency_map.min()
    saliency_map_max = saliency_map.max()
    saliency_map_normalized = (saliency_map - saliency_map_min) / (saliency_map_max - saliency_map_min)

    fig, ax = plt.subplots(1, 4, figsize=(12, 5))
    ax[0].imshow(original_image.detach().numpy().squeeze().squeeze(), cmap='gray')
    ax[0].set_title('Original Image')
    ax[0].axis('off')
    
    ax[1].imshow(noisy_img.detach().numpy().squeeze(), cmap="gray")
    ax[1].set_title("Noisy Image")
    ax[1].axis("off")
    
    ax[2].imshow(res.detach().numpy().squeeze(), cmap="gray")
    ax[2].set_title("Reconstruction")
    ax[2].axis("off")

    saliency_heatmap = ax[3].imshow(saliency_map_normalized, cmap='hot')
    ax[3].set_title('Saliency in Latent Space')
    ax[3].axis('off')
    
    cbar = fig.colorbar(saliency_heatmap, ax=ax[3], orientation="vertical", fraction=0.046, pad=0.04)
    cbar.set_label("Saliency Intensity", rotation=270, labelpad=15)
    plt.savefig(f"./final graphics/dae_map_{label}.png")
    plt.show()

from matplotlib import pyplot as plt
from dae import ConvDenoiser
import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms


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

# obtain one batch of test images
dae_model = ConvDenoiser()
dae_model.load_state_dict(torch.load('dae_weights.pth'))
dae_model.eval()
dataiter = iter(test_loader)
torch.random.manual_seed(0)
# random.seed(0)
images, labels = next(dataiter)
noise_factor=0.5

# add noise to the test images
noisy_imgs = images + noise_factor * torch.randn(*images.shape)
noisy_imgs = np.clip(noisy_imgs, 0., 1.)

# get sample outputs
output, latent = dae_model(noisy_imgs)
# prep images for display
noisy_imgs = noisy_imgs.numpy()

# output is resized into a batch of iages
output = output.view(batch_size, 1, 28, 28)
# use detach when it's an output that requires_grad
output = output.detach().numpy()

# plot the first ten input images and then reconstructed images
num_images = 5
fig, axes = plt.subplots(3, num_images, figsize=(15, 7))

for i in range(num_images):
    # Original Image
    axes[0, i].imshow(images[i, 0], cmap="gray")
    axes[0, i].set_title("Original")
    axes[0, i].axis("off")
    # Masked Image: overlay mask on original image
    axes[1, i].imshow(noisy_imgs[i, 0], cmap="gray")
    # axes[1, i].imshow(mask_overlay[i, 0], cmap="gray", alpha=0.5)  # Mask overlay with transparency
    axes[1, i].set_title("Input")
    axes[1, i].axis("off")
    # Reconstructed Image
    axes[2, i].imshow(output[i, 0], cmap="gray")
    axes[2, i].set_title("Reconstructed")
    axes[2, i].axis("off")
plt.show()


noisy_imgs = images + noise_factor * torch.randn(*images.shape)
noisy_imgs = np.clip(noisy_imgs, 0., 1.)
# Clip the images to be between 0 and 1
num_images = 5 
for i in range(num_images):  
    compute_latent_saliency(dae_model, noisy_imgs[i], images[i], labels[i])
    