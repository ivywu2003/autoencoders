from cifar10_models import CIFAR10DenoisingAutoencoder
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import math
from scipy.ndimage import gaussian_filter

# Function to compute saliency in the latent space
def compute_latent_saliency(dae_model, input_image, original_image, label, index, fig, ax):
    # Enable gradients for input
    input_image = input_image.clone()
    input_image.requires_grad_(True)
    
    # Get latent representations for both noisy and clean images
    noisy_latent = dae_model.encoder(input_image)
    with torch.no_grad():
        clean_latent = dae_model.encoder(original_image)
    
    # Compute L2 distance between clean and noisy latents at each spatial location
    latent_diff = torch.norm(noisy_latent - clean_latent, p=2, dim=1)  # Norm across channels
    latent_diff = latent_diff.mean()  # Mean across spatial locations
    
    # Backward pass
    latent_diff.backward()
    
    # Get saliency map from gradients
    saliency_map = input_image.grad.abs()
    saliency_map = saliency_map.squeeze().permute(1, 2, 0).detach().cpu().numpy()
    
    # Apply Gaussian smoothing to reduce noise
    saliency_map = gaussian_filter(saliency_map, sigma=1.0)
    
    # Normalize the saliency map
    saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-8)
    
    # Display original image
    ax[index, 0].imshow(original_image.permute(1, 2, 0).detach().cpu().numpy())
    ax[index, 0].set_title('Original Image')
    ax[index, 0].axis('off')
    
    # Display noisy image
    ax[index, 1].imshow(input_image.squeeze().permute(1, 2, 0).detach().cpu().numpy())
    ax[index, 1].set_title("Noisy Image")
    ax[index, 1].axis("off")
    
    # Display latent differences
    latent_diff_vis = (noisy_latent - clean_latent).squeeze().mean(dim=0).detach().cpu().numpy()
    latent_diff_vis = (latent_diff_vis - latent_diff_vis.min()) / (latent_diff_vis.max() - latent_diff_vis.min() + 1e-8)
    ax[index, 2].imshow(latent_diff_vis, cmap='coolwarm')
    ax[index, 2].set_title("Latent Differences")
    ax[index, 2].axis("off")
    
    # Display original image with saliency overlay
    ax[index, 3].imshow(original_image.permute(1, 2, 0).detach().cpu().numpy())
    saliency_overlay = saliency_map.mean(axis=2)
    saliency_heatmap = ax[index, 3].imshow(saliency_overlay,
                                          cmap='hot',
                                          alpha=0.5)
    ax[index, 3].set_title('Denoising Saliency')
    ax[index, 3].axis('off')
    
    # Add colorbar
    cbar = fig.colorbar(saliency_heatmap, ax=ax[index, 3], orientation="vertical", fraction=0.046, pad=0.04)
    cbar.set_label("Impact on Denoising", rotation=270, labelpad=15)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dae_model = CIFAR10DenoisingAutoencoder().to(device)
    dae_model.load_state_dict(torch.load("cifar10_dae_weights_20_epochs.pth", map_location=torch.device('cpu')))
    dae_model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=8, shuffle=True)

    dataiter = iter(testloader)
    images, labels = next(dataiter)
    noise_factor=0.1

    noisy_imgs = images + noise_factor * torch.randn(*images.shape)
    noisy_imgs = np.clip(noisy_imgs, 0., 1.)

    fig, ax = plt.subplots(8, 4, figsize=(12, 24))

    for i in range(8):
        compute_latent_saliency(dae_model, noisy_imgs[i], images[i], labels[i], i, fig, ax)

    plt.savefig(f"cifar10_dae_saliency_maps.png")
    plt.show()
