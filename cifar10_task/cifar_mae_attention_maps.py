from cifar10_models import CIFAR10Autoencoder, CustomCIFAR10MaskedAutoencoder


import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import torchvision

def patchify(img, patch_size):
    """
    Converts an image tensor into patches.
    Args:
        img: (N, C, H, W) tensor.
        patch_size: Patch size (int).
    Returns:
        Patches: (N, L, patch_size**2 * C).
    """
    assert img.shape[2] % patch_size == 0 and img.shape[3] % patch_size == 0, "Image dimensions must be divisible by patch size."
    h = w = img.shape[2] // patch_size
    patches = img.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
    patches = patches.view(img.shape[0], h * w, -1)
    return patches

def unpatchify(x, patch_size):
    """
    Reconstructs an image tensor from patches.
    Args:
        patches: (N, L, patch_size**2 * C) tensor.
        img_size: Tuple (H, W) for the image size.
        patch_size: Patch size (int).
    Returns:
        Images: (N, C, H, W) tensor.
    """
    x_shape = x.shape
    h = w = int(x_shape[1]**.5)
    x = x.reshape((x.shape[0], h, w, patch_size, patch_size, 1))
    x = torch.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(shape=(x.shape[0], 1, h * patch_size, h * patch_size))
    return imgs

def visualize_attention_heatmaps(mae_model, dataloader, device, img_size, patch_size, num_images=5, color_channels = 1):
    """
    Visualize attention heatmaps from the MAE for masked images.
    
    Args:
        mae_model: Trained MAE model.
        dataloader: DataLoader providing input images.
        device: Device (CPU or GPU).
        img_size: Tuple (H, W) for the image size.
        patch_size: Patch size (int).
        num_images: Number of images to visualize.
    """
    mae_model.eval()
    with torch.no_grad():
        images, labels = next(iter(dataloader))
        images = images[:num_images].to(device)
        labels = labels[:num_images].to(device)
        _, pred, original_mask, attention_weights = mae_model(images.float(), return_attention=True)
        reconstructed = mae_model.unpatchify(pred)
        reconstructed = torch.einsum('nchw->nhwc', reconstructed).detach().cpu() 
        reconstructed = reconstructed.reshape((num_images, color_channels, *img_size))

        mask = original_mask.unsqueeze(-1).repeat(1, 1, patch_size**2)  # (N, H*W, p*p*C)
        mask = unpatchify(mask, patch_size)
        masked_images = images * (1 - mask)
        augmented = ((1-mask)*images) + (mask*reconstructed)
        attention_weights = np.maximum.reduce(attention_weights, axis = 0)
        attention_weights = attention_weights.mean(axis=1) # Aggregate across heads
        
        patched_images = patchify(masked_images, patch_size=2)
        for i in range(num_images):
            original_image = images[i].cpu().squeeze().numpy()
            masked_image = masked_images[i].cpu().squeeze().numpy()
            # attention_map = attention_weights[i].reshape(img_size)
            attention_map = attention_weights[i, 1:, 1:]
            attention_map = np.abs(attention_map)
            patched_image = patched_images[i]
            patched_image = patched_image[original_mask[i] == 0, :]

            mask_attention = attention_map @ patched_image.numpy()
            mask_attention = np.abs(mask_attention)
            mask_attention = (mask_attention - mask_attention.min()) / (
                mask_attention.max() - mask_attention.min()
            )  # Normalize to [0, 1]

            num_patches = (img_size[0]//patch_size)*(img_size[1]//patch_size)
            full_attention_map = np.zeros((num_patches, patch_size**2))
            flattened_mask = original_mask[i]
            full_attention_map[flattened_mask == 0] = mask_attention
            full_attention_map = full_attention_map.reshape((1, *full_attention_map.shape[:]))
            full_attention_map = torch.from_numpy(full_attention_map)
            full_attention_map = unpatchify(full_attention_map,  patch_size=patch_size)
            full_attention_map = full_attention_map.reshape(img_size)
            
            # Plotting
            fig, ax = plt.subplots(1, 4, figsize=(12, 4))
            ax[0].imshow(original_image, cmap="gray")
            ax[0].set_title("Original Image")
            ax[0].axis("off")
            
            ax[1].imshow(masked_image, cmap="gray")
            ax[1].set_title("Masked Image")
            ax[1].axis("off")

            ax[2].imshow(augmented[i,0], cmap="gray")
            ax[2].set_title("Reconstruction")
            ax[2].axis("off")
            
            ax[3].imshow(original_image, cmap="gray")
            attention_heatmap = ax[3].imshow(full_attention_map, cmap="hot", alpha=0.7)
            ax[3].set_title("Attention Heatmap")
            ax[3].axis("off")
            
            cbar = fig.colorbar(attention_heatmap, ax=ax[3], orientation="vertical", fraction=0.046, pad=0.04)
            cbar.set_label("Attention Intensity", rotation=270, labelpad=15)
            plt.savefig(f"./final graphics/mae_map_{labels[i]}.png")
            plt.show()

# Example usage:
img_size = (28, 28)
patch_size = 2
# Prepare data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), 
                       (0.2023, 0.1994, 0.2010))
])

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=8, shuffle=True)
print("data loaded")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mae_model = model = CustomCIFAR10MaskedAutoencoder().to(device)
mae_model.load_state_dict(torch.load('cifar10_mae_weights_10_epochs_custom.pth',  map_location=torch.device('cpu')))  # Load the saved weights
print("model loaded")

# visualize_latent_space_with_cluster_radius(mae_model, testloader, device, num_samples_per_class=100)

visualize_attention_heatmaps(mae_model, testloader, device, img_size, patch_size)