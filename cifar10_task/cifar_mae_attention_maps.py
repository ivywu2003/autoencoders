from cifar10_models import CIFAR10Autoencoder, CustomCIFAR10MaskedAutoencoder


import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import torchvision
from einops import repeat, rearrange

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

def unpatchify(x, patch_size, mask=None):
    """
    Reconstructs an image tensor from patches using a mask.
    Args:
        patches: (N, L, patch_size**2 * C) tensor, where C is number of channels (3 for RGB).
        patch_size: Patch size (int).
        mask: (H, W) tensor where 0 indicates unmasked pixels.
    Returns:
        Images: (N, C, H, W) tensor.
    """
    x_shape = x.shape
    num_channels = x_shape[2] // (patch_size * patch_size)  # Calculate number of channels
    
    if mask is not None:
        # Initialize output tensor
        H, W = mask.shape
        output = torch.zeros((x_shape[0], num_channels, H, W), device=x.device)
        
        # Get indices of unmasked patches
        patch_rows = torch.arange(H // patch_size).repeat_interleave(W // patch_size)
        patch_cols = torch.arange(W // patch_size).repeat(H // patch_size)
        
        # For each patch position
        patch_idx = 0
        for i in range(0, H, patch_size):
            for j in range(0, W, patch_size):
                if not mask[i:i+patch_size, j:j+patch_size].any():  # If patch is unmasked
                    # Reshape the patch data
                    patch_data = x[:, patch_idx].reshape(x_shape[0], num_channels, patch_size, patch_size)
                    # Place it in the output
                    output[:, :, i:i+patch_size, j:j+patch_size] = patch_data
                    patch_idx += 1
        return output
    else:
        h = w = int(x_shape[1]**.5)  # Number of patches in each dimension
        x = x.reshape((x.shape[0], h, w, patch_size, patch_size, num_channels))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], num_channels, h * patch_size, h * patch_size))
        return imgs

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mae_model = CustomCIFAR10MaskedAutoencoder(return_attention = True).to(device)
    mae_model.load_state_dict(torch.load('cifar10_mae_weights_10_epochs_custom.pth',  map_location=torch.device('cpu')))

    mae_model.eval()
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=8, shuffle=True)

    fig, ax = plt.subplots(8, 5, figsize=(12, 10))

    img_size = (32, 32)
    patch_size = 2
    with torch.no_grad():
        images, labels = next(iter(testloader))
        images = images.to(device)
        labels = labels.to(device)
        reconstructed, mask, attention_weights = mae_model(images.float())  # Attention weights has shape [12, 8, 3, 65, 65]

        # Get the patches
        patched_images = patchify(images, patch_size=2)

        # Process each image in the batch
        for i in range(images.shape[0]):
            # Get the original image
            original_image = images[i].permute(1, 2, 0).cpu().numpy()

            # Convert the masks to patches
            raw_indices = torch.where(mask[i][0] == 0)
            # Convert to patch indices by dividing by patch size and removing duplicates
            patch_rows = raw_indices[0] // 2
            patch_cols = raw_indices[1] // 2
            patch_indices = torch.stack([patch_rows, patch_cols])
            unique_patches = torch.unique(patch_indices.T, dim=0) # In terms of patch indices

            patched_image = patched_images[i][unique_patches[:, 0] * 16 + unique_patches[:, 1]]
            
            # Get attention from last layer for this image
            last_layer_attention = attention_weights[-1][i]  # [num_heads, N_patches, N_patches]
            
            # Remove CLS token attention and keep only patch-to-patch attention
            patch_attention = last_layer_attention[:, 1:, 1:]  # Remove CLS token
            
            # For each head, reshape attention to image dimensions
            H = W = 32  # Should be 16 for 32x32 image with patch_size=2
            attention_maps = []
            
            for head in range(3):
                # Get attention map for this head
                head_attention = patch_attention[head]  # [N_patches, N_patches]
                
                # Average attention across source tokens (how much attention each position receives)
                avg_attention = head_attention.mean(dim=0)  # [N_patches]
                
                # Reshape to 2D grid
                attention_map = avg_attention
                
                # Min-max normalize
                attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min())

                # Add to real image size
                attention_image = torch.zeros((H, W))
                for j in range(len(unique_patches)):
                    patch_row = unique_patches[j, 0]
                    patch_col = unique_patches[j, 1]
                    img_row = patch_row * 2
                    img_col = patch_col * 2
                    attention_image[img_row:img_row + 2, img_col:img_col + 2] = attention_map[j]
                attention_maps.append(attention_image)

            masked_image = original_image.copy()
            mask_i = mask[i]  # Shape: [3, 32, 32]
            
            # Apply mask to each channel
            for c in range(3):  # RGB channels
                channel_mask = mask_i[c]  # Shape: [32, 32]
                masked_image[:, :, c][channel_mask == 1] = 0

            ax[i, 0].imshow(original_image)
            ax[i, 0].set_title('Original')
            ax[i, 0].axis('off')

            ax[i, 1].imshow(masked_image)
            ax[i, 1].set_title('Masked')
            ax[i, 1].axis('off')

            # Plot attention heatmaps with consistent colormap
            vmin, vmax = 0, 1  # Fixed range for better comparison
            for idx, attn_map in enumerate(attention_maps):
                ax_idx = i, idx + 2
                # Plot original image first
                ax[ax_idx].imshow(original_image)
                # Overlay attention map with transparency
                im = ax[ax_idx].imshow(attn_map.cpu().numpy(), 
                                     cmap='hot',
                                     alpha=0.7,  # Adjust transparency
                                     vmin=vmin,
                                     vmax=vmax)
                ax[ax_idx].set_title(f'Attention Map - Head {idx+1}')
                ax[ax_idx].axis('off')
                plt.colorbar(im, ax=ax[ax_idx])

    plt.tight_layout()
    # plt.show()
    plt.savefig('cifar10_mae_attention_maps.png')
    plt.close()
