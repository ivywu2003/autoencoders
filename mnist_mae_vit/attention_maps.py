from mae_vit import MaskedAutoencoderViTForMNIST

import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

def patchify(img):
    """
    imgs: (N, 1, H, W)
    x: (N, L, patch_size**2 *1)
    """
    p = 2
    print(img.shape)
    assert img.shape[1] == img.shape[2] and img.shape[1] % p == 0
    h = w = img.shape[1] // p
    x = img.reshape(shape=(1, h, p, w, p))
    x = torch.einsum('chpwq->hwpqc', x)
    x = x.reshape(shape=(h * w, p**2))
    return x

def unpatchify(x):
    """
    x: (N, L, patch_size**2 *3)
    imgs: (N, 3, H, W)
    """
    # print("x shape", x.shape)
    x = torch.tensor(x, dtype=torch.float32)
    p = 2
    h = w = int(x.shape[0]**.5)
    
    x = x.reshape((h, w, p, p, 1))
    x = torch.einsum('hwpqc->chpwq', x)
    imgs = x.reshape((1, h * p, h * p))
    return imgs

def visualize_attention_heatmaps(mae_model, dataloader, device, num_images=1):
    """
    Visualize attention heatmaps from the MAE for masked MNIST images.
    
    Args:
        mae_model: Trained MAE model.
        dataloader: DataLoader providing input images.
        device: Device (CPU or GPU).
        num_images: Number of images to visualize.
    """
    mae_model.eval()
    images_shown = 0
    with torch.no_grad():
        images, labels = next(iter(dataloader))
        images = images[:num_images].to(device)
        labels = labels[:num_images].to(device)
        # Forward pass to retrieve attention weights and reconstruction
        loss, reconstructions, original_mask, attention_weights = mae_model(images.float(), return_attention=True)
        
        mask = original_mask.detach()
        mask = mask.unsqueeze(-1).repeat(1, 1, mae_model.patch_embed.patch_size[0]**2)  # (N, H*W, p*p*3)
        mask = mae_model.unpatchify(mask)  # 1 is removing, 0 is keeping
        mask = torch.einsum('nchw->nhwc', mask).detach().cpu().numpy()
        mask = mask.reshape((num_images, 1, 28, 28))
        masked_image_list = images * (1 - mask)
        original_mask = original_mask.detach().cpu().numpy()
        
        for block in range(4):
            attention_block = attention_weights[block]
            for i in range(num_images):
                original_image = images[i].cpu().squeeze().numpy()
                masked_image = masked_image_list[i]
                patched_image = patchify(masked_image).numpy()
                patched_image = patched_image[original_mask[i] == 0, :]
                
                # Take the absolute value of the attention map
                attention_map = attention_block[i, :, 1:, 1:].numpy()
                attention_map = np.abs(attention_map)
                attention_map = np.max(attention_map, axis=0)  # Aggregate attention across heads
                
                print(attention_map)
                mask_attention = attention_map @ patched_image
                mask_attention = np.abs(mask_attention)
                mask_attention = (mask_attention - mask_attention.min()) / (
                    mask_attention.max() - mask_attention.min()
                )  # Normalize to [0, 1]
                full_attention_map = np.zeros((196, 4))
                flattened_mask = original_mask[i]
                full_attention_map[flattened_mask == 0] = mask_attention
                # full_attention_map = np.abs(full_attention_map)  # Absolute value
                # full_attention_map = (full_attention_map - full_attention_map.min()) / (
                #     full_attention_map.max() - full_attention_map.min()
                # )  # Normalize to [0, 1]
                full_attention_map = unpatchify(full_attention_map)
                print(full_attention_map)
                # Plot the original, masked, and attention heatmap
                fig, ax = plt.subplots(1, 3, figsize=(12, 4))
                ax[0].imshow(original_image, cmap="gray")
                ax[0].set_title("Original Image")
                ax[0].axis("off")
                
                ax[1].imshow(masked_image.squeeze(), cmap="gray")
                ax[1].set_title("Masked Image")
                ax[1].axis("off")
                
                attention_overlay = ax[2].imshow(original_image, cmap="gray")
                attention_heatmap = ax[2].imshow(full_attention_map.squeeze(), cmap="jet", alpha=0.7)  # Overlay attention map
                ax[2].set_title("Attention Heatmap")
                ax[2].axis("off")
                
                # Add color legend for the attention heatmap
                cbar = fig.colorbar(attention_heatmap, ax=ax[2], orientation="vertical", fraction=0.046, pad=0.04)
                cbar.set_label("Attention Intensity", rotation=270, labelpad=15)
                
                plt.title(f"Block number: {block}")
                plt.savefig(f"./heatmaps/{labels[i]}_{block}")
                plt.show()
                
                images_shown += 1


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print("data loaded")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mae_model = MaskedAutoencoderViTForMNIST(
    img_size=28, patch_size=2, in_chans=1, 
    embed_dim=64, depth=4, num_heads=4, 
    decoder_embed_dim=32, decoder_depth=2, decoder_num_heads=4
).to(device)
mae_model.load_state_dict(torch.load('mae_weights_vit_patch2_20epochs.pth'))  # Load the saved weights
print("model loaded")

visualize_attention_heatmaps(mae_model, train_loader, device)


