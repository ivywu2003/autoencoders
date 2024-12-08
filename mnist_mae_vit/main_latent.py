from mae_vit import MaskedAutoencoderViTForMNIST
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print("data loaded")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mae_model = MaskedAutoencoderViTForMNIST(
    img_size=28, patch_size=2, in_chans=1, 
    embed_dim=64, depth=4, num_heads=4, 
    decoder_embed_dim=32, decoder_depth=2, decoder_num_heads=4
).to(device)

print("model initialized")
mae_model.load_state_dict(torch.load('mae_weights_vit_patch2_20epochs.pth'))  # Load the saved weights
print("model loaded")


def visualize_reconstruction(model, dataloader, device, mask_ratio=0.75, num_images=5):
    model.eval()
    images, _ = next(iter(dataloader))
    images = images[:num_images].to(device)


    loss, pred, mask, latent = model(images.float(), mask_ratio=mask_ratio)
    print(loss)
    print(mask)
    # print(pred[0])
    reconstructed = model.unpatchify(pred)
    print(reconstructed.shape)
    reconstructed = torch.einsum('nchw->nhwc', reconstructed).detach().cpu() 
    print(reconstructed.shape)

    # Convert to CPU for plotting
    images = images.cpu().numpy()
    reconstructed = reconstructed.numpy()



    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2)  # (N, H*W, p*p*3)
    mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu().numpy()
    mask = mask.reshape((5, 1, 28, 28))
    reconstructed = reconstructed.reshape((5, 1, 28, 28))
    #Reconstructed + visible
    print("images shape", images.shape)
    print("mask overlay shape", mask.shape)
    print("Recon shape", reconstructed.shape)
    augmented = ((1-mask)*images) + (mask*reconstructed)
    print(augmented.shape)
    # Plot original, masked, and reconstructed images
    fig, axes = plt.subplots(4, num_images, figsize=(15, 7))
    masked_image = images*(1-mask)
    for i in range(num_images):
        # Original Image
        axes[0, i].imshow(images[i, 0], cmap="gray")
        axes[0, i].set_title("Original")
        axes[0, i].axis("off")

        # Masked Image: overlay mask on original image
        axes[1, i].imshow(masked_image[i, 0], cmap="gray")
        # axes[1, i].imshow(mask_overlay[i, 0], cmap="gray", alpha=0.5)  # Mask overlay with transparency
        axes[1, i].set_title("Masked")
        axes[1, i].axis("off")

        # Reconstructed Image
        axes[2, i].imshow(reconstructed[i, 0], cmap="gray")
        axes[2, i].set_title("Reconstructed")
        axes[2, i].axis("off")

        # Reconstructed Image
        axes[3, i].imshow(augmented[i, 0], cmap="gray")
        axes[3, i].set_title("Augmented")
        axes[3, i].axis("off")

    plt.show()
    return augmented


visualize_reconstruction(mae_model, test_loader, device)