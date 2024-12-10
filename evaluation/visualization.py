import torch
import matplotlib.pyplot as plt
from loaders import *

def denormalize(x):
    """Denormalize images"""
    mean = torch.tensor([0.5])
    std = torch.tensor([0.5])
    return x * std + mean

def clamp(x):
    """Clamp images between 0 and 1"""
    return (x - torch.min(x)) / (torch.max(x) - torch.min(x))

def visualize_mae_pairs(dataloader, model, mask_ratio=0.75):
    with torch.no_grad():
        for images, _ in dataloader:
            images = images[:5]
            print(images.shape)
            loss, pred, mask = model(images.float())
            mask = mask.unsqueeze(-1).repeat(1, 1, 2**2*3)
            mask = model.unpatchify(mask)
            reconstructed = model.unpatchify(pred)
            reconstructed = reconstructed * mask + images * (1 - mask)

            # renormalize
            for i in range(5):
                image = clamp(images[i])
                r = clamp(reconstructed[i])
                print(image.shape)
                print(reconstructed.shape)
                
                plt.subplot(2, 5, i + 1)  # Top row
                plt.imshow(image.permute(1, 2, 0).numpy())
                plt.title("Original")  # Add title
                plt.axis("off")  # Remove axis for better visual appeal

                plt.subplot(2, 5, i + 6)  # Bottom row
                plt.imshow(r.permute(1, 2, 0).numpy())
                plt.title("Reconstructed")  # Add title
                plt.axis("off")
            
            plt.suptitle("MAE Reconstruction for Color-Jittered Images", fontsize=16, fontweight='bold')
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig('final_writeup/Reconstructions/jitter_mae_reconstruction.png')
            break
    return

def visualize_dae_pairs(dataloader, model):
    with torch.no_grad():
        for images, _ in dataloader:
            images = images[:5]
            noise = torch.randn_like(images) * 0.1
            noisy_images = images + noise

            reconstructed = model(noisy_images.float())
            images = denormalize(images)
            reconstructed = denormalize(reconstructed)

            # renormalize
            for i in range(5):
                image = clamp(images[i])
                r = clamp(reconstructed[i])
                
                plt.subplot(2, 5, i + 1)  # Top row
                plt.imshow(image.permute(1, 2, 0).numpy())
                plt.title("Original")  # Add title
                plt.axis("off")  # Remove axis for better visual appeal

                plt.subplot(2, 5, i + 6)  # Bottom row
                plt.imshow(r.permute(1, 2, 0).numpy())
                plt.title("Reconstructed")  # Add title
                plt.axis("off")
            
            plt.suptitle("DAE Reconstruction for Color-Jittered Images", fontsize=16, fontweight='bold')
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig('final_writeup/Reconstructions/jitter_dae_reconstruction.png')
            break
    return


def visualize_single_pair(image, reconstructed):
    plt.subplot(1, 2, 1)
    plt.imshow(image.squeeze(0).permute(1, 2, 0).numpy())
    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed.squeeze(0).permute(1, 2, 0).numpy())
    plt.show()
    return


if __name__ == "__main__":
    trainloader, testloader = load_jittered_cifar10_for_dae()
    model = load_jittered_dae()

    visualize_dae_pairs(trainloader, model)