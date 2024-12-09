import argparse
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms
from cifar10_models import CIFAR10Autoencoder, CustomCIFAR10MaskedAutoencoder, CIFAR10DenoisingAutoencoder

def denormalize(x, device):
    """Denormalize images"""
    mean = torch.tensor([0.5]).to(device)
    std = torch.tensor([0.5]).to(device)
    return x * std + mean

def visualize_reconstructions(model, dataloader, device, model_type='ae', num_images=8):
    """Visualize original images and their reconstructions"""
    model.eval()
    
    # Get a batch of images
    images, _ = next(iter(dataloader))
    images = images[:num_images].to(device)
    
    with torch.no_grad():
        if model_type == 'ae':
            reconstructed = model(images)
            # Denormalize images
            images = denormalize(images, device)
            reconstructed = denormalize(reconstructed, device)
        elif model_type == 'mae':  # mae
            predicted, mask = model(images)
            # Unmask
            reconstructed = predicted * mask + images * (1 - mask)
            # Denormalize images
            images = denormalize(images, device)
            reconstructed = denormalize(reconstructed, device)
        elif model_type == 'dae':  # dae
            # Add noise to images
            noise = torch.randn_like(images) * 0.1
            noisy_images = images + noise
            noisy_images = torch.clamp(noisy_images, 0., 1.)

            reconstructed = model(noisy_images)
    
    # Plot original and reconstructed images
    fig, axes = plt.subplots(3, num_images, figsize=(2*num_images, 6))
    
    for i in range(num_images):
        # Original images
        orig_img = images[i].cpu().permute(1, 2, 0).clamp(0, 1)
        axes[0, i].imshow(orig_img)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original', pad=10)

        if model_type == 'dae':
            # No mask for DAE
            axes[1, i].imshow(noisy_images[i].cpu().permute(1, 2, 0).clamp(0, 1))
            axes[1, i].axis('off')
            if i == 0:
                axes[1, i].set_title('Noisy Input', pad=10)
        elif model_type == 'mae':
            # Masked input
            masked_img = images[i].cpu().permute(1, 2, 0).clamp(0, 1) * (1 - mask[i].cpu().permute(1, 2, 0).clamp(0, 1))
            axes[1, i].imshow(masked_img)
            axes[1, i].axis('off')
            if i == 0:
                axes[1, i].set_title('Masked Input', pad=10)
        
        # Reconstructed images
        recon_img = reconstructed[i].cpu().permute(1, 2, 0).clamp(0, 1)
        axes[2, i].imshow(recon_img)
        axes[2, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Reconstructed', pad=10)
    
    plt.tight_layout()
    plt.show()
    # plt.savefig(f'reconstructions_{model_type}.png')
    # plt.close()

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['ae', 'mae', 'dae'], default='ae')
    args = parser.parse_args()
    
    # Load the model
    if args.model == 'ae':
        model = CIFAR10Autoencoder().to(device)
    elif args.model == 'dae':
        model = CIFAR10DenoisingAutoencoder().to(device)
    elif args.model == 'mae':
        model = CustomCIFAR10MaskedAutoencoder().to(device)
    
    model.load_state_dict(torch.load(f'cifar10_{args.model}_weights_20_epochs.pth', map_location=torch.device('cpu')))
    model.eval()
    
    # Prepare data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])
    if args.model == 'dae':
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=8, shuffle=True)
    
    # Visualize reconstructions
    visualize_reconstructions(model, testloader, device, args.model)
    # print(f'Reconstructions saved as reconstructions_{args.model}.png')
