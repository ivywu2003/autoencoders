import argparse
import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms
from cifar10_models import CIFAR10Autoencoder, CIFAR10MaskedAutoencoder, CustomCIFAR10MaskedAutoencoder

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
        else:  # mae
            predicted, mask = model(images)
            # Unmask
            reconstructed = predicted * mask + images * (1 - mask)
            # Denormalize images
            images = denormalize(images, device)
            reconstructed = denormalize(reconstructed, device)
    
    # Plot original and reconstructed images
    fig, axes = plt.subplots(2, num_images, figsize=(2*num_images, 4))
    
    for i in range(num_images):
        # Original images
        orig_img = images[i].cpu().permute(1, 2, 0).clamp(0, 1)
        axes[0, i].imshow(orig_img)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original', pad=10)
        
        # Reconstructed images
        recon_img = reconstructed[i].cpu().permute(1, 2, 0).clamp(0, 1)
        axes[1, i].imshow(recon_img)
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Reconstructed', pad=10)
    
    plt.tight_layout()
    plt.show()
    # plt.savefig(f'reconstructions_{model_type}.png')
    # plt.close()

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['ae', 'mae'], default='ae')
    args = parser.parse_args()
    
    # Load the model
    if args.model == 'ae':
        model = CIFAR10Autoencoder().to(device)
    else:
        model = CustomCIFAR10MaskedAutoencoder().to(device)
    
    model.load_state_dict(torch.load(f'cifar10_{args.model}_weights_10_epochs_custom.pth', map_location=torch.device('cpu')))
    model.eval()
    
    # Prepare data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])
    
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=8, shuffle=True)
    
    # Visualize reconstructions
    visualize_reconstructions(model, testloader, device, args.model)
    # print(f'Reconstructions saved as reconstructions_{args.model}.png')
