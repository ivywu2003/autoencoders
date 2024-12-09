import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torchvision
from torchvision import transforms
import skimage.metrics
import matplotlib.pyplot as plt
from mnist_mae_vit.mae_vit import MaskedAutoencoderViTForMNIST
from mnist_mae_vit.dae import ConvDenoiser
from cifar10_task.cifar10_models import CIFAR10DenoisingAutoencoder, CustomCIFAR10MaskedAutoencoder


def load_mnist_for_mae():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    testset = torchvision.datasets.MNIST('./data', download=True, train=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, shuffle=False)
    
    return testloader

def load_mnist_for_dae():
    transform = transforms.ToTensor()
    testset = torchvision.datasets.MNIST('./data', download=True, train=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, shuffle=False)
    
    return testloader

def load_cifar10_for_mae():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ])

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(
        testset, shuffle=False)
    
    return testloader

def load_cifar10_for_dae():
    transform = transforms.ToTensor()

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(
        testset, shuffle=False)
    
    return testloader

def denormalize(x):
    """Denormalize images"""
    mean = torch.tensor([0.5])
    std = torch.tensor([0.5])
    return x * std + mean

def load_mnist_mae():
    mae_model = MaskedAutoencoderViTForMNIST(
        img_size=28, patch_size=2, in_chans=1, 
        embed_dim=64, depth=4, num_heads=4, 
        decoder_embed_dim=32, decoder_depth=2, decoder_num_heads=4
    )
    mae_model.load_state_dict(torch.load("mnist_mae_vit/mae_weights.pth", weights_only=True))
    mae_model.eval()
    return mae_model

def load_cifar10_mae():
    mae_model = CustomCIFAR10MaskedAutoencoder()
    mae_model.load_state_dict(torch.load("cifar10_task/cifar10_mae_weights_20_epochs_custom.pth", weights_only=True, map_location=torch.device('cpu')))
    mae_model.eval()
    return mae_model

def load_mnist_dae():
    dae_model = ConvDenoiser()
    dae_model.load_state_dict(torch.load('mnist_mae_vit/dae_weights.pth', weights_only=True)) 
    dae_model.eval()
    return dae_model

def load_cifar10_dae():
    dae_model = CIFAR10DenoisingAutoencoder()
    dae_model.load_state_dict(torch.load('cifar10_task/cifar10_dae_weights_20_epochs.pth', weights_only=True, map_location=torch.device('cpu'))) 
    dae_model.eval()
    return dae_model


def single_image_psnr(img1, img2):
    """
    Calculate the Peak Signal-to-Noise Ratio (PSNR) between two images.
    """
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * torch.log10(1 / torch.sqrt(mse))

def generate_mae_psnr_plot(image_loader, model, save_path, mask_ratio=0.75):
    psnrs = []

    with torch.no_grad():
        for image, _ in image_loader:
            # MNIST VERSION
            # _, pred, mask = model(image.float(), mask_ratio=mask_ratio)
            # mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2)  # (N, H*W, p*p*3)
            # mask = model.unpatchify(mask)

            # reconstructed = model.unpatchify(pred)
            # reconstructed = ((1-mask)*image) + (mask*reconstructed)

            # CIFAR10 VERSION
            reconstructed, mask = model(image.float(), mask_ratio=mask_ratio)
            reconstructed = reconstructed * mask + image * (1 - mask)
            image = denormalize(image)
            reconstructed = denormalize(reconstructed)

            # renormalize
            image = (image - torch.min(image))/(torch.max(image) - torch.min(image))
            reconstructed = (reconstructed - torch.min(reconstructed))/(torch.max(reconstructed) - torch.min(reconstructed))

            # plt.subplot(1, 2, 1)
            # plt.imshow(image.squeeze(0).permute(1, 2, 0).numpy())
            # plt.subplot(1, 2, 2)
            # plt.imshow(reconstructed.squeeze(0).permute(1, 2, 0).numpy())
            # plt.show()
            psnr = single_image_psnr(image, reconstructed)
            psnrs.append(psnr)

    # Create histogram
    plt.hist(psnrs, bins=10, edgecolor='black')

    # Add labels and title
    plt.xlabel('PSNR')
    plt.ylabel('Frequency')
    plt.title('Histogram of PSNR for MAE reconstruction')

    # Save plot to folder
    plt.savefig('evaluation/' + save_path)
    plt.show()
    return

def generate_dae_psnr_plot(image_loader, model, save_path):
    psnrs = []

    with torch.no_grad():
        for image, _ in image_loader:
            # MNIST VERSION
            # noise_factor = 0.5
            # noisy_imgs = image + noise_factor * torch.randn(image.shape)
            # noisy_imgs = np.clip(noisy_imgs, 0., 1.)
            # reconstructed, _ = model(noisy_imgs)
            
            # CIFAR10 VERSION
            noise = torch.randn_like(image) * 0.1
            noisy_images = image + noise
            noisy_images = torch.clamp(noisy_images, 0., 1.)
            reconstructed = model(image.float())
            image = denormalize(image)
            reconstructed = denormalize(reconstructed)

            # renormalize
            image = (image - torch.min(image))/(torch.max(image) - torch.min(image))
            reconstructed = (reconstructed - torch.min(reconstructed))/(torch.max(reconstructed) - torch.min(reconstructed))

            # plt.subplot(1, 2, 1)
            # plt.imshow(image.squeeze(0).permute(1, 2, 0).numpy())
            # plt.subplot(1, 2, 2)
            # plt.imshow(reconstructed.squeeze(0).permute(1, 2, 0).numpy())
            # plt.show()

            psnr = single_image_psnr(image, reconstructed)
            psnrs.append(psnr)

    # Create histogram
    plt.hist(psnrs, bins=10, edgecolor='black')

    # Add labels and title
    plt.xlabel('PSNR')
    plt.ylabel('Frequency')
    plt.title('Histogram of PSNR for DAE reconstruction')

    # Save plot to folder
    plt.savefig('evaluation/' + save_path)
    plt.show()
    return


def structural_similarity(true_images, generated_images, greyscale=False):
    ssim = [skimage.metrics.structural_similarity(true_image, generated_image, channel_axis=0, data_range=1) for true_image, generated_image in zip(true_images, generated_images)]
    return ssim

def generate_mae_ssim_plot(image_loader, model, save_path, greyscale, mask_ratio=0.75):
    true_images = []
    generated_images = []

    with torch.no_grad():
        for image, _ in image_loader:
            # MNIST VERSION
            # _, pred, mask = model(image.float(), mask_ratio=mask_ratio)
            # mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2)  # (N, H*W, p*p*3)
            # mask = model.unpatchify(mask)

            # reconstructed = model.unpatchify(pred)
            # reconstructed = ((1-mask)*image) + (mask*reconstructed)

            # CIFAR10 VERSION
            reconstructed, mask = model(image.float(), mask_ratio=mask_ratio)
            reconstructed = reconstructed * mask + image * (1 - mask)
            image = denormalize(image)
            reconstructed = denormalize(reconstructed)

            # renormalize
            image = (image - torch.min(image))/(torch.max(image) - torch.min(image))
            reconstructed = (reconstructed - torch.min(reconstructed))/(torch.max(reconstructed) - torch.min(reconstructed))

            # plt.subplot(1, 2, 1)
            # plt.imshow(image.squeeze(0).squeeze(0).numpy(), cmap='gray')
            # plt.subplot(1, 2, 2)
            # plt.imshow(reconstructed.squeeze(0).squeeze(0).numpy(), cmap='gray')
            # plt.show()
            # break

            true_images.append(image.squeeze(0).numpy())
            generated_images.append(reconstructed.squeeze(0).numpy())
    
    ssim0 = structural_similarity(true_images, generated_images, greyscale=greyscale)

    # Create histogram
    plt.hist(ssim0, bins=10, edgecolor='black')

    # Add labels and title
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of SSIM for MAE reconstruction')

    # Show plot
    plt.savefig('evaluation/' + save_path)
    plt.show()
    return

def generate_dae_ssim_plot(image_loader, model, save_path, greyscale):
    true_images = []
    generated_images = []

    with torch.no_grad():
        for image, _ in image_loader:
            # MNIST VERSION
            # noise_factor = 0.5
            # noisy_imgs = image + noise_factor * torch.randn(image.shape)
            # noisy_imgs = np.clip(noisy_imgs, 0., 1.)
            # reconstructed, _ = model(noisy_imgs)
            
            # CIFAR10 VERSION
            noise = torch.randn_like(image) * 0.1
            noisy_images = image + noise
            noisy_images = torch.clamp(noisy_images, 0., 1.)
            reconstructed = model(image.float())
            image = denormalize(image)
            reconstructed = denormalize(reconstructed)

            # plt.subplot(1, 2, 1)
            # plt.imshow(image.squeeze(0).squeeze(0).numpy(), cmap='gray')
            # plt.subplot(1, 2, 2)
            # plt.imshow(reconstructed.squeeze(0).squeeze(0).numpy(), cmap='gray')
            # plt.show()
            # break

            # renormalize
            image = (image - torch.min(image))/(torch.max(image) - torch.min(image))
            reconstructed = (reconstructed - torch.min(reconstructed))/(torch.max(reconstructed) - torch.min(reconstructed))

            true_images.append(image.squeeze(0).numpy())
            generated_images.append(reconstructed.squeeze(0).numpy())
    
    ssim0 = structural_similarity(true_images, generated_images, greyscale=greyscale)

    # Create histogram
    plt.hist(ssim0, bins=10, edgecolor='black')

    # Add labels and title
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of SSIM for DAE reconstruction')

    # Show plot
    plt.savefig('evaluation/' + save_path)
    plt.show()
    return

if __name__ == "__main__":
    # reconstruct_adjusted_reflectance()
    # best_image = generate_psnr_plot()
    # best_image = best_image.squeeze(0)
    # best_image = torch.permute(best_image, (1, 2, 0))
    # best_image = best_image.detach().numpy()

    # plt.figure(figsize=(10, 5))
    # plt.subplot(1, 1, 1)
    # plt.title('Best Image')
    # plt.imshow(best_image)
    # plt.axis('off')
    # plt.show()
    mnist_mae_dataloader = load_mnist_for_mae()
    mnist_dae_dataloader = load_mnist_for_dae()
    cifar10_mae_dataloader = load_cifar10_for_mae()
    cifar10_dae_dataloader = load_cifar10_for_dae()

    mae_mnist_model = load_mnist_mae()
    dae_mnist_model = load_mnist_dae()
    mae_cifar10_model = load_cifar10_mae()
    dae_cifar10_model = load_cifar10_dae()


    # generate_mae_psnr_plot(mnist_mae_dataloader, mae_mnist_model, 'mae_psnr_mnist.png')
    generate_mae_psnr_plot(cifar10_mae_dataloader, mae_cifar10_model, 'mae_psnr_cifar10.png')
    # generate_dae_psnr_plot(mnist_dae_dataloader, dae_mnist_model, 'dae_psnr_mnist.png')
    generate_dae_psnr_plot(cifar10_dae_dataloader, dae_cifar10_model, 'dae_psnr_cifar10.png')

    # generate_mae_ssim_plot(mnist_mae_dataloader, mae_mnist_model, 'mae_ssim_mnist.png', greyscale=True)
    generate_mae_ssim_plot(cifar10_mae_dataloader, mae_cifar10_model, 'mae_ssim_cifar10.png', greyscale=False)
    # generate_dae_ssim_plot(mnist_dae_dataloader, dae_mnist_model, 'dae_ssim_mnist.png', greyscale=True)
    generate_dae_ssim_plot(cifar10_dae_dataloader, dae_cifar10_model, 'dae_ssim_cifar10.png', greyscale=False)