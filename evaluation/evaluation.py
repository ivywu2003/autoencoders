import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torchvision
from torchvision import transforms
import skimage.metrics
import torchmetrics
import matplotlib.pyplot as plt
from mnist_mae_vit.mae_vit import MaskedAutoencoderViTForMNIST
from mnist_mae_vit.dae import ConvDenoiser
from cifar10_task.cifar10_models import CIFAR10Autoencoder, CustomCIFAR10MaskedAutoencoder


def load_mnist():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    testset = torchvision.datasets.MNIST('./data', download=True, train=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, shuffle=False)
    
    return testloader

def load_cifar10():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616])
    ])

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(
        testset, shuffle=False)
    
    return testloader

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
    dae_model = CIFAR10Autoencoder()
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
            loss, pred, mask = model(image.float(), mask_ratio=mask_ratio)
            reconstructed = model.unpatchify(pred)

            # renormalize
            image = (image - torch.min(image))/(torch.max(image) - torch.min(image))
            reconstructed = (reconstructed - torch.min(reconstructed))/(torch.max(reconstructed) - torch.min(reconstructed))
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
            reconstructed, _ = model(image.float())

            # renormalize
            image = (image - torch.min(image))/(torch.max(image) - torch.min(image))
            reconstructed = (reconstructed - torch.min(reconstructed))/(torch.max(reconstructed) - torch.min(reconstructed))
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
    print(true_images[0].shape)
    print(generated_images[0].shape)

    if greyscale:
        channel_axis = 0
    else:
        channel_axis = 2
    ssim = [skimage.metrics.structural_similarity(true_image, generated_image, channel_axis=channel_axis, data_range=1) for true_image, generated_image in zip(true_images, generated_images)]
    return ssim

def generate_mae_ssim_plot(image_loader, model, save_path, greyscale, mask_ratio=0.75):
    true_images = []
    generated_images = []

    with torch.no_grad():
        for image, _ in image_loader:
            loss, pred, mask = model(image.float(), mask_ratio=mask_ratio)
            reconstructed = model.unpatchify(pred)

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
            reconstructed, _ = model(image.float())

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
    mnist_dataloader = load_mnist()
    cifar10_dataloader = load_cifar10()

    mae_mnist_model = load_mnist_mae()
    dae_mnist_model = load_mnist_dae()
    mae_cifar10_model = load_cifar10_mae()
    dae_cifar10_model = load_cifar10_dae()


    # generate_mae_psnr_plot(mnist_dataloader, mae_model, 'mae_psnr_mnist.png')
    generate_mae_psnr_plot(cifar10_dataloader, mae_cifar10_model, 'mae_psnr_cifar10.png')
    # generate_dae_psnr_plot(mnist_dataloader, dae_model, 'dae_psnr_mnist.png')
    generate_dae_psnr_plot(cifar10_dataloader, dae_cifar10_model, 'dae_psnr_cifar10.png')

    # generate_mae_ssim_plot(mnist_dataloader, mae_model, 'mae_ssim_mnist.png', greyscale=True)
    generate_mae_ssim_plot(cifar10_dataloader, mae_cifar10_model, 'mae_ssim_cifar10.png', greyscale=False)
    # generate_dae_ssim_plot(mnist_dataloader, dae_model, 'dae_ssim_mnist.png', greyscale=True)
    generate_dae_ssim_plot(cifar10_dataloader, dae_cifar10_model, 'dae_ssim_cifar10.png', greyscale=False)