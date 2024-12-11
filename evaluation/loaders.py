import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from mnist_mae_vit.mae_vit import MaskedAutoencoderViTForMNIST
from mnist_mae_vit.dae import ConvDenoiser
from cifar10_task.cifar10_models import CIFAR10DenoisingAutoencoder, CustomCIFAR10MaskedAutoencoder
from color_jittering.color_jitter_mae import MaskedAutoencoderCIFAR10
from color_jittering.color_jitter_dae import DenoisingAutoencoderCIFAR10

def load_mnist_for_mae():
    """
    Returns train and test data loaders for MNIST dataset, with pixel values normalized to [-0.5, 0.5].

    Returns:
        trainloader: A DataLoader for training data.
        testloader: A DataLoader for testing data.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    trainset = torchvision.datasets.MNIST('./data', download=True, train=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    testset = torchvision.datasets.MNIST('./data', download=True, train=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)
    
    return trainloader, testloader

def load_mnist_for_dae():
    """
    Returns train and test data loaders for MNIST dataset, with pixel values normalized to [0, 1].

    Returns:
        trainloader: A DataLoader for training data.
        testloader: A DataLoader for testing data.
    """
    transform = transforms.ToTensor()

    trainset = torchvision.datasets.MNIST('./data', download=True, train=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    testset = torchvision.datasets.MNIST('./data', download=True, train=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)
    
    return trainloader, testloader

def load_cifar10_for_mae():
    """
    Returns train and test data loaders for CIFAR-10 dataset, with pixel values normalized to [-0.5, 0.5].

    Returns:
        trainloader: A DataLoader for training data.
        testloader: A DataLoader for testing data.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=64, shuffle=True
    )
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=64, shuffle=False)
    
    return trainloader, testloader

def load_cifar10_for_dae():
    """
    Returns train and test data loaders for CIFAR-10 dataset, with pixel values normalized to [0, 1].

    Returns:
        trainloader: A DataLoader for training data.
        testloader: A DataLoader for testing data.
    """
    transform = transforms.ToTensor()

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=64, shuffle=True
    )
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=64, shuffle=False)
    
    return trainloader,testloader

def load_jittered_cifar10_for_mae():
    """
    Returns train and test data loaders for CIFAR-10 dataset, with pixel values normalized to [0, 1]
    and randomly jittered by brightness (0.5), contrast (0.5), saturation (0.5), and hue (0.1).

    Returns:
        trainloader: A DataLoader for training data.
        testloader: A DataLoader for testing data.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=64, shuffle=True
    )
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=64, shuffle=False)

    return trainloader, testloader

def load_jittered_cifar10_for_dae():
    """
    Returns train and test data loaders for CIFAR-10 dataset, with pixel values normalized to [-0.5, 0.5]
    and randomly jittered by brightness (0.5), contrast (0.5), saturation (0.5), and hue (0.1).

    Returns:
        trainloader: A DataLoader for training data.
        testloader: A DataLoader for testing data.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1),
        # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=64, shuffle=True
    )
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=64, shuffle=False)

    return trainloader, testloader


def load_mnist_mae():
    mae_model = MaskedAutoencoderViTForMNIST(
        img_size=28, patch_size=2, in_chans=1, 
        embed_dim=64, depth=4, num_heads=4, 
        decoder_embed_dim=32, decoder_depth=2, decoder_num_heads=4
    )
    mae_model.load_state_dict(torch.load("mnist_mae_vit/mae_weights_vit_patch2_20epochs.pth", weights_only=True))
    mae_model.eval()
    return mae_model

def load_mnist_dae():
    dae_model = ConvDenoiser()
    dae_model.load_state_dict(torch.load('mnist_mae_vit/dae_weights.pth', weights_only=True)) 
    dae_model.eval()
    return dae_model

def load_ivy_cifar10_mae():
    mae_model = CustomCIFAR10MaskedAutoencoder()
    mae_model.load_state_dict(torch.load("cifar10_task/cifar10_mae_weights_20_epochs_custom.pth", weights_only=True, map_location=torch.device('cpu')))
    mae_model.eval()
    return mae_model

def load_ivy_cifar10_dae():
    dae_model = CIFAR10DenoisingAutoencoder()
    dae_model.load_state_dict(torch.load('cifar10_task/cifar10_dae_weights_20_epochs.pth', weights_only=True, map_location=torch.device('cpu'))) 
    dae_model.eval()
    return dae_model

def load_cifar10_mae():
    mae_model = MaskedAutoencoderCIFAR10()
    mae_model.load_state_dict(torch.load("cifar10_alt/mae_weights_20_epochs.pth", weights_only=True, map_location=torch.device('cpu')))
    mae_model.eval()
    return mae_model

def load_cifar10_dae():
    dae_model = DenoisingAutoencoderCIFAR10()
    dae_model.load_state_dict(torch.load('cifar10_alt/dae_weights_20_epochs.pth', weights_only=True, map_location=torch.device('cpu'))) 
    dae_model.eval()
    return dae_model

def load_jittered_mae():
    mae_model = MaskedAutoencoderCIFAR10(img_size=32, patch_size=2, in_chans=3, embed_dim=128, depth=12, num_heads=4, decoder_embed_dim=64, decoder_depth=4, decoder_num_heads=4, mlp_ratio=2, norm_layer=nn.LayerNorm, norm_pix_loss=False)
    mae_model.load_state_dict(torch.load("color_jittering/color_jitter_mae_weights_epoch_20.pth", weights_only=True, map_location=torch.device('cpu')))
    mae_model.eval()
    return mae_model

def load_jittered_dae():
    dae_model = DenoisingAutoencoderCIFAR10()
    dae_model.load_state_dict(torch.load('color_jittering/color_jitter_dae_weights_epoch_20.pth', weights_only=True, map_location=torch.device('cpu'))) 
    dae_model.eval()
    return dae_model
