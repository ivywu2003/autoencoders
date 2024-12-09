import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
import math

from cifar10_models import CIFAR10Autoencoder, CustomCIFAR10MaskedAutoencoder, CIFAR10DenoisingAutoencoder

def train_autoencoder(model, dataloader, optimizer, criterion, device):
    model.train()
    train_loss = 0.0
    pbar = tqdm(dataloader, desc='Training AE')
    
    # Define normalization constants
    mean = torch.tensor([0.5]).to(device)
    std = torch.tensor([0.5]).to(device)
    
    for images, _ in pbar:
        images = images.to(device)
        
        # Forward pass
        outputs = model(images)
        
        # Denormalize both images and outputs for loss calculation
        images_denorm = images * std + mean
        outputs_denorm = outputs * std + mean
        
        # Backward pass
        optimizer.zero_grad()
        loss = criterion(outputs_denorm, images_denorm)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return train_loss / len(dataloader)

def train_masked_autoencoder(model, dataloader, device):
    model.train()
    train_loss = 0.0

    optim = torch.optim.AdamW(model.parameters(), lr=1e-4 * 64 / 256, betas=(0.9, 0.95), weight_decay=1e-5)
    lr_func = lambda epoch: min((epoch + 1) / (epoch / 10 + 1e-8), 0.5 * (math.cos(epoch / 10 * math.pi) + 1))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func, verbose=True)

    pbar = tqdm(dataloader, desc='Training MAE')
    for images, _ in pbar:
        images = images.to(device)

        # Forward pass
        predicted_img, mask = model(images)
        loss = torch.mean((predicted_img - images) ** 2 * mask) / 0.75

        # Backward pass
        optim.zero_grad()
        loss.backward()
        optim.step()
        
        train_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
    return train_loss / len(dataloader)

def train_denoising_autoencoder(model, dataloader, device, noise_factor=0.1):
    model.train()
    train_loss = 0.0

    # Use Adam with slightly lower learning rate and weight decay
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-5)
    criterion = nn.MSELoss()

    pbar = tqdm(dataloader, desc='Training DAE')
    for images, _ in pbar:
        # Move images to device first
        images = images.to(device)
        
        # Add noise to images
        noise = torch.randn_like(images) * noise_factor
        noisy_images = images + noise
        noisy_images = torch.clamp(noisy_images, 0., 1.)

        # Forward pass
        reconstructed = model(noisy_images)
        loss = criterion(reconstructed, images)  # Compare with clean images

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return train_loss / len(dataloader)

def get_data_loaders(batch_size=64, no_norm=False):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ])
    if no_norm:
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    trainset = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform)
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True)

    testset = datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform)
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False)
    
    return trainloader, testloader


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['ae', 'mae', 'dae'], default='ae')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--load_pretrained', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    if args.model == 'ae':
        model = CIFAR10Autoencoder().to(device)
        if args.load_pretrained:
            model.load_state_dict(torch.load(f'cifar10_ae_weights.pth'))
    elif args.model == 'mae':
        model = CustomCIFAR10MaskedAutoencoder().to(device)
        if args.load_pretrained:
            model.load_state_dict(torch.load(f'cifar10_mae_weights_10_epochs_custom.pth'))
    elif args.model == 'dae':
        model = CIFAR10DenoisingAutoencoder().to(device)
        if args.load_pretrained:
            model.load_state_dict(torch.load(f'cifar10_dae_weights_20_epochs.pth'))
    else:
        raise ValueError(f'Unknown model: {args.model}')

    # Reduce learning rate and add scheduler
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    trainloader, testloader = get_data_loaders()
    if args.model == 'dae':
        trainloader, testloader = get_data_loaders(no_norm=True)

    for epoch in range(args.epochs):
        print(f'Epoch {epoch+1}/{args.epochs}')
        if args.model == 'ae':
            train_loss = train_autoencoder(model, trainloader, optimizer, nn.MSELoss(), device)
        elif args.model == 'mae':
            train_loss = train_masked_autoencoder(model, trainloader, device)
        elif args.model == 'dae':
            train_loss = train_denoising_autoencoder(model, trainloader, device)
        print(f'Train loss: {train_loss:.4f}')
        with torch.no_grad():
            test_loss = 0.0
            pbar = tqdm(testloader, desc='Testing')
            for images, _ in pbar:
                images = images.to(device)
                if args.model == 'ae':
                    outputs = model(images)
                    test_loss += nn.MSELoss()(outputs, images).item()
                elif args.model == 'mae':
                    predicted_img, mask = model(images)
                    output = predicted_img * mask + images * (1 - mask)
                    loss = torch.mean((predicted_img - images) ** 2)
                    test_loss += loss.item()
                elif args.model == 'dae':
                    outputs = model(images)
                    test_loss += nn.MSELoss()(outputs, images).item()
                pbar.set_postfix({'loss': f'{test_loss/len(testloader):.4f}'})
        
        print(f'Test loss: {test_loss/len(testloader):.4f}')
    
    torch.save(model.state_dict(), f'cifar10_{args.model}_weights_20_epochs.pth')
