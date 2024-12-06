import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from tqdm import tqdm

from cifar10_models import CIFAR10Autoencoder, CIFAR10MaskedAutoencoder

def train_autoencoder(model, dataloader, optimizer, criterion, device):
    model.train()
    train_loss = 0.0
    pbar = tqdm(dataloader, desc='Training AE')
    
    # Define normalization constants
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.2023, 0.1994, 0.2010]).view(1, 3, 1, 1).to(device)
    
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

def train_masked_autoencoder(model, dataloader, optimizer, device):
    model.train()
    train_loss = 0.0
    pbar = tqdm(dataloader, desc='Training MAE')
    for images, _ in pbar:
        images = images.to(device)

        # Forward pass
        loss, _, _ = model(images)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
    return train_loss / len(dataloader)

def get_data_loaders(batch_size=64):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                           (0.2023, 0.1994, 0.2010))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                           (0.2023, 0.1994, 0.2010))
    ])

    trainset = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True)

    testset = datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False)
    
    return trainloader, testloader

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['ae', 'mae'], default='ae')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--load_pretrained', action=argparse.BooleanOptionalAction)
    args = parser.parse_args()

    if args.model == 'ae':
        model = CIFAR10Autoencoder().to(device)
        if args.load_pretrained:
            model.load_state_dict(torch.load(f'cifar10_ae_weights.pth'))
    elif args.model == 'mae':
        model = CIFAR10MaskedAutoencoder().to(device)
        if args.load_pretrained:
            model.load_state_dict(torch.load(f'cifar10_mae_weights_2.pth'))
    else:
        raise ValueError(f'Unknown model: {args.model}')

    # Reduce learning rate and add scheduler
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    trainloader, testloader = get_data_loaders()

    for epoch in range(args.epochs):
        print(f'Epoch {epoch+1}/{args.epochs}')
        if args.model == 'ae':
            train_loss = train_autoencoder(model, trainloader, optimizer, nn.MSELoss(), device)
        elif args.model == 'mae':
            train_loss = train_masked_autoencoder(model, trainloader, optimizer, device)
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
                    loss, _, _ = model(images)
                    test_loss += loss.item()
                pbar.set_postfix({'loss': f'{test_loss/len(testloader):.4f}'})
        
        print(f'Test loss: {test_loss/len(testloader):.4f}')
        scheduler.step(test_loss / len(testloader))
    
    torch.save(model.state_dict(), f'cifar10_{args.model}_weights_30_epochs.pth')
