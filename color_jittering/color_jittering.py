import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from jitter_images import load_mnist_data, load_cifar10_data
from color_jitter_mae import MaskedAutoencoderCIFAR10
from color_jitter_dae import CIFAR10DenoisingAutoencoder

def train_mae_helper(mae, device, train_loader, optimizer, epoch):
    mae.train()
    total_loss = 0

    pbar = tqdm(train_loader, desc='Training MAE')
    for batch_idx, (images, _) in enumerate(pbar):
        # data, mask = data.to(device), mask.to(device)
        optimizer.zero_grad()
        loss, pred, mask = mae(images)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    print(f'Epoch {epoch} loss: {total_loss / len(train_loader)}')

def train_mae():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = load_cifar10_data()

    model = MaskedAutoencoderCIFAR10()
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    for epoch in range(5):
        train_mae_helper(model, device, train_loader, optimizer, epoch)

    torch.save(model.state_dict(), 'color_jittering/color_jitter_mae_weights.pth')


def train_dae_helper(dae, device, train_loader, optimizer, epoch):
    dae.train()
    criterion = nn.MSELoss()
    total_loss = 0

    pbar = tqdm(train_loader, desc='Training DAE')
    for batch_idx, (images, _) in enumerate(pbar):
        noise = torch.randn_like(images) * 0.1
        noisy_images = images + noise
        # noisy_images = torch.clamp(noisy_images, 0., 1.)
        
        decoded = dae(noisy_images)
        loss = criterion(decoded, images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    print(f'Epoch {epoch} loss: {total_loss / len(train_loader)}')

def train_dae():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = load_cifar10_data()

    model = CIFAR10DenoisingAutoencoder()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)

    for epoch in range(30):
        train_dae_helper(model, device, train_loader, optimizer, epoch)
        if (epoch+1) % 5 == 0:
            torch.save(model.state_dict(), f'color_jittering/color_jitter_dae_weights_epoch_{epoch}.pth')

if __name__ == '__main__':
    train_dae()
