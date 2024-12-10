import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.optim as optim
from tqdm import tqdm
from jitter_images import load_mnist_data, load_cifar10_data
from color_jitter_mae import MaskedAutoencoderCIFAR10

def train(mae, device, train_loader, optimizer, epoch):
    mae.train()
    total_loss = 0

    pbar = tqdm(train_loader, desc='Training MAE')
    for batch_idx, (images, _) in enumerate(pbar):
        # data, mask = data.to(device), mask.to(device)
        optimizer.zero_grad()
        loss, pred, mask = mae(images)
        loss = (loss ** 2).mean()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    print(f'Epoch {epoch} loss: {total_loss / len(train_loader)}')


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = load_cifar10_data()

    model = MaskedAutoencoderCIFAR10()
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1.5e-4)

    for epoch in range(10):
        train(model, device, train_loader, optimizer, epoch)

if __name__ == '__main__':
    main()
