import tqdm
from mae_vit import MaskedAutoencoderViTForMNIST

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# --------------------------
# Load and preprocess MNIST
# --------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# --------------------------
# Initialize model, optimizer, and loss
# --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = MaskedAutoencoderViTForMNIST(
    img_size=28, patch_size=2, in_chans=1, 
    embed_dim=64, depth=4, num_heads=4, 
    decoder_embed_dim=32, decoder_depth=2, decoder_num_heads=4
).to(device)

optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

# --------------------------
# Training loop
# --------------------------
def train_one_epoch(model, dataloader, optimizer, epoch, device, mask_ratio=0.75):
    model.train()
    total_loss = 0
    for images, _ in dataloader:
        images = images.to(device)

        # Forward pass
        loss, _, _, _ = model(images, mask_ratio=mask_ratio)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    avg_loss = total_loss / len(dataloader)
    # print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")
    return avg_loss

# --------------------------
# Train the model
# --------------------------
n_epochs = 20
for epoch in range(n_epochs):
    train_loss = train_one_epoch(model, train_loader, optimizer, epoch, device)
    print(f'Train loss: {train_loss:.4f}')
    with torch.no_grad():
        test_loss = 0.0
        for images, _ in test_loader:
            images = images.to(device)
            loss, _, _, _ = model(images)
            test_loss += loss.item()
    
    print(f'Test loss: {test_loss/len(test_loader):.4f}')
n_epochs = 20
torch.save(model.state_dict(), 'mae_weights.pth')  # Save MAE weights

