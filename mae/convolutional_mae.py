import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class ConvolutionalEncoder(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()
        
        # Encoder architecture similar to the basic autoencoder but with more capacity
        self.conv1 = nn.Conv2d(3, 64, 4, stride=2, padding=1)  # 64x64 -> 32x32
        self.conv2 = nn.Conv2d(64, 128, 4, stride=2, padding=1)  # 32x32 -> 16x16
        self.conv3 = nn.Conv2d(128, 256, 4, stride=2, padding=1)  # 16x16 -> 8x8
        self.conv4 = nn.Conv2d(256, 512, 4, stride=2, padding=1)  # 8x8 -> 4x4
        self.conv5 = nn.Conv2d(512, latent_dim, 4, stride=2, padding=1)  # 4x4 -> 2x2
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)
        
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.activation(self.bn2(self.conv2(x)))
        x = self.activation(self.bn3(self.conv3(x)))
        x = self.activation(self.bn4(self.conv4(x)))
        x = self.conv5(x)
        return x

class ConvolutionalDecoder(nn.Module):
    def __init__(self, latent_dim=512):
        super().__init__()
        
        self.deconv1 = nn.ConvTranspose2d(latent_dim, 512, 4, stride=2, padding=1)  # 2x2 -> 4x4
        self.deconv2 = nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1)  # 4x4 -> 8x8
        self.deconv3 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)  # 8x8 -> 16x16
        self.deconv4 = nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1)  # 16x16 -> 32x32
        self.deconv5 = nn.ConvTranspose2d(64, 3, 4, stride=2, padding=1)  # 32x32 -> 64x64
        
        self.bn1 = nn.BatchNorm2d(512)
        self.bn2 = nn.BatchNorm2d(256)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(64)
        
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.activation(self.bn1(self.deconv1(x)))
        x = self.activation(self.bn2(self.deconv2(x)))
        x = self.activation(self.bn3(self.deconv3(x)))
        x = self.activation(self.bn4(self.deconv4(x)))
        x = torch.tanh(self.deconv5(x))  # Output in range [-1, 1]
        return x

class ConvolutionalMAE(nn.Module):
    def __init__(self, latent_dim=512, mask_ratio=0.75, patch_size=8):
        super().__init__()
        self.encoder = ConvolutionalEncoder(latent_dim)
        self.decoder = ConvolutionalDecoder(latent_dim)
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        
        # Learnable mask token
        self.mask_token = nn.Parameter(torch.randn(1, latent_dim, 1, 1))
        
    def patchify(self, imgs):
        """Convert images into patches"""
        B, C, H, W = imgs.shape
        p = self.patch_size
        assert H % p == 0 and W % p == 0
        
        # Reshape into patches
        patches = imgs.view(B, C, H//p, p, W//p, p)
        patches = patches.permute(0, 2, 4, 1, 3, 5).contiguous()
        patches = patches.view(B, (H//p)*(W//p), C*p*p)
        return patches

    def unpatchify(self, patches):
        """Convert patches back to images"""
        B, L, D = patches.shape
        p = self.patch_size
        h = w = int((L ** 0.5))
        C = 3
        
        patches = patches.view(B, h, w, C, p, p)
        patches = patches.permute(0, 3, 1, 4, 2, 5).contiguous()
        imgs = patches.view(B, C, h*p, w*p)
        return imgs

    def random_masking(self, x):
        """
        Randomly mask input patches. Unlike the transformer-based MAE,
        we mask at the feature map level.
        """
        B, C, H, W = x.shape
        num_patches = (H // 2) * (W // 2)  # Size after first conv layer
        num_mask = int(self.mask_ratio * num_patches)
        
        # Generate random indices for masking
        noise = torch.rand(B, num_patches, device=x.device)
        mask_indices = torch.argsort(noise, dim=1)[:, :num_mask]
        
        # Create binary mask
        mask = torch.ones((B, num_patches), device=x.device)
        mask.scatter_(1, mask_indices, 0)
        
        return mask.view(B, 1, H//2, W//2)  # Reshape for broadcasting

    def forward(self, x, mask_ratio=None):
        if mask_ratio is not None:
            self.mask_ratio = mask_ratio
            
        # Initial feature extraction
        features = self.encoder.conv1(x)
        features = self.encoder.bn1(features)
        features = self.encoder.activation(features)
        
        # Generate mask
        mask = self.random_masking(features)
        
        # Apply mask by replacing masked regions with learnable token
        B = features.shape[0]
        mask_tokens = self.mask_token.expand(B, -1, features.shape[2], features.shape[3])
        features = features * mask.expand_as(features) + mask_tokens * (1 - mask.expand_as(features))
        
        # Continue with encoding
        features = self.encoder.activation(self.encoder.bn2(self.encoder.conv2(features)))
        features = self.encoder.activation(self.encoder.bn3(self.encoder.conv3(features)))
        features = self.encoder.activation(self.encoder.bn4(self.encoder.conv4(features)))
        latent = self.encoder.conv5(features)
        
        # Decode
        reconstruction = self.decoder(latent)
        
        return reconstruction, mask

def train_mae(model, dataloader, num_epochs=100, device='cuda'):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-4)
    criterion = nn.MSELoss()
    
    model = model.to(device)
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            imgs = batch.to(device)
            
            # Forward pass
            reconstruction, mask = model(imgs)
            
            # Compute loss only on masked patches
            loss = criterion(reconstruction, imgs)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}')