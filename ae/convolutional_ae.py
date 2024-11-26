import torch
import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from tqdm import tqdm
        
class Encoder(nn.Module):
    def __init__(self, latent_dim, normalize: bool = False):
        r'''
        latent_dim (int): Dimension of latent space
        normalize (bool): Whether to restrict the output latent onto the unit hypersphere
        '''
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 4, stride=2, padding=1) # 64x64 --> 32x32
        self.conv2 = nn.Conv2d(32, 32*2, 4, stride=2, padding=1) # 32x32 --> 16x16
        self.conv3 = nn.Conv2d(32*2, 32*4, 4, stride=2, padding=1) # 16x16 --> 8x8
        self.conv4 = nn.Conv2d(32*4, 32*8, 4, stride=2, padding=1) # 8x8 --> 4x4
        self.conv5 = nn.Conv2d(32*8, 32*16, 4, stride=2, padding=1) # 4x4 --> 2x2
        self.conv6 = nn.Conv2d(32*16, latent_dim, 4, stride=2, padding=1) # 2x2 --> 1x1
        self.fc = nn.Linear(latent_dim, latent_dim)

        self.nonlinearity = nn.ReLU()
        self.normalize = normalize

    def forward(self, x):
        x = self.nonlinearity(self.conv1(x))
        x = self.nonlinearity(self.conv2(x))
        x = self.nonlinearity(self.conv3(x))
        x = self.nonlinearity(self.conv4(x))
        x = self.nonlinearity(self.conv5(x))
        x = self.nonlinearity(self.conv6(x).flatten(1))
        x = self.fc(x)
        if self.normalize:
            x = nn.functional.normalize(x)
        return x

    def extra_repr(self):
        return f'normalize={self.normalize}'

class Decoder(nn.Module):
    def __init__(self, latent_dim):
        r'''
        latent_dim (int): Dimension of latent space
        '''
        super(Decoder, self).__init__()

        self.conv1 = nn.ConvTranspose2d(latent_dim, 32*16, 4, stride=2, padding=1) # 1x1 --> 2x2
        self.conv2 = nn.ConvTranspose2d(32*16, 32*8, 4, stride=2, padding=1) # 2x2 --> 4x4
        self.conv3 = nn.ConvTranspose2d(32*8, 32*4, 4, stride=2, padding=1) # 4x4 --> 8x8
        self.conv4 = nn.ConvTranspose2d(32*4, 32*2, 4, stride=2, padding=1) # 8x8 --> 16x16
        self.conv5 = nn.ConvTranspose2d(32*2, 32, 4, stride=2, padding=1) # 16x16 --> 32x32
        self.conv6 = nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1) # 32x32 --> 64x64
        self.nonlinearity = nn.ReLU()

    def forward(self, z):
        z = z[..., None, None]  # make it convolution-friendly

        x = self.nonlinearity(self.conv1(z))
        x = self.nonlinearity(self.conv2(x))
        x = self.nonlinearity(self.conv3(x))
        x = self.nonlinearity(self.conv4(x))
        x = self.nonlinearity(self.conv5(x))
        return self.conv6(x)

def train_autoencoder(latent_dim: int, dataset: Dataset, device: torch.device):
    r'''
    Train encoder and decoder networks with `latent_dim` latent dimensions according
    to the autoencoder objective (i.e., MSE reconstruction).

    Returns the trained encoder and decoder.
    '''
    enc = Encoder(latent_dim).to(device)
    dec = Decoder(latent_dim).to(device)

    optim = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=2e-4)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True, pin_memory=True)
    num_epochs = 30

    for epoch in tqdm(range(num_epochs), desc=f'{num_epochs} epochs total'):
        for batch, in dataloader:
            batch = batch.to(device)
            # batch: a batched image tensor of shape [B x 3 x 64 x 64]
            reconstruction = dec(enc(batch))
            loss = nn.MSELoss()(reconstruction, batch)

            optim.zero_grad()
            loss.backward()
            optim.step()
        print(f'[Autoencoder] epoch {epoch: 4d}   loss = {loss.item():.4g}')

    return enc, dec