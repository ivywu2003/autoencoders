import torch.nn as nn
import torch.nn.functional as F

class ConvDenoiser(nn.Module):
    def __init__(self):
        super(ConvDenoiser, self).__init__()

        self.encoder = nn.Sequential(
            # Encoder layers
            nn.Conv2d(1, 32, 3, padding=1),  # Conv layer 1 (1 --> 32 channels)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 16, 3, padding=1),  # Conv layer 2 (32 --> 16 channels)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 8, 3, padding=1),   # Conv layer 3 (16 --> 8 channels)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.decoder = nn.Sequential(
            # Decoder layers
            nn.ConvTranspose2d(8, 8, 3, stride=2),  # ConvTranspose layer 1 (upscale)
            nn.ReLU(),
            nn.ConvTranspose2d(8, 16, 2, stride=2), # ConvTranspose layer 2 (upscale)
            nn.ReLU(),
            nn.ConvTranspose2d(16, 32, 2, stride=2),# ConvTranspose layer 3 (upscale)
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, padding=1),  # Final Conv layer to reduce depth to 1
            nn.Sigmoid()  # Sigmoid activation for output
        )

    def forward(self, x):
        latent = self.encoder(x)
        res = self.decoder(latent)
        return res, latent
dae_model = ConvDenoiser()