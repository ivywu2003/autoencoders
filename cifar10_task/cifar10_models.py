import sys
sys.path.insert(1, '../mae')

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

from models_mae import MaskedAutoencoderViT

class CIFAR10Autoencoder(nn.Module):
    def __init__(self, latent_dim=64, norm_pix_loss=False):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),  # 32x32 -> 16x16
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 16x16 -> 8x8
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  # 8x8 -> 4x4
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, latent_dim, 4, stride=1, padding=0),  # 4x4 -> 1x1
            nn.BatchNorm2d(latent_dim)
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 128, 4, stride=1, padding=0),  # 1x1 -> 4x4
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),  # 4x4 -> 8x8
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),  # 8x8 -> 16x16
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),  # 16x16 -> 32x32
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class CIFAR10MaskedAutoencoder(MaskedAutoencoderViT):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=64, depth=4, num_heads=8, decoder_embed_dim=32, decoder_depth=2, decoder_num_heads=8, mlp_ratio=4.0, norm_layer=nn.LayerNorm, norm_pix_loss=False):
        self.patch_size = patch_size
        MaskedAutoencoderViT.__init__(self, img_size=img_size, patch_size=patch_size, in_chans=in_chans, 
                                     embed_dim=embed_dim, depth=depth, num_heads=num_heads, 
                                     decoder_embed_dim=decoder_embed_dim, decoder_depth=decoder_depth, 
                                     decoder_num_heads=decoder_num_heads, mlp_ratio=mlp_ratio, 
                                     norm_layer=norm_layer, norm_pix_loss=norm_pix_loss)

    def forward_encoder(self, x, mask_ratio, return_attention=False):
        x = self.patch_embed(x)
        x = x + self.pos_embed[:, 1:, :]
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        attn_weights_all = []
        for blk in self.blocks:
            if return_attention:
                x, attn = blk(x, return_attention=True)
                attn_weights_all.append(attn)
            else:
                x = blk(x, return_attention=False)

        x = self.norm(x)

        if return_attention:
            return x, mask, ids_restore, attn_weights_all
        return x, mask, ids_restore

    def forward(self, imgs, mask_ratio=0.75, return_latent=False, return_attention=False):
        if return_attention:
            latent, mask, ids_restore, attn_weights_all = self.forward_encoder(imgs, mask_ratio, True)
            pred = self.forward_decoder(latent, ids_restore)
            loss = self.forward_loss(imgs, pred, mask)
            if return_latent:
                return latent, loss, pred, mask, attn_weights_all
            return loss, pred, mask, attn_weights_all
        else:
            latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio, False)
            pred = self.forward_decoder(latent, ids_restore)
            loss = self.forward_loss(imgs, pred, mask)
            if return_latent:
                return latent, loss, pred, mask
            return loss, pred, mask