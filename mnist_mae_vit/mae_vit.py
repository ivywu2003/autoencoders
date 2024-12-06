import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from mae.models_mae import MaskedAutoencoderViT

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class MaskedAutoencoderViTForMNIST(MaskedAutoencoderViT):
    def __init__(self, img_size=28, patch_size=4, in_chans=1, 
                 embed_dim=64, depth=4, num_heads=8,
                 decoder_embed_dim=32, decoder_depth=2, decoder_num_heads=8,
                 mlp_ratio=4.0, norm_layer=nn.LayerNorm, norm_pix_loss=False):
        self.patch_size = patch_size
        self.in_chans = in_chans
        super().__init__(img_size, patch_size, in_chans,
                         embed_dim, depth, num_heads,
                         decoder_embed_dim, decoder_depth, decoder_num_heads,
                         mlp_ratio, norm_layer, norm_pix_loss)


    def patchify(self, imgs):
        """
        imgs: (N, 1, H, W)
        x: (N, L, patch_size**2 *1)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 1, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2))
        return x
    
    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        # print("x shape", x.shape)
        p = self.patch_size
        h = w = int(x.shape[1]**.5)
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 1))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 1, h * p, h * p))
        return imgs
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
                x, attn_weights = blk(x, return_attention=True)
                attn_weights_all.append(attn_weights)
            else:
                x = blk(x)

        x = self.norm(x)
        if return_attention:
            return x, mask, ids_restore, attn_weights_all
        return x, mask, ids_restore
    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore, attn = self.forward_encoder(imgs, mask_ratio, return_attention = True)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask, attn