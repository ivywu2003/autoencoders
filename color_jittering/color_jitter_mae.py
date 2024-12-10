import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from mae.models_mae import MaskedAutoencoderViT

import torch.nn as nn

class MaskedAutoencoderCIFAR10(MaskedAutoencoderViT):
    def __init__(self, img_size=32, patch_size=2, in_chans=3, embed_dim=192, depth=12, num_heads=3, decoder_embed_dim=192, decoder_depth=4, decoder_num_heads=3, mlp_ratio=4, norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__(img_size, patch_size, in_chans, embed_dim, depth, num_heads, decoder_embed_dim, decoder_depth, decoder_num_heads, mlp_ratio, norm_layer, norm_pix_loss)