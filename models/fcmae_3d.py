# https://github.com/facebookresearch/ConvNeXt-V2
import numpy as np

import torch
import torch.nn as nn

from .convnextv2_3d import ConvNeXtV2_3D
from .convnextv2_3d_sparse import SparseConvNeXtV2_3D
from .convnextv2_3d import Block

from timm.models.layers import trunc_normal_

class FCMAE_3D(nn.Module):
    ''' Fully Convolutional Masked Autoencoder with ConvNeXtV2 3D backbone
    '''
    def __init__(
        self,
        size=128,
        in_chans=1,
        depths=[3, 3, 9, 3],
        dims=[96, 192, 384, 768],
        kernel_size=7,
        padding=3,
        downsampling='avgpool3d', # 'conv3d', 'avgpool3d', 'maxpool3d
        decoder_depth=1,
        decoder_embed_dim=512,
        patch_size=32,
        mask_ratio=0.6,
        norm_pix_loss=False,
        encoder_type='convnextv2'
    ):
        super().__init__()
        
        # configs
        self.img_size = size
        self.depths = depths
        self.imds = dims
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        self.num_patches = (size // patch_size) ** 3
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_depth = decoder_depth
        self.norm_pix_loss = norm_pix_loss
        
        if encoder_type == 'convnextv2':
            self.encoder = ConvNeXtV2_3D(
                in_chans=in_chans, 
                depths=depths, 
                dims=dims,
                kernel_size=kernel_size,
                padding=padding,
                downsampling=downsampling
            )
        elif encoder_type == 'sparse_convnextv2':
            self.encoder = SparseConvNeXtV2_3D(
                in_chans=in_chans,
                depths=depths,
                dims=dims,
                kernel_size=kernel_size,
                padding=padding,
                downsampling=downsampling
            )
        else:
            raise ValueError(f'Unknown encoder type: {encoder_type}')
        
        # decoder
        self.proj = nn.Conv3d(
            in_channels=dims[-1], 
            out_channels=decoder_embed_dim, 
            kernel_size=1)
        # mask tokens
        self.mask_token = nn.Parameter(torch.zeros(1, decoder_embed_dim, 1, 1, 1))
        decoder = [Block(
            dim=decoder_embed_dim, 
            drop_path=0., kernel_size=kernel_size, padding=padding) for _ in range(decoder_depth)]
        self.decoder = nn.Sequential(*decoder)
        # pred
        self.pred = nn.Conv3d(
            in_channels=decoder_embed_dim,
            out_channels=patch_size**3 * in_chans,
            kernel_size=1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv3d):
            w = m.weight.data
            trunc_normal_(w.view([w.shape[0], -1]))
            nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        if hasattr(self, 'mask_token'):    
            torch.nn.init.normal_(self.mask_token, std=.02)
            
    def upsample_mask(self, mask, upsample_size):
        assert len(mask.shape)==2
        p = int(round(mask.shape[1] ** (1/3)))
        scale = upsample_size // p
        return mask.reshape(-1, p, p, p).\
                repeat_interleave(scale, axis=1).\
                repeat_interleave(scale, axis=2).\
                repeat_interleave(scale, axis=3)
    
    def patchify3D(self, imgs):
        """
        imgs: (N, 1, H, W, D)
        x: (N, L, patch_size**3 *1)
        """
        p = self.patch_size
        assert imgs.shape[2] % p == 0 and imgs.shape[3] % p == 0 and imgs.shape[4] % p == 0

        h = w = d = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], imgs.shape[1], h, p, w, p, d, p))
        x = torch.einsum('nchpwqdk->nhwdpqkc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w * d, p**3 * imgs.shape[1]))
        return x

    def unpatchify3D(self, x):
        """
        x: (N, L, patch_size**3 *1)
        imgs: (N, 1, H, W, D)
        """
        p = self.patch_size
        h = w = d = int(np.cbrt(x.shape[1]))
        assert h * w * d == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, d, p, p, p, 1))
        x = torch.einsum('nhwdpqkc->nchpwqdk', x)
        imgs = x.reshape(shape=(x.shape[0], 1, h * p, h * p, h * p))
        return imgs

    def gen_random_mask(self, x, mask_ratio):
        N = x.shape[0]
        L = (x.shape[2] // self.patch_size) ** 3
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.randn(N, L, device=x.device)

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # generate the binary mask: 0 is keep 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return mask
            
    def forward_encoder(self, imgs, mask_ratio):
        # generate random masks
        mask = self.gen_random_mask(imgs, mask_ratio)
        # encoding
        x = self.encoder(imgs, mask)
        return x, mask
    
    def forward_decoder(self, x, mask):
        x = self.proj(x)
        # append mask token
        n, c, h, w, d = x.shape
        mask = mask.reshape(-1, h, w, d).unsqueeze(1).type_as(x)
        mask_token = self.mask_token.repeat(x.shape[0], 1, x.shape[2], x.shape[3], x.shape[4])
        x = x * (1. - mask) + mask_token * mask
        # decoding
        x = self.decoder(x)
        # pred
        pred = self.pred(x)
        return pred
    
    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 1, H, W, D]
        pred: [N, L, p*p*p*1]
        mask: [N, L], 0 is keep, 1 is remove
        """
        if len(pred.shape) == 5:
            n, c, _, _, _= pred.shape
            pred = pred.reshape(n, c, -1)
            pred = torch.einsum('ncl->nlc', pred)
        
        target = self.patchify3D(imgs)        
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5
            
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, labels=None, mask_ratio=0.6):
        x, mask = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(x, mask)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask