# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------
import numpy as np
import pandas as pd
import numpy as np
from functools import partial

import torch
import torch.nn as nn
from timm.models.vision_transformer import Block
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin



class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding

    flux capactitor - allows us to trace mask
    """

    def __init__(
            self,
            img_size=[400, 160],
            patch_size=[400,1],
            in_chans=1,
            embed_dim=768,
            norm_layer=None,
            flatten=True,
            bias=True,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (self.img_size[0] // self.patch_size[0], self.img_size[1] // self.patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0], f"Input image height ({H}) doesn't match model ({self.img_size[0]})."
        assert W == self.img_size[1], f"Input image width ({W}) doesn't match model ({self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class CustomHead(nn.Module):

    def __init__(self, head_embed_dim, img_channels, patch_shape, patchify_fn, unpatchify_fn, num_classes):
        super().__init__()

        self.head_embed_dim = head_embed_dim
        self.img_channels = img_channels
        self.patch_shape = patch_shape
        self.patchify_fn = patchify_fn
        self.unpatchify_fn = unpatchify_fn
        self.num_classes = num_classes

        self.conv2d_embed = nn.Linear(self.head_embed_dim, self.patch_shape[0]*self.patch_shape[1], bias=True)

        self.conv2d_1 = torch.nn.Conv2d(self.img_channels, 256, kernel_size=7, padding=3,  bias=False)
        self.conv2d_1_rl = torch.nn.LeakyReLU()
        self.conv2d_1_bn = torch.nn.BatchNorm2d(256)

        self.conv2d_2 = torch.nn.Conv2d(256, 128, kernel_size=5, padding=2, bias=False)
        self.conv2d_2_rl = torch.nn.LeakyReLU()
        self.conv2d_2_bn = torch.nn.BatchNorm2d(128)

        self.conv2d_3 = torch.nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False)
        self.conv2d_3_rl = torch.nn.LeakyReLU()
        self.conv2d_3_bn = torch.nn.BatchNorm2d(64)

        #for semantic segmentation use self.num_classes, otherwise use img_channels
        self.conv2d_output = torch.nn.Conv2d(64, self.num_classes, kernel_size=1, padding=0, bias=False)

    def forward(self, x):

        x = self.conv2d_embed(x)
        x = x[:, 1:, :]  # remove cls token

        x = self.unpatchify_fn(x)

        x = self.conv2d_1(x)
        x = self.conv2d_1_rl(x)
        x = self.conv2d_1_bn(x)

        x = self.conv2d_2(x)
        x = self.conv2d_2_rl(x)
        x = self.conv2d_2_bn(x)

        x = self.conv2d_3(x)
        x = self.conv2d_3_rl(x)
        x = self.conv2d_3_bn(x)

        x = self.conv2d_output(x)

        #apply softmax to get probabilities of classes
        #x = F.softmax(x, dim=1)

        # (BCWH -> BWHC) move channels to last

        #apply torch.argmax() to channels dim
        #x = torch.argmax(x, dim=-1)

        return x


class ElasticViTMAE(nn.Module, PyTorchModelHubMixin):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=[400, 160], patch_size=[400, 1], in_chans=1,
                 embed_dim=1200, depth=16, num_heads=20,
                 decoder_embed_dim=800, decoder_depth=12, decoder_num_heads=20,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False,
                 custom_head=False, full_image_loss=True, classes=10):
        super().__init__()

        self.in_chans = in_chans
        self.custom_head = custom_head
        self.full_image_loss = full_image_loss
        self.classes = classes
        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)# qk_scale=False,
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer) # qk_scale=False,
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)

        if self.custom_head:
            self.decoder_pred = CustomHead(head_embed_dim=decoder_embed_dim,
                                           img_channels=self.in_chans,
                                           patch_shape=self.patch_embed.patch_size,
                                           patchify_fn=self.patchify,
                                           unpatchify_fn=self.unpatchify,
                                          num_classes = self.classes) # custom decoder to patch
        else:
            self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size[0] * patch_size[1] * in_chans, bias=True) # decoder to patch
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        # pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], self.patch_embed.grid_size, cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], self.patch_embed.grid_size, cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """

        p_h = self.patch_embed.patch_size[0]
        p_w = self.patch_embed.patch_size[1]

        h = imgs.shape[2] // p_h
        w = imgs.shape[3] // p_w

        # x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = imgs.reshape(shape=(imgs.shape[0], self.in_chans, h, p_h, w, p_w))  # one channel
        x = torch.einsum('nchpwq->nhwpqc', x)
        # x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3)) # one channel
        x = x.reshape(shape=(imgs.shape[0], h * w, p_h * p_w * self.in_chans))

        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """

        p_h = self.patch_embed.patch_size[0]
        p_w = self.patch_embed.patch_size[1]

        h = self.patch_embed.grid_size[0]
        w = self.patch_embed.grid_size[1]

        # x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = x.reshape(shape=(x.shape[0], h, w, p_h, p_w, self.in_chans))  # one channel
        x = torch.einsum('nhwpqc->nchpwq', x)
        # imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        imgs = x.reshape(shape=(x.shape[0], self.in_chans, h * p_h, w * p_w))  # one channel

        return imgs

    def random_masking(self, x, patch_idx, len_keep):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """

        N, L, D = x.shape  # batch, length, dim
        #len_keep = len_keep.item()

        # sort noise for each sample
        ids_shuffle = torch.argsort(patch_idx, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, idx_shuffle, len_keep):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, idx_shuffle, len_keep)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        if not self.custom_head:
            # remove cls token
            x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        if self.full_image_loss:
            loss = (pred - target) ** 2

            batch_size = mask.shape[0]
            num_patches = mask.shape[1]
            loss = loss.sum() / batch_size / num_patches # loss on all patches
        else:
            m = mask.unsqueeze(-1)
            p = pred * m
            t = target * m
            loss = (p - t) ** 2

            batch_size = mask.shape[0]
            num_masked = mask[0].sum().item()

            loss = loss.sum() / batch_size / num_masked  # loss on mask

        return loss

    def forward_loss_pixel_accuracy(self, pred, label):
        out = torch.where(label==pred, 1, 0)
        numerator = sum(out.flatten())
        denominator = len(out.flatten())
        accuracy = round((numerator/denominator),5)
        loss = 1 - accuracy
        return loss

    def forward(self, imgs, idx_shuffle, len_keep):
        latent, mask, ids_restore = self.forward_encoder(imgs, idx_shuffle, len_keep)
        pred = self.forward_decoder(latent, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask


def build_2d_sincos_position_embedding(self, temperature=10000.):
        h, w = self.patch_embed.grid_size
        grid_w = torch.arange(w, dtype=torch.float32)
        grid_h = torch.arange(h, dtype=torch.float32)
        grid_w, grid_h = torch.meshgrid(grid_w, grid_h)
        assert self.embed_dim % 4 == 0, 'Embed dimension must be divisible by 4 for 2D sin-cos position embedding'
        pos_dim = self.embed_dim // 4
        omega = torch.arange(pos_dim, dtype=torch.float32) / pos_dim
        omega = 1. / (temperature**omega)
        out_w = torch.einsum('m,d->md', [grid_w.flatten(), omega])
        out_h = torch.einsum('m,d->md', [grid_h.flatten(), omega])
        pos_emb = torch.cat([torch.sin(out_w), torch.cos(out_w), torch.sin(out_h), torch.cos(out_h)], dim=1)[None, :, :]

        assert self.num_tokens == 1, 'Assuming one and only one token, [cls]'
        pe_token = torch.zeros([1, 1, self.embed_dim], dtype=torch.float32)
        self.pos_embed = nn.Parameter(torch.cat([pe_token, pos_emb], dim=1))
        self.pos_embed.requires_grad = False


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size[0], dtype=np.float32)
    grid_w = np.arange(grid_size[1], dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)
    grid = grid.reshape([2, 1, grid_size[0], grid_size[1]])

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)

    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)

    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)

    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)

    return emb
