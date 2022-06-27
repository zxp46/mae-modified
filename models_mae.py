# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

from timm.models.vision_transformer import Block, to_2tuple

from util.pos_embed import get_2d_sincos_pos_embed


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, sep=2):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * \
            (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.smaller_num_patches = num_patches // sep * (sep - 1)

        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


""" 

class GridPadding(object):
    def __init__(self, sep=None, window_size=None, masked_height=None, masked_width=None, device=None):
        self.sep = sep
        self.masked_height = masked_height
        self.masked_width = masked_width
        self.window_size = window_size
        self.unmask = None
        self.device = device

    def __call__(self, image):
        B, C, H, W = image.shape

        mat_size = (H // self.window_size, W // self.window_size)
        x = torch.ones(mat_size, dtype=image.dtype).to(image.device)
        img_size = (H - self.masked_height, W - self.masked_width)
        tmp = torch.zeros(B, C, H - self.masked_height, W - self.masked_width).to(image.device)
        for i in range(B):
            tmp[i] = torchvision.transforms.Resize(img_size)(image[i])
            tmp[i] = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(tmp[i])
        image = tmp
        x[::self.sep] = 0
        x[:, ::self.sep] = 0
        self.mask = torch.nonzero(x.view(-1)).t().squeeze()
        repeat_idx = torch.arange(x.shape[0]).repeat_interleave(self.window_size)
        x = x.flatten().repeat_interleave(self.window_size)
        x = x.view(H // self.window_size, W)
        x = x[repeat_idx]
        x = torch.stack(C * [x], dim=2)
        x = torch.stack(B * [x], dim=3)
        x = x.permute(3, 2, 0, 1).reshape(B*C, -1)
        x[x == 1] = image.view(-1)
        x = x.view(B, C, H, W)

        return x, self.mask

    def get_mask(self):
        return self.unmask

    def __repr__(self):
        return self.__class__.__name__ + '()'

"""


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False, global_pool=False, use_linear=True):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.img_size = img_size
        self.patch_size = patch_size
        self.patch_embed = PatchEmbed(
            img_size, patch_size, in_chans, embed_dim)

        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim),
                                      requires_grad=False)  # fixed sin-cos embedding
        # end modify#

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True,
                  qk_scale=None, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # dim = 2048
        # pred_dim = 512
        # self.projector = nn.Sequential(nn.BatchNorm1d(decoder_embed_dim),
        #                                nn.ReLU(inplace=True),  # first layer
        #                                nn.Linear(decoder_embed_dim,
        #                                          decoder_embed_dim, bias=False),
        #                                nn.BatchNorm1d(decoder_embed_dim),
        #                                nn.ReLU(inplace=True),  # second layer
        #                                nn.Linear(decoder_embed_dim,
        #                                          dim, bias=False),
        #                                nn.BatchNorm1d(dim, affine=False))  # output layer

        # self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
        #                                nn.BatchNorm1d(pred_dim),
        #                                nn.ReLU(inplace=True),  # hidden layer
        #                                nn.Linear(pred_dim, dim))  # output layer

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim),
                                              requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio,
                  qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, patch_size ** 2 * in_chans, bias=True)  # decoder to patch

        self.pool_embed = nn.AvgPool2d(2, stride=2)

        """
        self.global_pool = global_pool

        if global_pool:
            self.decoder_head = nn.Linear(decoder_embed_dim, decoder_embed_dim, bias=True)
        else:
            self.decoder_head = nn.Linear(decoder_embed_dim, embed_dim, bias=True)
        self.batch_norm = nn.BatchNorm1d(decoder_embed_dim, affine=False, eps=1e-6)
        """
        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss
        self.log_vars = nn.Parameter(torch.zeros(2))

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
                                            cls_token=True)
        self.pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
                                                    int(self.patch_embed.num_patches ** .5), cls_token=True)
        self.decoder_pos_embed.data.copy_(
            torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

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
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        # ascend: small is keep, large is remove
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio, pad):
        # embed patches
        x = self.patch_embed(x)
        # add pos embed w/o cls token
        if mask_ratio > 0:
            x = x + self.pos_embed[:, 1:, :]
        else:
            B, N, C = self.pos_embed.shape
            len = int(self.patch_embed.num_patches ** .5)
            smaller_pos_embed = self.pos_embed[:, 1:, :].reshape(
                (B, len, -1, C))
            smaller_pos_embed = self.pool_embed(smaller_pos_embed)
            smaller_pos_embed = smaller_pos_embed.reshape((B, -1, C))
            x = x + smaller_pos_embed

        mask, ids_restore = None, None

        # masking: length -> length * mask_ratio
        if mask_ratio > 0:
            x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        cls = x[:, 0, :].squeeze()

        return x, mask, ids_restore, cls

    def forward_decoder(self, x, ids_restore, need_mask=True, global_pool=False):
        # embed tokens
        cls = None
        p = None
        x = self.decoder_embed(x)
        if not need_mask:
            x = self.decoder_norm(x)
            cls = x[:, 0, :].squeeze()
            return x, cls, p

        # append mask tokens to sequence
        if need_mask:
            mask_tokens = self.mask_token.repeat(
                x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
            x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
            x_ = torch.gather(
                x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
            x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

            # add pos embed
            x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for i, blk in enumerate(self.decoder_blocks):
            if torch.isnan(x).any():
                print("break at block", i)
            x = blk(x)
            """
            if i == 0:
                if not global_pool:
                    cls = x[:, 0, :].squeeze()
                    cls = self.decoder_head(cls)
                    if not need_mask:
                        return x, cls, p
                else:
                    cls = x[:, 1:, :].mean(dim=1)  # global pool without cls token
                    cls = self.decoder_norm(cls)
                    cls = self.batch_norm(cls)
            elif i == 1:
                if global_pool:
                    p = x[:, 1:, :].mean(dim=1)  # global pool without cls token
                    p = self.decoder_norm(p)
                    p = self.batch_norm(p)
                    p = self.decoder_head(p)
                    if not need_mask:
                        return x, cls, p
            """
        x = self.decoder_norm(x)
        cls = x[:, 0, :].squeeze()

        # predictor projection
        x = self.decoder_pred(x)

        # remove cls token
        x = x[:, 1:, :]

        return x, cls, p

    def forward_linears(self, x1, x2):
        z1 = self.projector(x1)  # NxC
        z2 = self.projector(x2)  # NxC

        p1 = self.predictor(z1)  # NxC
        p2 = self.predictor(z2)  # NxC

        return p1, p2, z1.detach(), z2.detach()

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
            target = (target - mean) / (var + 1.e-6) ** .5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    # pzx modified #
    def gram(self, x):
        b, hw, c = x.shape
        feature = x.view(b, c, hw)
        g = torch.bmm(feature, feature.transpose(1, 2))
        return g.div(hw)

    def compute_similarity(self, features, targets):
        mse_criterion = torch.nn.MSELoss(reduction='mean')
        gram_loss = mse_criterion(self.gram(features), self.gram(targets))
        return gram_loss

    def twin_loss(self, z, z_, p, p_):
        criterion = nn.CosineSimilarity(dim=1)
        return -(criterion(p, z_).mean() + criterion(p_, z).mean()) * 0.5

    def kl_loss(self, small_cls, mask_cls, student_temp=0.1, teacher_temp=0.04):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        total_loss = self.H(small_cls, mask_cls, student_temp, teacher_temp) / \
            2 + self.H(mask_cls, small_cls, student_temp, teacher_temp)/2
        return total_loss

    def H(self, s, t,  temps, tempt):
        t = t.detach()
        s = F.softmax(s/temps, dim=1).unsqueeze(dim=-1)
        t = F.softmax(t/tempt, dim=1).unsqueeze(dim=-1)
        tmp = t*torch.log(s).transpose(1,2)
        tmp = tmp.squeeze()
        print(tmp.shape)
        return -torch.sum(tmp, dim=-1)

    def forward(self, imgs, smaller_imgs, mask_ratio=0.75, device=None, double_loss=False, epoch=0):
        latent, mask, ids_restore, _ = self.forward_encoder(
            imgs, mask_ratio, None)
        pred, z, _ = self.forward_decoder(
            latent, ids_restore, True, False)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask)
        if double_loss:
            latent_smaller, _, _, _ = self.forward_encoder(
                smaller_imgs, 0, None)
            _, z_, _ = self.forward_decoder(
                latent_smaller, ids_restore, False, False)
            # p, p_, z, z_ = self.forward_linears(z, z_)
            knowledge_loss = self.kl_loss(z_, z)
            """
            _, p_, pp_ = self.forward_decoder(latent_smaller, ids_restore, False, self.global_pool)
            if self.global_pool:
                p, p_ = p.detach(), p_.detach()
                sim_loss = self.twin_loss(p, p_, pp, pp_)
            else:
                sim_loss = self.twin_loss(z, z_, p, p_)
            """
            total_loss = loss + 0.25 * (0.995 ** epoch) * knowledge_loss

        else:
            knowledge_loss = torch.zeros(1).to(device)
            total_loss = loss

        return total_loss, knowledge_loss, loss, pred, mask


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
