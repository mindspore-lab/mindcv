from functools import partial
from typing import Optional

import numpy as np

import mindspore as ms
from mindspore import Parameter, Tensor, nn, ops
from mindspore.common.initializer import Normal, initializer

from .helpers import load_pretrained
from .registry import register_model
from .vit_encoder import Block, VisionTransformerEncoder

__all__ = [
    "mae_b_16_224_pretrain",
    "mae_l_16_224_pretrain",
    "mae_h_16_224_pretrain",
    "mae_b_16_224_finetune",
    "mae_l_16_224_finetune",
    "mae_h_14_224_finetune"
]


def _cfg(url="", **kwargs):
    return {
        "url": url,
        "num_classes": 1000,
        "input_size": (3, 224, 224),
        "first_conv": "patch_embed.proj",
        "classifier": "head",
        **kwargs,
    }


default_cfgs = {
    "mae_b_16_224_finetune": _cfg(
        url="https://download.mindspore.cn/toolkits/mindcv/mae/mae_b_16_224_finetune-cc05b899.ckpt"
    ),
    "mae_l_16_224_finetune": _cfg(url=""),
    "mae_h_14_224_finetune": _cfg(url=""),
}


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
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
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class MAEForPretrain(VisionTransformerEncoder):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
        act_layer: nn.Cell = nn.GELU,
        norm_layer: nn.Cell = nn.LayerNorm,
        norm_pix_loss: bool = True,
        mask_ratio: float = 0.75,
        **kwargs
    ):
        super(MAEForPretrain, self).__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            init_values=None,
            act_layer=act_layer,
            norm_layer=norm_layer,
            use_abs_pos_emb=True,
            use_rel_pos_bias=False,
            use_shared_rel_pos_bias=False,
            **kwargs
        )
        self.cls_token = Parameter(initializer(Normal(sigma=0.02), (1, 1, embed_dim)))

        self.unmask_len = int(np.floor(self.num_patches * (1 - mask_ratio)))

        encoder_pos_emb = Tensor(get_2d_sincos_pos_embed(
            embed_dim, int(self.num_patches ** 0.5), cls_token=True), ms.float32
        )
        encoder_pos_emb = ops.expand_dims(encoder_pos_emb, axis=0)
        self.pos_embed = Parameter(encoder_pos_emb, requires_grad=False)
        self.norm = norm_layer((embed_dim,))

        self.decoder_embed = nn.Dense(embed_dim, decoder_embed_dim)
        self.mask_token = Parameter(initializer(Normal(sigma=0.02), (1, 1, decoder_embed_dim)))

        decoder_pos_emb = Tensor(get_2d_sincos_pos_embed(
            decoder_embed_dim, int(self.num_patches ** 0.5), cls_token=True), ms.float32
        )
        decoder_pos_emb = ops.expand_dims(decoder_pos_emb, axis=0)
        self.decoder_pos_embed = Parameter(decoder_pos_emb, requires_grad=False)

        self.decoder_blocks = nn.CellList([
            Block(
                dim=decoder_embed_dim, num_heads=decoder_num_heads, qkv_bias=True,
                mlp_ratio=mlp_ratio, init_values=None, act_layer=act_layer, norm_layer=norm_layer,
            ) for _ in range(decoder_depth)
        ])
        self.decoder_norm = norm_layer((decoder_embed_dim,))
        self.decoder_pred = nn.Dense(decoder_embed_dim, patch_size ** 2 * in_chans)

        self.sort = ops.Sort()

        self.norm_pix_loss = norm_pix_loss
        self._init_weights()

    def _init_weights(self):
        for name, cell in self.cells_and_names():
            if isinstance(cell, nn.Dense):
                cell.weight.set_data(
                    initializer("xavier_uniform", cell.weight.shape, cell.weight.dtype)
                )
                if cell.bias is not None:
                    cell.bias.set_data(
                        initializer('zeros', cell.bias.shape, cell.bias.dtype)
                    )

            elif isinstance(cell, nn.LayerNorm):
                cell.gamma.set_data(
                    initializer('ones', cell.gamma.shape, cell.gamma.dtype)
                )
                cell.beta.set_data(
                    initializer('zeros', cell.beta.shape, cell.beta.dtype)
                )

            if name == "patch_embed.proj":
                cell.weight.set_data(
                    initializer("xavier_uniform", cell.weight.shape, cell.weight.dtype)
                )

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size ** 2 * 3)
        """
        N, _, H, W = imgs.shape
        p = self.patch_embed.patch_size[0]
        assert H == W and H % p == 0
        h = w = H // p

        x = ops.reshape(imgs, (N, 3, h, p, w, p))
        x = ops.transpose(x, (0, 2, 4, 3, 5, 1))
        x = ops.reshape(x, (N, h * w, p ** 2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size ** 2 * 3)
        imgs: (N, 3, H, W)
        """
        N, L, _ = x.shape
        p = self.patch_embed.patch_size[0]
        h = w = int(L ** 0.5)
        assert h * w == L

        imgs = ops.reshape(x, (N, h, w, p, p, 3))
        imgs = ops.transpose(imgs, (0, 5, 1, 3, 2, 4))
        imgs = ops.reshape(imgs, (N, 3, h * p, w * p))
        return imgs

    def apply_masking(self, x, mask):
        D = x.shape[2]
        _, ids_shuffle = self.sort(mask.astype(ms.float32))
        _, ids_restore = self.sort(ids_shuffle.astype(ms.float32))

        ids_keep = ids_shuffle[:, :self.unmask_len]
        ids_keep = ops.broadcast_to(ops.expand_dims(ids_keep, axis=-1), (-1, -1, D))
        x_unmasked = ops.gather_elements(x, dim=1, index=ids_keep)

        return x_unmasked, ids_restore

    def forward_features(self, x, mask):
        x = self.patch_embed(x)
        bsz = x.shape[0]

        x = x + self.pos_embed[:, 1:, :]
        x, ids_restore = self.apply_masking(x, mask)

        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_token = ops.broadcast_to(cls_token, (bsz, -1, -1))
        cls_token = cls_token.astype(x.dtype)
        x = ops.concat((cls_token, x), axis=1)

        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)
        return x, ids_restore

    def forward_decoder(self, x, ids_restore):
        x = self.decoder_embed(x)
        bsz, L, D = x.shape

        mask_len = self.num_patches + 1 - L
        mask_tokens = ops.broadcast_to(self.mask_token, (bsz, mask_len, -1))
        mask_tokens = mask_tokens.astype(x.dtype)

        x_ = ops.concat((x[:, 1:, :], mask_tokens), axis=1)
        ids_restore = ops.broadcast_to(ops.expand_dims(ids_restore, axis=-1), (-1, -1, D))
        x_ = ops.gather_elements(x_, dim=1, index=ids_restore)
        x = ops.concat((x[:, :1, :], x_), axis=1)

        x = x + self.decoder_pos_embed

        for blk in self.decoder_blocks:
            x = blk(x)

        x = self.decoder_norm(x)
        x = self.decoder_pred(x)

        return x[:, 1:, :]

    def forward_loss(self, imgs, pred, mask):
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(axis=-1, keep_dims=True)
            std = target.std(axis=-1, keepdims=True)
            target = (target - mean) / std

        loss = (pred - target) ** 2
        loss = loss.mean(axis=-1)

        mask = mask.astype(loss.dtype)
        loss = (loss * mask).sum() / mask.sum()
        return loss

    def construct(self, imgs, mask):
        bsz = imgs.shape[0]
        mask = ops.reshape(mask, (bsz, -1))
        features, ids_restore = self.forward_features(imgs, mask)
        pred = self.forward_decoder(features, ids_restore)
        loss = self.forward_loss(imgs, pred, mask)
        return loss


class MAEForFinetune(VisionTransformerEncoder):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        attn_head_dim: Optional[int] = None,
        mlp_ratio: float = 4.,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        pos_drop_rate: float = 0.,
        proj_drop_rate: float = 0.,
        attn_drop_rate: float = 0.,
        drop_path_rate: float = 0.,
        act_layer: nn.Cell = nn.GELU,
        norm_layer: nn.Cell = nn.LayerNorm,
        num_classes: int = 1000,
        use_mean_pooling: bool = True,
        **kwargs
    ):
        super(MAEForFinetune, self).__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            attn_head_dim=attn_head_dim,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            pos_drop_rate=pos_drop_rate,
            proj_drop_rate=proj_drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            init_values=None,
            act_layer=act_layer,
            norm_layer=norm_layer,
            use_abs_pos_emb=True,
            use_rel_pos_bias=False,
            use_shared_rel_pos_bias=False,
            **kwargs
        )
        self.use_mean_pooling = use_mean_pooling
        if self.use_mean_pooling:
            self.fc_norm = norm_layer((embed_dim,))
        else:
            self.norm = norm_layer((embed_dim,))
        self.head = nn.Dense(embed_dim, num_classes, weight_init='TruncatedNormal')

        self._init_weights()
        self._fix_init_weights()

    def construct(self, x):
        x = self.forward_features(x)
        if self.use_mean_pooling:
            x = x[:, 1:].mean(axis=1)
            x = self.fc_norm(x)
        else:
            x = self.norm(x)
            x = x[:, 0]
        x = self.head(x)
        return x


@register_model
def mae_b_16_224_pretrain(pretrained=False, **kwargs):
    model = MAEForPretrain(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        act_layer=partial(nn.GELU, approximate=False),
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6), **kwargs
    )
    if pretrained:
        pass
    return model


@register_model
def mae_l_16_224_pretrain(pretrained=False, **kwargs):
    model = MAEForPretrain(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        act_layer=partial(nn.GELU, approximate=False),
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6), **kwargs
    )
    if pretrained:
        pass
    return model


@register_model
def mae_h_16_224_pretrain(pretrained=False, **kwargs):
    model = MAEForPretrain(
        patch_size=16, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        act_layer=partial(nn.GELU, approximate=False),
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6), **kwargs
    )
    if pretrained:
        pass
    return model


@register_model
def mae_b_16_224_finetune(pretrained=True, in_chans=3, num_classes=1000, **kwargs):
    default_cfg = default_cfgs["mae_b_16_224_finetune"]
    model = MAEForFinetune(
        patch_size=16, in_chans=in_chans, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, epsilon=1e-6), num_classes=num_classes, **kwargs
    )
    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_chans)
    return model


@register_model
def mae_l_16_224_finetune(pretrained=True, in_chans=3, num_classes=1000, **kwargs):
    default_cfg = default_cfgs["mae_l_16_224_finetune"]
    model = MAEForFinetune(
        patch_size=16, in_chans=in_chans, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, epsilon=1e-6), num_classes=num_classes, **kwargs
    )
    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_chans)
    return model


@register_model
def mae_h_14_224_finetune(pretrained=True, in_chans=3, num_classes=1000, **kwargs):
    default_cfg = default_cfgs["mae_h_14_224_finetune"]
    model = MAEForFinetune(
        patch_size=14, in_chans=in_chans, embed_dim=1280, depth=32, num_heads=16, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, epsilon=1e-6), num_classes=num_classes, **kwargs
    )
    if pretrained:
        load_pretrained(model, default_cfg, num_classes=num_classes, in_channels=in_chans)
    return model
