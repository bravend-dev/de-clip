"""Vision tower of DermLIP (PanDerm-base ViT-B/16).

DermLIP pairs a PubMedBERT text encoder with the PanDerm image encoder, a ViT-B/16 with
layer-scale, fixed 2-D sin-cos position embeddings and mean pooling. The module names
match the released checkpoint so its state dict loads directly; nothing is trained here.
"""

from __future__ import annotations

from functools import partial

import torch
import torch.nn.functional as F
from torch import nn


class PatchEmbed(nn.Module):
    def __init__(self, image_size: int, patch_size: int, in_chans: int, embed_dim: int):
        super().__init__()
        self.patch_shape = (image_size // patch_size, image_size // patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)


class Attention(nn.Module):
    """Multi-head self-attention with separate q/v biases (no k bias), as in BEiT/CAE."""

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(dim))
        self.v_bias = nn.Parameter(torch.zeros(dim))
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, dim = x.shape
        bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias), self.v_bias))
        qkv = F.linear(x, self.qkv.weight, bias)
        qkv = qkv.reshape(batch, tokens, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(batch, tokens, dim)
        return self.proj(x)


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float, norm_layer, init_values: float):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio))
        self.gamma_1 = nn.Parameter(init_values * torch.ones(dim))
        self.gamma_2 = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.gamma_1 * self.attn(self.norm1(x))
        x = x + self.gamma_2 * self.mlp(self.norm2(x))
        return x


def sincos_position_embedding(patch_shape: tuple[int, int], embed_dim: int, temperature=10000.0):
    h, w = patch_shape
    grid_w, grid_h = torch.meshgrid(
        torch.arange(w, dtype=torch.float32), torch.arange(h, dtype=torch.float32), indexing="ij"
    )
    pos_dim = embed_dim // 4
    omega = 1.0 / (temperature ** (torch.arange(pos_dim, dtype=torch.float32) / pos_dim))
    out_w = torch.einsum("m,d->md", grid_w.flatten(), omega)
    out_h = torch.einsum("m,d->md", grid_h.flatten(), omega)
    pos = torch.cat([out_w.sin(), out_w.cos(), out_h.sin(), out_h.cos()], dim=1)[None]
    cls = torch.zeros(1, 1, embed_dim)
    return torch.cat([cls, pos], dim=1)


class PanDermVisionTransformer(nn.Module):
    """ViT-B/16 image encoder of DermLIP; ``forward`` returns the projected image embedding."""

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        init_values: float = 0.1,
        output_dim: int = 512,
    ):
        super().__init__()
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.image_size = (image_size, image_size)
        self.embed_dim = embed_dim
        self.patch_embed = PatchEmbed(image_size, patch_size, in_chans, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            sincos_position_embedding(self.patch_embed.patch_shape, embed_dim), requires_grad=False
        )
        self.blocks = nn.ModuleList(
            [Block(embed_dim, num_heads, mlp_ratio, norm_layer, init_values) for _ in range(depth)]
        )
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls, x), dim=1) + self.pos_embed.type_as(x)
        for block in self.blocks:
            x = block(x)
        # The released model normalises the token sequence and then the mean-pooled vector.
        x = self.norm(x)
        pooled = self.norm(x[:, 1:, :].mean(dim=1))
        return self.head(pooled)

    def set_grad_checkpointing(self, enable: bool = True) -> None:  # open_clip API compatibility
        del enable

    def lock(self, *args, **kwargs) -> None:  # open_clip API compatibility
        del args, kwargs
        for p in self.parameters():
            p.requires_grad = False


__all__ = ["PanDermVisionTransformer"]
