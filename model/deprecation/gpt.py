"""
2020-12-22 已经验证 收敛速度较慢
@Author: zhangluoyang
@E-mail: 55058629@qq.com
@Time:  12月 16, 2020
"""
import torch
import torch.nn as nn
from typing import Union, Tuple


class Block(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 num_heads: int):
        super(Block, self).__init__()
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # attn mask will add attn weight
        attn_mask = torch.full((len(x), len(x)), -float("Inf"), device=x.device, dtype=x.dtype)
        attn_mask = torch.triu(attn_mask, diagonal=1)

        x = self.ln_1(x)
        a, _ = self.attn(x, x, x, attn_mask=attn_mask, need_weights=False)
        x = x + a
        m = self.mlp(self.ln_2(x))
        x = x + m
        return x


class GPT2(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 num_layers: int,
                 num_positions: int,
                 num_vocab: int,
                 num_classes: int):
        super(GPT2, self).__init__()
        self.embed_dim = embed_dim
        # start of sequence token
        self.sos = torch.nn.Parameter(torch.zeros(embed_dim))
        nn.init.normal_(self.sos)

        self.token_embeddings = nn.Embedding(num_vocab, embed_dim)
        self.position_embeddings = nn.Embedding(num_positions, embed_dim)

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(Block(embed_dim, num_heads))

        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_vocab, bias=False)
        self.clf_head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor, classify: bool = False) \
            -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        Expect input as shape [sequence len, batch]
        If classify, return classification logits
        """
        length, batch = x.shape
        # []
        h = self.token_embeddings(x)

        # prepend sos token
        sos = torch.ones(1, batch, self.embed_dim, device=x.device) * self.sos
        h = torch.cat([sos, h[:-1, :, :]], axis=0)

        # add positional embeddings
        positions = torch.arange(length, device=x.device).unsqueeze(-1)
        h = h + self.position_embeddings(positions).expand_as(h)

        # transformer
        for layer in self.layers:
            h = layer(h)

        h = self.ln_f(h)

        logits = self.head(h)
        if not classify:
            return logits

        h = torch.mean(h, dim=0)
        return self.clf_head(h), logits


class ImageGPT(nn.Module):

    def __init__(self,
                 num_pixels: int,
                 num_vocab: int,
                 num_classes: int,
                 embed_dim: int = 64,
                 num_heads: int = 2,
                 num_layers: int = 8,
                 classify: bool = False):
        super(ImageGPT, self).__init__()
        self.num_pixels = num_pixels
        self.num_vocab = num_vocab
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.classify = classify

        self.gpt = GPT2(embed_dim=embed_dim,
                        num_heads=num_heads,
                        num_layers=num_layers,
                        num_positions=num_pixels * num_pixels,
                        num_vocab=num_vocab,
                        num_classes=num_classes)

    def forward(self, x: torch.Tensor, classify: bool = False) \
            -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """

        :param x: [sequence, batch]
        :param classify:
        :return:
        """
        return self.gpt(x, classify)
