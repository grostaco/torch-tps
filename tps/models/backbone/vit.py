import torch
from torch import nn, Tensor
from collections import OrderedDict

from ..embeddings.patch import PatchEmbedding


class ViTMLP(nn.Module):
    def __init__(self, num_hiddens: int, num_outputs: int, dropout=.5) -> None:
        super().__init__()
        self.dense1 = nn.LazyLinear(num_hiddens)
        self.gelu = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.dense2 = nn.LazyLinear(num_outputs)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = self.dense1(x)
        x = self.gelu(x)
        x = self.dropout1(x)
        x = self.dense2(x)
        x = self.dropout2(x)
        return x


class ViTBlock(nn.Module):
    def __init__(self, num_hiddens: int, norm_shape: int, mlp_num_hiddens: int, num_heads: int, dropout: float, use_bias=False) -> None:
        super().__init__()

        self.ln1 = nn.LayerNorm(norm_shape)
        self.attention = nn.MultiheadAttention(
            num_hiddens, num_heads, dropout, use_bias)
        self.ln2 = nn.LayerNorm(norm_shape)
        self.mlp = ViTMLP(mlp_num_hiddens, num_hiddens, dropout)

    def forward(self, X: Tensor, valid_lens=None) -> Tensor:
        X = X + self.attention(*([self.ln1(X)] * 3), valid_lens)
        return X + self.mlp(self.ln2(X))


class ViT(nn.Module):
    def __init__(self, img_size: int, patch_size: int, num_hiddens: int, mlp_num_hiddens: int, num_heads: int, num_blks: int, emb_dropout: int, blk_dropout: int,
                 *,
                 use_bias=False, num_classes=10) -> None:
        super().__init__()

        self.patch_embedding = PatchEmbedding(
            img_size, patch_size, num_hiddens)
        self.cls_token = nn.Parameter(torch.zeros(
            1, 1, num_hiddens), requires_grad=False)

        num_steps = self.patch_embedding.num_patches + 1
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_steps, num_hiddens))

        self.dropout = nn.Dropout(emb_dropout)
        self.blks = nn.Sequential(OrderedDict(
            {f'ViTBlock.{i}': ViTBlock(num_hiddens, num_hiddens, mlp_num_hiddens, num_heads, blk_dropout, use_bias) for i in range(num_blks)}))

        self.head = nn.Sequential(nn.LayerNorm(
            num_hiddens), nn.Linear(num_hiddens, num_classes))

    def forward(self, X: Tensor) -> Tensor:
        X = self.patch_embedding(X)
        X = torch.cat((self.cls_token.expand(X.shape[0], -1, -1), X), 1)
        X = self.dropout(X + self.pos_embedding)
        X = self.blks(X)
        return self.head(X[:, 0])
