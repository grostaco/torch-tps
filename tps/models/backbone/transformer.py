import torch
from torch import nn, Tensor
from collections import OrderedDict
import math
from typing import cast, overload, Literal


class AddNorm(nn.Module):
    def __init__(self, shape: int, dropout=.5) -> None:
        super().__init__()

        self.ln = nn.LayerNorm(shape)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X: Tensor, Y: Tensor):
        return self.dropout(self.ln(Y) + X)


class PositionWiseFFN(nn.Module):
    def __init__(self, num_hiddens: int, num_outputs: int) -> None:
        super().__init__()

        self.dense1 = nn.LazyLinear(num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.LazyLinear(num_outputs)

    def forward(self, X: Tensor) -> Tensor:
        X = self.dense1(X)
        X = self.relu(X)
        X = self.dense2(X)

        return X


class PositionEncoding(nn.Module):
    def __init__(self, num_hiddens: int, dropout: float, max_len: int = 1000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        self.P = nn.Parameter(torch.zeros(
            (1, max_len, num_hiddens)), requires_grad=False)
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(
            10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)

        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X: Tensor) -> Tensor:
        X = X + self.P[:, :X.shape[1], :]
        return self.dropout(X)


class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim: int, ffn_num_hiddens: int, num_heads: int, dropout: float, bias=False) -> None:
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout, bias)

        self.addnorm1 = AddNorm(embed_dim, dropout)
        self.ffn = PositionWiseFFN(ffn_num_hiddens, embed_dim)
        self.addnorm2 = AddNorm(embed_dim, dropout)

    @overload
    def forward(self, X: Tensor, valid_lens: Tensor,
                need_weights: Literal[False]) -> Tensor: ...

    @overload
    def forward(self, X: Tensor, valid_lens: Tensor,
                need_weights: Literal[True]) -> tuple[Tensor, Tensor]: ...

    def forward(self, X: Tensor, valid_lens: Tensor, need_weights=False):
        attn_output, attn_weights = self.attention(
            X, X, X, valid_lens, need_weights=need_weights)
        X = self.addnorm1(X, attn_output)
        X = self.addnorm2(X, self.ffn(X))

        if need_weights:
            return X, attn_weights
        return X


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size: int, num_hiddens: int, ffn_num_hiddens: int, num_heads: int,
                 num_blks: int, dropout: float, bias=False) -> None:
        super().__init__()

        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential(
            OrderedDict((f'block.{i}', TransformerEncoderBlock(
                num_hiddens, ffn_num_hiddens, num_heads, dropout, bias)) for i in range(num_blks))
        )
        self.attention_weights: list[Tensor | None] = [None] * len(self.blks)

    def forward(self, X: Tensor, valid_lens: Tensor) -> Tensor:
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))

        for i, blk in enumerate(self.blks):
            blk = cast(TransformerEncoderBlock, blk)

            X, weights = blk.forward(X, valid_lens, need_weights=True)
            self.attention_weights[i] = weights

        return X
