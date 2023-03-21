from torch import nn, Tensor
from collections.abc import Sequence


class PatchEmbedding(nn.Module):
    def __init__(self, img_size: int | Sequence[int] = 96, patch_size: int | Sequence[int] = 16, num_hiddens: int = 512) -> None:
        super().__init__()

        if not isinstance(img_size, Sequence):
            img_size = (img_size, img_size)

        if not isinstance(patch_size, Sequence):
            patch_size = (patch_size, patch_size)

        self.num_patches = (
            img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])

        self.conv = nn.LazyConv2d(num_hiddens, kernel_size=tuple(
            patch_size), stride=tuple(patch_size))

    def forward(self, X: Tensor) -> Tensor:
        return self.conv(X).flatten(2).transpose(1, 2)
