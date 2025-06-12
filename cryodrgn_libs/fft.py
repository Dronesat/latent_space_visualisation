"""Utility functions used in Fast Fourier transform calculations on image tensors."""

import numpy as np
import torch
from torch.fft import fftshift, fft2


def fft2_center(img: torch.Tensor) -> torch.Tensor:
    """2-dimensional discrete Fourier transform reordered with origin at center."""
    if img.dtype == torch.float16:
        img = img.type(torch.float32)

    return fftshift(fft2(fftshift(img, dim=(-1, -2))), dim=(-1, -2))

#cryo
def ht2_center(img: torch.Tensor) -> torch.Tensor:
    """2-dimensional discrete Hartley transform reordered with origin at center."""
    img = fft2_center(img)
    return img.real - img.imag

#cryo
def symmetrize_ht(ht: torch.Tensor) -> torch.Tensor:
    if ht.ndim == 2:
        ht = ht[np.newaxis, ...]
    assert ht.ndim == 3
    n = ht.shape[0]

    D = ht.shape[-1]
    sym_ht = torch.empty((n, D + 1, D + 1), dtype=ht.dtype, device=ht.device)
    sym_ht[:, 0:-1, 0:-1] = ht

    assert D % 2 == 0
    sym_ht[:, -1, :] = sym_ht[:, 0, :]  # last row is the first row
    sym_ht[:, :, -1] = sym_ht[:, :, 0]  # last col is the first col
    sym_ht[:, -1, -1] = sym_ht[:, 0, 0]  # last corner is first corner

    if n == 1:
        sym_ht = sym_ht[0, ...]

    return sym_ht