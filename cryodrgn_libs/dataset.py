"""Classes for using particle image datasets in PyTorch learning methods.

This module contains classes that implement various preprocessing and data access
methods acting on the image data stored in a cryodrgn.source.ImageSource class.
These methods are used by learning methods such as those used in volume reconstruction
algorithms; the classes are thus implemented as children of torch.utils.data.Dataset
to allow them to inherit behaviour such as batch training.

For example, during initialization, ImageDataset initializes an ImageSource class and
then also estimates normalization parameters, a non-trivial computational step. When
image data is retrieved during model training using __getitem__, the data is whitened
using these parameters.

"""
import numpy as np

import logging
import torch
from typing import Union
import cryodrgn_libs.fft as fft
from cryodrgn_libs.source import ImageSource
from cryodrgn_libs.masking import spherical_window_mask


logger = logging.getLogger(__name__)


class ImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        mrcfile,
        lazy=True,
        norm=None,
        keepreal=False,
        invert_data=False,
        ind=None,
        window=True,
        datadir=None,
        window_r=0.85,
        max_threads=16,
        device: Union[str, torch.device] = "cpu",
    ):
        assert not keepreal, "Not implemented yet"
        datadir = datadir or ""
        self.ind = ind
        self.src = ImageSource.from_file(
            mrcfile,
            lazy=lazy,
            datadir=datadir,
            indices=ind,
            max_threads=max_threads,
        )
        ny = self.src.D
        assert ny % 2 == 0, "Image size must be even."
        self.N = self.src.n
        self.D = ny + 1  # after symmetrization
        self.invert_data = invert_data

        if window:
            self.window = spherical_window_mask(D=ny, in_rad=window_r, out_rad=0.99).to(
                device
            )
        else:
            self.window = None

        norm = norm or self.estimate_normalization()
        self.norm = [float(x) for x in norm]
        self.device = device
        self.lazy = lazy

        if np.issubdtype(self.src.dtype, np.integer):
            self.window = self.window.int()

    def estimate_normalization(self, n=1000):
        n = min(n, self.N) if n is not None else self.N
        indices = range(0, self.N, self.N // n)  # FIXME: what if the data is not IID??

        imgs = torch.stack([fft.ht2_center(img) for img in self.src.images(indices)])
        if self.invert_data:
            imgs *= -1

        imgs = fft.symmetrize_ht(imgs)
        norm = (0, torch.std(imgs))
        logger.info("Normalizing HT by {} +/- {}".format(*norm))

        return norm

    

    

    