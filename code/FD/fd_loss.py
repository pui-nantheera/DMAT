import cv2 as cv
import matplotlib.pylab as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from matplotlib.pyplot import imread, imsave
from PIL import Image
from scipy.stats import linregress
from skimage import io
from torch.nn import AvgPool2d, AvgPool3d

from .net_canny import Net

transform = transforms.ToTensor()


class FdLoss:
    """Class to generate FD from tensor"""

    def __init__(
        self,
        canny: bool = True,
        cuda_dev: int = 0,
        levels: int = 512,
        default_fd: float = 0.18,
        divisor: float = 10.0,
    ):
        """Initialise class

        Args:
            canny (bool, optional): Use canny or 3d FD. Defaults to True.
            cuda_dev (int, optional): GPU device number. Defaults to 0.
            levels (int, optional): When using 3d FD, how many levels to analyse. Defaults to 512.
            default_fd (float, optional): If nans are found in batch FD what to default FD to. Defaults to 0.18.
            divisor (float, optional): Reduce the range of FD values to be less than 1.0. Defaults to 10.0.
        """
        self.canny = canny
        self.cuda_dev = cuda_dev
        self.levels = levels
        self.levels_3d = torch.Tensor(range(1, self.levels - 1)) / self.levels
        self.default_fd = default_fd
        self.divisor = divisor
        # torch.set_default_tensor_type("torch.cuda.FloatTensor")

    def get_fd_batch(self, batch_colour_im: torch.Tensor) -> torch.Tensor:
        """Main generation method

        Args:
            batch_colour_im (torch.Tensor): Input torch tensor to generate FD values: BxCxWxH

        Returns:
            torch.Tensor: vector of FD values (one per batch index)
        """
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        if not self.canny:
            batch_bw_im = transforms.functional.rgb_to_grayscale(batch_colour_im)
            for idx, level in enumerate(self.levels_3d):
                this_mask = (batch_bw_im > level).float() * 1
                if idx == 0:
                    cat_mask = this_mask
                cat_mask = torch.cat((this_mask, cat_mask), 1)
            batch_fd_im = cat_mask

            tensor_size = batch_fd_im.shape
            # Get the list of sizes for box-counting
            sizes = self.get_sizes(tensor_size)

            # Perform box-counting
            count = self.get_count_batch3d(batch_fd_im, sizes, tensor_size)
        else:
            net = Net()
            batch_bw_im = transforms.functional.rgb_to_grayscale(batch_colour_im)
            _, thin_edges, _ = net(batch_bw_im)

            tensor_size = thin_edges.shape
            sizes = self.get_sizes(tensor_size)
            count = self.get_count_batch(thin_edges, sizes, tensor_size)

        # Get fractal dimensionality
        slopes = torch.zeros((tensor_size[0], 1))

        for batch_idx in range(tensor_size[0]):
            x = torch.log(torch.tensor(sizes))
            y = torch.log(count[batch_idx, :])

            xplusone = torch.cat(
                (x.unsqueeze(0).transpose(0, 1), torch.ones(x.size(0), 1)), 1
            )
            slopes[batch_idx] = (
                -torch.linalg.lstsq(xplusone, y).solution[0] / self.divisor
            )
        # return slopes.squeeze()
        return_value = torch.nan_to_num(slopes.squeeze(), self.default_fd)
        torch.set_default_tensor_type(torch.FloatTensor)
        return return_value

    def get_sizes(self, tensor_size: torch.Tensor) -> np.array:
        """Generate a list of sizes for the analysis of FD scales

        Args:
            tensor_size (torch.Tensor): Input tensor size

        Returns:
            np.array:  list of sizes for FractD scale / box based analysis
        """

        # Minimal dimension of image
        if tensor_size[1] > 1:
            min_dim = min(tensor_size[1], tensor_size[2], tensor_size[3])
        else:
            min_dim = min(tensor_size[2], tensor_size[3])

        # Greatest power of 2 less than or equal to min_dim/2
        n = 2 ** np.floor(np.log(min_dim / 2) / np.log(2))

        # Extract the exponent
        n = int(np.log(n) / np.log(2))

        # Build successive box sizes (from 2**n down to 2**1)
        return 2 ** np.arange(n, 1, -1)

    def get_count_batch3d(
        self, in_tensor: torch.Tensor, sizes: torch.Tensor, tensor_size: torch.Tensor
    ) -> torch.Tensor:
        """Generate non canny based FD

        Args:
            in_tensor (torch.Tensor): input tensor
            sizes (torch.Tensor): sizes of scale analysis
            tensor_size (torch.Tensor): size of in_tensor

        Returns:
            torch.Tensor: torch tensor
        """
        # Pre-allocation
        counts = torch.zeros(tensor_size[0], len(sizes))
        # counts.requires_grad = True
        index = 0

        # Transfer the array to a 4D CUDA Torch Tensor

        # Box-counting
        for idx, size in enumerate(sizes):
            stride = (size, size, size)
            kernel_size = (size, size, size)
            pool = AvgPool3d(kernel_size=kernel_size, stride=stride)
            pool_tensor = pool(in_tensor)
            count = torch.sum(
                torch.where(
                    (pool_tensor > 0) & (pool_tensor < 1),
                    torch.tensor([1]),
                    torch.tensor([0]),
                ),
                dim=(1, 2, 3),
            )
            counts[:, idx] = count
            index += 1
        return counts

    def narrow_gaussian(self, x, ell):
        return torch.exp(-0.5 * (x / ell) ** 2)

    def approx_count_nonzero(self, x, ell=1e-3):
        # Approximation of || x ||_0
        return x.shape[1] - self.narrow_gaussian(x, ell).sum(dim=1)

    def get_count_batch(
        self, in_tensor: torch.Tensor, sizes: torch.Tensor, tensor_size: torch.Tensor
    ) -> torch.Tensor:
        """Generate canny based FD

        Args:
            in_tensor (torch.Tensor): input tensor
            sizes (torch.Tensor): sizes of scale analysis
            tensor_size (torch.Tensor): size of in_tensor

        Returns:
            torch.Tensor: torch tensor of counts
        """
        # Pre-allocation
        counts = torch.zeros(tensor_size[0], len(sizes))
        index = 0

        # Box-counting
        for idx, size in enumerate(sizes):
            stride = (size, size)
            kernel_size = (size, size)
            pool = AvgPool2d(kernel_size=kernel_size, stride=stride)
            pool_input = pool(in_tensor)
            pool_input2 = torch.reshape(
                pool_input,
                (
                    pool_input.shape[0],
                    -1,
                ),
            )
            count = self.approx_count_nonzero(pool_input2)
            counts[:, idx] = count
            index += 1
        return counts
