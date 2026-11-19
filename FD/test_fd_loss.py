import urllib
import urllib.request

import cv2 as cv
import numpy as np
import pytest
import torch
from PIL import Image
from skimage import io
from src.FD.fd_loss import FdLoss


def pil2tensor(im):
    """_summary_

    Args:
        im (PIL image): input Image

    Returns:
        Tensor: float32 Tensor
    """
    # in: [PIL Image with 3 channels]. out: [B=1, C=3, H, W] (0, 1)
    return torch.Tensor((np.float32(im) / 255).transpose(2, 0, 1)).unsqueeze(0)


def get_image():
    # load image
    my_url = "https://iiif.lib.ncsu.edu/iiif/0052574/full/800,/0/default.jpg"
    # Download the image using urllib
    urllib.request.urlretrieve(my_url, "default.png")


def test_canny_fd():
    """Test of the fd loss function with canny edge detection"""

    get_image()
    # Open the downloaded image in PIL
    im = Image.open("default.png")
    ground_truth = pil2tensor(im).cuda(0)
    fdloss = FdLoss(canny=True)
    loss_fd = fdloss.get_fd_batch(ground_truth)
    loss_fd = np.float32(loss_fd.detach().cpu().numpy())
    assert abs(loss_fd - 0.18582) < 0.0001


def test_non_canny_fd():
    """Test of the fd loss function with non canny edge detection"""

    get_image()
    # Open the downloaded image in PIL
    im = Image.open("default.png")
    ground_truth = pil2tensor(im).cuda(0)
    fdloss = FdLoss(canny=False)
    loss_fd = fdloss.get_fd_batch(ground_truth)
    loss_fd = np.float32(loss_fd.detach().cpu().numpy())
    assert abs(loss_fd - 0.2554221) < 0.0001
