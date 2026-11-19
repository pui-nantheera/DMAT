import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

import data.transforms as T
from util.misc import NestedTensor


class MyDetr(nn.Module):
    """
    A wrapper around a DETR model for processing single video frames with additional TMT features.

    This module performs normalization, resizing, and creates attention masks before passing the input
    to a DETR model along with optionally weighted temporal features (TMT features).

    Attributes:
        detr_model (nn.Module): The DETR model used for object detection.
        norm (callable): A normalization transform using standard ImageNet mean and std.
        detr_input_size (tuple): Target resolution (height, width) for the DETR input.
        weight_tmt_features (float): Scaling factor for TMT feature input.
        device (torch.device): The target device for tensors and model computations.
    """

    def __init__(self, detr_model, weight_tmt_features, device, detr_input_size=(800, 800)):
        """
        Initializes the MyDetr module.

        Args:
            detr_model (nn.Module): Pretrained DETR model instance.
            weight_tmt_features (float): Scalar weight to scale TMT features before fusion.
            device (torch.device): Device to use for computation (e.g., 'cuda' or 'cpu').
            detr_input_size (tuple, optional): Target (H, W) resolution for input frames. Default is (800, 800).
        """
        super(MyDetr, self).__init__()
        self.device = device
        self.detr_model = detr_model
        self.norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.detr_input_size = detr_input_size
        self.weight_tmt_features = weight_tmt_features

    def forward(self, x, tmt_features, frame):
        """
        Forward pass through the model for a single frame in a video sequence.

        Args:
            x (Tensor): Input tensor of shape (B, T, C, H, W), a sequence of frames.
            tmt_features (Tensor): Temporal features to fuse with the DETR model.
            frame (int): Index of the frame to extract from the sequence for processing.

        Returns:
            dict: The output of the DETR model, typically containing bounding boxes, labels, and logits.
        """
        x = x[:, frame, :, :, :]  # Extract selected frame
        x, _ = self.norm(x)       # Normalize with ImageNet stats
        x = TF.resize(x, self.detr_input_size)  # Resize to DETR input size

        b, _, h, w = np.shape(x)
        mask = torch.zeros((b, h, w), dtype=torch.bool, device=self.device)  # No padding mask

        # Pass through DETR model with weighted TMT features
        x = self.detr_model(
            NestedTensor(x, mask),
            tmt_features * self.weight_tmt_features,
        )
        return x
