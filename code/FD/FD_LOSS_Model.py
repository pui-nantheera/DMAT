import os

import torch
from torch import nn
from torchvision import transforms
from transformers import ViTConfig, ViTFeatureExtractor

from .ViTForRegression import ViTForRegression


class FDLoss_Model(nn.Module):

    def __init__(
        self,
        not_freeze=True,
        model_path="./best_FD2D.pth.tar",
        # self, not_freeze=True, model_path="./best_FD3D.pth.tar"
    ):
        super().__init__()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_id = "google/vit-base-patch16-224"
        custom_config = ViTConfig.from_pretrained(model_id)
        # adding custom parameter
        custom_config.my_num_labels = 1
        self.model = ViTForRegression.from_pretrained(
            model_id, config=custom_config
        ).to(device)

        self.model.load_state_dict(
            torch.load(os.path.join(model_path), map_location=device)
        )

        self.feature_extractor = ViTFeatureExtractor.from_pretrained(
            "google/vit-base-patch16-224", return_tensors="pt"
        )

        self.model.requires_grad_(not_freeze)
        for param in self.model.parameters():
            param.requires_grad = not_freeze
        self.model.eval()

    def forward(self, inputs, target):
        inputs = transforms.Resize((224, 224))(inputs)
        inputs = (inputs * 2.0) - 1.0
        outputs = self.model(inputs)
        return outputs["logits"].sum()
