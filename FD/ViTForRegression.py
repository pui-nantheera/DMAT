import torch
import torchvision.transforms as transforms
from torchvision import models, transforms
from transformers import ViTConfig, ViTModel, ViTPreTrainedModel

transform = transforms.ToTensor()


class ViTForRegression(ViTPreTrainedModel):
    def __init__(self, config: ViTConfig):
        # not needed because config is already passed
        # config = ViTConfig.from_pretrained(model_name_or_path)
        super().__init__(config)
        self.vit = ViTModel(config)
        self.regressor = torch.nn.Linear(config.hidden_size, config.my_num_labels)
        self.activation = torch.nn.Sigmoid()

        self.loss_fn = torch.nn.MSELoss()

        self.post_init()

    def forward(self, pixel_values, labels=None):
        loss = None
        outputs = self.vit(pixel_values=pixel_values)
        reg = self.regressor(outputs.last_hidden_state[:, 0])
        logits = self.activation(reg)
        self.loss_fn = torch.nn.MSELoss()
        if labels is not None:
            loss = self.loss_fn(logits, labels)
        return {"logits": logits, "loss": loss}
