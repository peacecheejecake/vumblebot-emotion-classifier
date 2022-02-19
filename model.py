import os

import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class EmotionClassifier(nn.Module):

    def __init__(self, encoder, encoder_name, model_path, device):
        super().__init__()

        setattr(self, encoder_name, encoder)
        self.classifier = nn.Linear(encoder.config.hidden_size, 60)
        self.device = device
        self.encoder_name = encoder_name

        if model_path and os.path.exists(model_path):
            self.load_state_dict(torch.load(model_path, map_location="cpu"))

        self.to(device)

    def forward(self, inputs):
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        out = getattr(self, self.encoder_name)(**inputs, return_dict=False)
        return self.classifier(out[0][:, 0])


def load_model(config):
    encoder_name = f"{'al' if config.albert else ''}bert"
    plm_name = f"kykim/{encoder_name}-kor-base"

    if config.model_path is None:
        encoder = AutoModel.from_pretrained(plm_name)
    else:
        encoder = AutoModel.from_config(
            AutoConfig.from_pretrained(plm_name)
        )

    device = "cuda.0" if not config.cpu and torch.cuda.is_available() else "cpu"
    return EmotionClassifier(encoder, encoder_name, config.model_path, device)
