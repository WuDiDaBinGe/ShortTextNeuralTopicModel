import torch
from torch import nn

class TopicNeuralModel(nn.Module):
    def __init__(self, config):
        raise NotImplementedError("not Implemented")

    def forward(self):
        raise NotImplementedError("not Implemented")

    def _encode(self):
        """
            encoder forward
        """
        raise NotImplementedError("not Implemented")

    def _decode(self):
        """
            decoder forward
        """
        raise NotImplementedError("not Implemented")

    def _assist_loss(self):
        """
            vae,wvae assist loss such as: KL Loss, MMD Loss etc
        """
        raise NotImplementedError("not Implemented")

    