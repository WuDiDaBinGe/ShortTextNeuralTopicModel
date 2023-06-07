import torch
from torch import nn
import torch.nn.functional as F
from configuration_utils import BaseTopicConfig
from modeling_utils import BaseTopicNeuralModel

class BaseTopicNeuralModel(nn.Module):
    def __init__(self, config:BaseTopicNeuralModel):
        super(BaseTopicNeuralModel, self).__init__()
        self.vocab_size = config.vocab_size
        self.hidden_dim = config.hidden_dim
        self.topic_num = config.topic_number
        self.device = config.device

        encode_dims = [self.vocab_size, self.hidden_dim, self.topic_num]
        self.encoder = nn.ModuleDict({
            f'enc_{i}': nn.Linear(encode_dims[i], encode_dims[i + 1])
            for i in range(len(encode_dims) - 1)
        })

        self.batch_norm = nn.BatchNorm1d(self.topic_num)
        self.decoder = nn.Linear(self.topic_num, self.vocab_size, bias=False)
        self.dropout = nn.Dropout(p=config.dropout)
        self.batch_norm_final = nn.BatchNorm1d(num_features=self.vocab_size)
        self.non_line = F.leaky_relu


    def forward(self, bows):
        raise NotImplementedError("not Implemented")

    def _encode(self, encodee_input):
        """
            encoder forward
        """
        hid = encodee_input
        for i, (_, layer) in enumerate(self.encoder.items()):
            hid = self.dropout(layer(hid))
            if i < len(self.encoder) - 1:
                hid = self.nonlin(hid)
        hid = self.batch_norm(hid)
        return hid

    def _decode(self, decoder_input):
        """
            decoder forward
        """
        x_rec = F.softmax(self.batch_norm_final(self.decoder(decoder_input)), dim=1)
        return x_rec

    def _assist_loss(self):
        """
            vae,wvae assist loss such as: KL Loss, MMD Loss etc
        """
        raise NotImplementedError("not Implemented")

    def get_topic_words(self):
        return self.decoder.weight.T

    