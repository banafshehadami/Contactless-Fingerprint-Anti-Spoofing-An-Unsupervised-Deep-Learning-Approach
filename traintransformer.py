import torch.nn as nn

from decoder_swin import Decoder_swin
from encoder_swin import Encoder_swin

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder_swin(embed_dim=16*8)

        # Decoder
        self.decoder = Decoder_swin(embed_dim=16*8)



    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x