import torch
from torch import nn
from torch.nn import functional as F
frin decoder import VAE_AttentionBlock, VAE_ResidualBlock


class VAE_Encoder(nn.Module):

    def __init__(self):
        super().__init__(
            # (batch_size, channels, height, width) -> (batch_size, 128, height, width)
            nn.Conv2d(3,128, kernel_size=3, padding=1),


            # (batch_size, 128, height, width) -> (batch_size, 128, height, width)
            VAE_ResidualBlock(128, 128),

        
            # (batch_size, 128, height, width) -> (batch_size, 128, height, width)
            VAE_ResidualBlock(128, 128),
            

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),




        )
