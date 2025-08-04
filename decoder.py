import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttentionBlock

class VAE_AttentionBlock(nn.Module):

    def __init__(self, channels: int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttentionBlock(1, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, features, height, width)
        
        residue = x

        n, c, h, w = x.shape

        # (batch_size, features, height, width) -> (batch_size, features, height * width)
        x = x.view(n, c, h * w) 

        # (batch_size, features, height * width) -> (batch_size, height * width, features)
        x = x.transpose(-1, -2)

        # (batch_size, height * width, features) -> (batch_size, height * width, features)  
        x = self.attention(x)

        # (batch_size, height * width, features) -> (batch_size, features, height * width)
        X = x.transpose(-1, -2)

        # (batch_size, features, height * width) -> (batch_size, features, height, width)
        X = X.view(n, c, h, w)

        X += residue

        return X

class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels != out_channels:
            self.residental_layer = nn.Identity()
        else:
            self.residental_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, in_channels, height, width)

        residue = x

        x = self.groupnorm_1(x)

        x = F.silu(x)

        z = self.conv_1(x)

        x = self.groupnorm_2(x)

        x = F.silu(x)

        x = self.conv_2(x)

        return x + self.residental_layer(residue)