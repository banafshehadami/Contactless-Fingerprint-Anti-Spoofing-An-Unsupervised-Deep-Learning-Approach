import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = self.sigmoid(avg_out + max_out)
        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        out = self.sigmoid(out)
        return out


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x_out = self.channel_attention(x) * x
        x_out = self.spatial_attention(x_out) * x_out
        return x_out


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # 128x128x64
            nn.ReLU(inplace=True),
            CBAM(64),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 64x64x128
            nn.ReLU(inplace=True),
            CBAM(128),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1), # 32x32x256
            nn.ReLU(inplace=True),
            CBAM(256),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1), # 16x16x512
            nn.ReLU(inplace=True),
            CBAM(512),
            nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1), # 8x8x1024
            nn.ReLU(inplace=True),
            CBAM(1024),
            nn.Conv2d(1024, 2048, kernel_size=4, stride=2, padding=1), # 4x4x2048
            nn.ReLU(inplace=True),
            CBAM(2048),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(2048, 1024, kernel_size=4, stride=2, padding=1),  # 8x8x1024
            nn.ReLU(inplace=True),
            CBAM(1024),
            nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),  # 16x16x512
            nn.ReLU(inplace=True),
            CBAM(512),
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 32x32x256
            nn.ReLU(inplace=True),
            CBAM(256),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 64x64x128
            nn.ReLU(inplace=True),
            CBAM(128),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 128x128x64
            nn.ReLU(inplace=True),
            CBAM(64),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),  # 256x256x3
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
