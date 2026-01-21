import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_norm_layer

norm_cfg = dict(type="GN", num_groups=2, requires_grad=True)


def convbn_2d(in_channels, out_channels, kernel_size, stride, pad):
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=pad,
            bias=False,
        ),
        build_norm_layer(norm_cfg, out_channels)[1],
    )


class SimpleUnet(nn.Module):
    def __init__(self, in_channels):
        super(SimpleUnet, self).__init__()

        self.conv1 = nn.Sequential(
            convbn_2d(in_channels, in_channels * 2, 3, 2, 1), nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            convbn_2d(in_channels * 2, in_channels * 2, 3, 1, 1), nn.ReLU(inplace=True)
        )

        self.conv3 = nn.Sequential(
            convbn_2d(in_channels * 2, in_channels * 4, 3, 2, 1), nn.ReLU(inplace=True)
        )

        self.conv4 = nn.Sequential(
            convbn_2d(in_channels * 4, in_channels * 4, 3, 1, 1), nn.ReLU(inplace=True)
        )

        self.conv5 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels * 4,
                in_channels * 2,
                3,
                padding=1,
                output_padding=1,
                stride=2,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels * 2),
        )

        self.conv6 = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels * 2,
                in_channels,
                3,
                padding=1,
                output_padding=1,
                stride=2,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
        )

        self.redir1 = convbn_2d(
            in_channels, in_channels, kernel_size=1, stride=1, pad=0
        )
        self.redir2 = convbn_2d(
            in_channels * 2, in_channels * 2, kernel_size=1, stride=1, pad=0
        )

    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)

        conv5 = F.relu(self.conv5(conv4) + self.redir2(conv2), inplace=True)
        conv6 = F.relu(self.conv6(conv5) + self.redir1(x), inplace=True)

        return conv6


class StereoVolumeEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(StereoVolumeEncoder, self).__init__()
        self.stem = convbn_2d(in_channels, out_channels, kernel_size=3, stride=1, pad=1)
        self.Unet = nn.Sequential(SimpleUnet(out_channels))
        self.conv_out = nn.Conv2d(
            out_channels, out_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.Unet(x)
        x = self.conv_out(x)
        return x


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class ChannelMixerBlock(nn.Module):
    def __init__(self, dim, channel_dim, dropout=0.0):
        super().__init__()

        self.channel_mix = nn.Sequential(
            nn.LayerNorm(dim), FeedForward(dim, channel_dim, dropout)
        )

    def forward(self, x):
        x = x + self.channel_mix(x)

        return x


class Disp2DepthChannelMixer(nn.Module):
    def __init__(self, in_channels, dim, channel_dim, out_channels, depth):
        super().__init__()
        if in_channels != dim:
            self.do_stem = True
            self.conv = nn.Conv2d(in_channels, dim, kernel_size=1)
        else:
            self.do_stem = False
        self.mixer_blocks = nn.ModuleList()
        for _ in range(depth):
            self.mixer_blocks.append(ChannelMixerBlock(dim, channel_dim))

        self.layer_norm = nn.LayerNorm(dim)

        self.mlp_head = nn.Sequential(nn.Linear(dim, out_channels))

    def forward(self, x):
        if self.do_stem:
            x = self.conv(x)
        B, C, H, W = x.shape
        x = x.view(B, C, -1).transpose(1, 2)  # [B, H*W, C]
        for mixer_block in self.mixer_blocks:
            x = mixer_block(x)

        x = self.layer_norm(x)

        x = self.mlp_head(x)

        x = x.transpose(1, 2).view(B, -1, H, W)
        return x
