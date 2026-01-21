import torch
import torch.nn as nn
from mmdet.models import NECKS
from mmdet3d_plugin.models import builder
import torch.nn.functional as F


class MS_CAM_3D(nn.Module):
    "From https://github.com/YimianDai/open-aff/blob/master/aff_pytorch/aff_net/fusion.py"

    def __init__(self, input_channel=64, output_channel=64, r=4):
        super(MS_CAM_3D, self).__init__()
        inter_channels = int(input_channel // r)

        self.local_att = nn.Sequential(
            nn.Conv3d(
                input_channel, inter_channels, kernel_size=1, stride=1, padding=0
            ),
            nn.BatchNorm3d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                inter_channels, output_channel, kernel_size=1, stride=1, padding=0
            ),
            nn.BatchNorm3d(output_channel),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool3d((1, 1, 1)),
            nn.Conv3d(
                input_channel, inter_channels, kernel_size=1, stride=1, padding=0
            ),
            nn.BatchNorm3d(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                inter_channels, output_channel, kernel_size=1, stride=1, padding=0
            ),
            nn.BatchNorm3d(output_channel),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        xl = self.local_att(x)
        xg = self.global_att(x)
        xlg = xl + xg

        return self.sigmoid(xlg)


class MS_CAM_3D_ONE_DIM(nn.Module):
    "From https://github.com/YimianDai/open-aff/blob/master/aff_pytorch/aff_net/fusion.py"

    def __init__(
        self,
        input_channel=64,
        output_channel=64,
        r=4,
        global_attn_dim="wz",
        norm_type="batch",
        num_groups=16,
    ):
        super(MS_CAM_3D_ONE_DIM, self).__init__()
        inter_channels = int(input_channel // r)

        self.global_attn_dim = global_attn_dim

        def make_norm(channels):
            if norm_type == "batch":
                return nn.BatchNorm3d(channels)
            elif norm_type == "group":
                return nn.GroupNorm(num_groups, channels)
            else:
                raise ValueError(f"Unsupported normalization type: {norm_type}")

        self.local_att = nn.Sequential(
            nn.Conv3d(
                input_channel, inter_channels, kernel_size=1, stride=1, padding=0
            ),
            make_norm(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                inter_channels, output_channel, kernel_size=1, stride=1, padding=0
            ),
            make_norm(output_channel),
        )

        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv3d(
                input_channel, inter_channels, kernel_size=1, stride=1, padding=0
            ),
            make_norm(inter_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                inter_channels, output_channel, kernel_size=1, stride=1, padding=0
            ),
            make_norm(output_channel),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        xl = self.local_att(x)

        xg = 0
        if self.global_attn_dim == "wz":
            xg = self.global_att(x)
        elif self.global_attn_dim == "hw":
            xg = self.global_att(x.permute(0, 1, 4, 2, 3)).permute(0, 1, 3, 4, 2)
        elif self.global_attn_dim == "hz":
            xg = self.global_att(x.permute(0, 1, 3, 2, 4)).permute(0, 1, 3, 2, 4)
        else:
            raise ValueError(
                f"Unsupported global attention dimension: {self.global_attn_dim}"
            )
        xlg = xl + xg

        return self.sigmoid(xlg)


@NECKS.register_module()
class DualFeatFusion(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(DualFeatFusion, self).__init__()
        self.ca = MS_CAM_3D(
            input_channel * 2, output_channel
        )  # LSS feature 和 HT feature concatenation

    def forward(self, x1, x2):
        channel_factor = self.ca(torch.cat((x1, x2), 1))  # learning weight
        out = channel_factor * x1 + (1 - channel_factor) * x2

        return out


@NECKS.register_module()
class DualFeatFusion_Tri(nn.Module):
    def __init__(self, input_channel, output_channel, norm_type="batch", num_groups=16):
        super(DualFeatFusion_Tri, self).__init__()
        self.ca_wz = MS_CAM_3D_ONE_DIM(
            input_channel * 2,
            output_channel,
            global_attn_dim="wz",
            norm_type=norm_type,
            num_groups=num_groups,
        )
        self.ca_hw = MS_CAM_3D_ONE_DIM(
            input_channel * 2,
            output_channel,
            global_attn_dim="hw",
            norm_type=norm_type,
            num_groups=num_groups,
        )
        self.ca_hz = MS_CAM_3D_ONE_DIM(
            input_channel * 2,
            output_channel,
            global_attn_dim="hz",
            norm_type=norm_type,
            num_groups=num_groups,
        )

        self.fuse = nn.Sequential(
            nn.Conv3d(input_channel * 3, output_channel, kernel_size=1, bias=False),
            nn.BatchNorm3d(output_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, x1, x2):
        channel_factor_wz = self.ca_wz(torch.cat((x1, x2), 1))
        channel_factor_hw = self.ca_hw(torch.cat((x1, x2), 1))
        channel_factor_hz = self.ca_hz(torch.cat((x1, x2), 1))

        out_wz = channel_factor_wz * x1 + (1 - channel_factor_wz) * x2
        out_hw = channel_factor_hw * x1 + (1 - channel_factor_hw) * x2
        out_hz = channel_factor_hz * x1 + (1 - channel_factor_hz) * x2

        out = out_wz + out_hw + out_hz

        return out


FUSION_MODULES = {
    "DualFeatFusion": DualFeatFusion,
    "DualFeatFusion_Tri": DualFeatFusion_Tri,
}
