# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_norm_layer
from mmcv.runner import BaseModule
from torch import Tensor

from mmdet.models import NECKS

# from mmdet.utils import MultiConfig, OptConfigType

from typing import List, Optional, Sequence, Tuple, Union

# from mmengine.config import ConfigDict
# ConfigType = Union[ConfigDict, dict]
# OptConfigType = Optional[ConfigType]
# MultiConfig = Union[ConfigType, List[ConfigType]]


@NECKS.register_module()
class SimpleFPN(BaseModule):
    """Simple Feature Pyramid Network for ViTDet."""

    def __init__(
        self,
        backbone_channel,
        in_channels,
        out_channels,
        num_outs,
        # num_layers: int = 1, # 默认使用最后一层输入
        layers=[4],  # 使用其中一层或四层都用
        conv_cfg=None,
        norm_cfg=None,
        act_cfg=None,
        init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)
        assert isinstance(in_channels, list)
        self.backbone_channel = backbone_channel
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs

        num_layers = len(layers)
        assert num_layers == 1 or num_layers == 4, "We only support 1 or 4 layers."
        self.layers = layers
        self.num_layers = num_layers

        self.fpn1 = nn.Sequential(
            nn.ConvTranspose2d(self.backbone_channel, self.backbone_channel // 2, 2, 2),
            build_norm_layer(norm_cfg, self.backbone_channel // 2)[1],
            nn.GELU(),
            nn.ConvTranspose2d(
                self.backbone_channel // 2, self.backbone_channel // 4, 2, 2
            ),
        )
        self.fpn2 = nn.Sequential(
            nn.ConvTranspose2d(self.backbone_channel, self.backbone_channel // 2, 2, 2)
        )
        self.fpn3 = nn.Sequential(nn.Identity())
        self.fpn4 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2))

        # self.lateral_convs = nn.ModuleList()
        # self.fpn_convs = nn.ModuleList()

        # for i in range(self.num_ins):
        #     l_conv = ConvModule(
        #         in_channels[i],
        #         out_channels,
        #         1,
        #         conv_cfg=conv_cfg,
        #         norm_cfg=norm_cfg,
        #         act_cfg=act_cfg,
        #         inplace=False)
        #     fpn_conv = ConvModule(
        #         out_channels,
        #         out_channels,
        #         3,
        #         padding=1,
        #         conv_cfg=conv_cfg,
        #         norm_cfg=norm_cfg,
        #         act_cfg=act_cfg,
        #         inplace=False)

        #     self.lateral_convs.append(l_conv)
        #     self.fpn_convs.append(fpn_conv)

    def forward(self, input):
        """Forward function.

        Args:
            inputs (Tensor): Features from the upstream network, 4D-tensor
        Returns:
            tuple: Feature maps, each is a 4D-tensor.
        """
        # build FPN
        if self.num_layers == 1:
            assert len(input) >= 1, "We only use last feature map."
            input = input[self.layers[0] - 1][0]
            # input = input[-1][0] # input[-1][1] is cls_token
            inputs = []
            inputs.append(self.fpn1(input))
            inputs.append(self.fpn2(input))
            inputs.append(self.fpn3(input))
            inputs.append(self.fpn4(input))
            return tuple(inputs)
        elif self.num_layers == 4:
            assert len(input) == 4, "We use 4 levels of feature maps."
            inputs = []
            inputs.append(self.fpn1(input[0][0]))
            inputs.append(self.fpn2(input[1][0]))
            inputs.append(self.fpn3(input[2][0]))
            inputs.append(self.fpn4(input[3][0]))
            return tuple(inputs)
        # # build laterals
        # laterals = [
        #     lateral_conv(inputs[i])
        #     for i, lateral_conv in enumerate(self.lateral_convs)
        # ]

        # # build outputs
        # # part 1: from original levels
        # outs = [self.fpn_convs[i](laterals[i]) for i in range(self.num_ins)]

        # # part 2: add extra levels
        # if self.num_outs > len(outs):
        #     for i in range(self.num_outs - self.num_ins):
        #         outs.append(F.max_pool2d(outs[-1], 1, stride=2))
        # return tuple(outs)
