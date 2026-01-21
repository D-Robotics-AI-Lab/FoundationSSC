import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet3d_plugin.models.builder import NECKS
from mmdet3d_plugin.utils.gaussian import generate_guassian_depth_target
from mmcv.runner import BaseModule, force_fp32
from torch.cuda.amp.autocast_mode import autocast
from .modules.Context_Net_modules import ContextNet
from .modules.DFormerv2 import dformerv2
from .modules.Depth_Net_modules import Disp2DepthChannelMixer, StereoVolumeEncoder
from .modules.utils import disp2depth_map, normalize_depth_to_255, normalize


@NECKS.register_module()
class DSGP_Net(BaseModule):
    def __init__(
        self,
        downsample=8,
        numC_input=512,
        numC_Trans=64,
        cam_channels=27,
        grid_config=None,
        loss_depth_weight=1.0,
        constant_std=0.5,
        loss_depth_type="bce",
        mlp_hidden_dim=256,
    ):
        super(DSGP_Net, self).__init__()

        self.downsample = downsample
        self.numC_input = numC_input
        self.numC_Trans = numC_Trans
        self.cam_channels = cam_channels
        self.grid_config = grid_config

        ds = torch.arange(*self.grid_config["dbound"], dtype=torch.float).view(-1, 1, 1)
        D, _, _ = ds.shape
        self.D = D
        self.cam_depth_range = self.grid_config["dbound"]

        self.context_net = ContextNet(
            self.numC_input,
            self.numC_input,
            self.numC_Trans,
            cam_channels=self.cam_channels,
        )

        self.dformerv2 = dformerv2(
            embed_dims=[128],
            depths=[4],
            num_heads=[8],
            heads_ranges=[4],
        )

        self.loss_depth_weight = loss_depth_weight
        self.loss_depth_type = loss_depth_type
        self.constant_std = constant_std

        # add for foundationstereo
        self.stereo_volume_encoder = StereoVolumeEncoder(
            in_channels=self.D, out_channels=self.D
        )
        self.max_disp = 104  # downsampled 4 times from 416
        self.disp2depth_mlp = Disp2DepthChannelMixer(self.max_disp, 256, 512, self.D, 8)

    def get_downsample_ratio(self, depth_labels, depth_preds):
        """
        Get the downsample ratio of depth labels and predictions.
        """
        H_orig, W_orig = depth_labels.shape[-2:]
        H_pred, W_pred = depth_preds.shape[-2:]
        assert (
            H_orig / H_pred == W_orig / W_pred
        ), "The height and width of depth_labels and depth_preds should have the same downsample ratio."
        downsample_ratio = H_orig // H_pred
        return downsample_ratio

    @force_fp32()
    def get_bce_depth_loss(self, depth_labels, depth_preds):
        downsample_ratio = self.get_downsample_ratio(depth_labels, depth_preds)
        _, depth_labels = self.get_downsampled_gt_depth(
            depth_labels, downsample=downsample_ratio
        )
        # depth_labels = self._prepare_depth_gt(depth_labels)
        depth_preds = depth_preds.permute(0, 2, 3, 1).contiguous().view(-1, self.D)
        fg_mask = torch.max(depth_labels, dim=1).values > 0.0
        depth_labels = depth_labels[fg_mask]
        depth_preds = depth_preds[fg_mask]

        with autocast(enabled=False):
            depth_loss = F.binary_cross_entropy(
                depth_preds, depth_labels, reduction="none"
            ).sum() / max(1.0, fg_mask.sum())

        return depth_loss

    @force_fp32()
    def get_klv_depth_loss(self, depth_labels, depth_preds):
        downsample_ratio = self.get_downsample_ratio(depth_labels, depth_preds)
        depth_gaussian_labels, depth_values = generate_guassian_depth_target(
            depth_labels,
            downsample_ratio,
            self.cam_depth_range,
            constant_std=self.constant_std,
        )

        depth_values = depth_values.view(-1)
        fg_mask = (depth_values >= self.cam_depth_range[0]) & (
            depth_values <= (self.cam_depth_range[1] - self.cam_depth_range[2])
        )

        depth_gaussian_labels = depth_gaussian_labels.view(-1, self.D)[fg_mask]
        depth_preds = (
            depth_preds.permute(0, 2, 3, 1).contiguous().view(-1, self.D)[fg_mask]
        )
        
        depth_loss = F.kl_div(
            torch.log(depth_preds + 1e-4),
            depth_gaussian_labels,
            reduction="batchmean",
            log_target=False,
        )

        return depth_loss

    @force_fp32()
    def get_depth_loss(self, depth_labels, depth_preds):
        if self.loss_depth_type == "bce":
            depth_loss = self.get_bce_depth_loss(depth_labels, depth_preds)

        elif self.loss_depth_type == "kld":
            depth_loss = self.get_klv_depth_loss(depth_labels, depth_preds)

        else:
            raise NotImplementedError(
                f"Depth loss type {self.loss_depth_type} is not implemented."
            )

        return self.loss_depth_weight * depth_loss

    def get_downsampled_gt_depth(self, gt_depths, downsample=8):
        """
        Input:
            gt_depths: [B, N, H, W]
        Output:
            gt_depths: [B*N*h*w, d]
        """
        B, N, H, W = gt_depths.shape
        gt_depths = gt_depths.view(
            B * N, H // downsample, downsample, W // downsample, downsample, 1
        )
        gt_depths = gt_depths.permute(
            0, 1, 3, 5, 2, 4
        ).contiguous()

        gt_depths = gt_depths.view(
            -1, downsample * downsample
        )
        gt_depths_tmp = torch.where(
            gt_depths <= 0.0, 1e5 * torch.ones_like(gt_depths), gt_depths
        )  # 替换 0 为 1e5
        gt_depths = torch.min(
            gt_depths_tmp, dim=-1
        ).values  # use the nearest depth value
        gt_depths = gt_depths.view(
            B * N, H // downsample, W // downsample
        )

        # [min - step / 2, min + step / 2] creates min depth
        gt_depths = (
            gt_depths
            - (self.grid_config["dbound"][0] - self.grid_config["dbound"][2] / 2)
        ) / self.grid_config["dbound"][2]
        gt_depths_vals = gt_depths.clone()

        gt_depths = torch.where(
            (gt_depths < self.D + 1) & (gt_depths >= 0.0),
            gt_depths,
            torch.zeros_like(gt_depths),
        )
        gt_depths = F.one_hot(gt_depths.long(), num_classes=self.D + 1).view(
            -1, self.D + 1
        )[:, 1:]

        return gt_depths_vals, gt_depths.float()

    def get_depth_dist(self, x):
        return x.softmax(dim=1)

    def get_mlp_input(self, rot, tran, intrin, post_rot, post_tran, bda=None):
        B, N, _, _ = rot.shape

        if bda is None:
            bda = torch.eye(3).to(rot).view(1, 3, 3).repeat(B, 1, 1)

        bda = bda.view(B, 1, *bda.shape[-2:]).repeat(1, N, 1, 1)

        if intrin.shape[-1] == 4:
            # for KITTI, the intrin matrix is 3x4
            mlp_input = torch.stack(
                [
                    intrin[:, :, 0, 0],
                    intrin[:, :, 1, 1],
                    intrin[:, :, 0, 2],
                    intrin[:, :, 1, 2],
                    intrin[:, :, 0, 3],
                    intrin[:, :, 1, 3],
                    intrin[:, :, 2, 3],
                    post_rot[:, :, 0, 0],
                    post_rot[:, :, 0, 1],
                    post_tran[:, :, 0],
                    post_rot[:, :, 1, 0],
                    post_rot[:, :, 1, 1],
                    post_tran[:, :, 1],
                    bda[:, :, 0, 0],
                    bda[:, :, 0, 1],
                    bda[:, :, 1, 0],
                    bda[:, :, 1, 1],
                    bda[:, :, 2, 2],
                ],
                dim=-1,
            )

            if bda.shape[-1] == 4:
                mlp_input = torch.cat((mlp_input, bda[:, :, :3, -1]), dim=2)
        else:
            mlp_input = torch.stack(
                [
                    intrin[:, :, 0, 0],
                    intrin[:, :, 1, 1],
                    intrin[:, :, 0, 2],
                    intrin[:, :, 1, 2],
                    post_rot[:, :, 0, 0],
                    post_rot[:, :, 0, 1],
                    post_tran[:, :, 0],
                    post_rot[:, :, 1, 0],
                    post_rot[:, :, 1, 1],
                    post_tran[:, :, 1],
                    bda[:, :, 0, 0],
                    bda[:, :, 0, 1],
                    bda[:, :, 1, 0],
                    bda[:, :, 1, 1],
                    bda[:, :, 2, 2],
                ],
                dim=-1,
            )

        sensor2ego = torch.cat([rot, tran.reshape(B, N, 3, 1)], dim=-1).reshape(
            B, N, -1
        )
        mlp_input = torch.cat([mlp_input, sensor2ego], dim=-1)

        return mlp_input

    def forward(self, input, img_metas, disp_volume=None):
        x, rots, trans, intrins, post_rots, post_trans, bda, mlp_input = input

        if disp_volume is not None:
            assert isinstance(
                disp_volume, list
            ), f"disp_volume should be a list, but got {type(disp_volume)}"

            final_disp = disp_volume[1]
            disp_volume = disp_volume[0]

            intrins_resize = post_rots @ intrins[:, :, :3, :3]

            stereo_depth = disp2depth_map(
                final_disp, img_metas["baseline"].squeeze(), intrins_resize[:, 0, :, :]
            )
        else:
            stereo_depth = img_metas["stereo_depth"]

        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)  # [1, 640, 48, 160]

        # geometry-aware context adapter
        img_feat = self.context_net(x, mlp_input)  # [1, 128, 48, 160]
        
        dformer_depth = stereo_depth.clone()
        dformer_depth[dformer_depth > self.cam_depth_range[1]] = self.cam_depth_range[1]
        dformer_depth[dformer_depth < 0] = 0

        modal_x = normalize(normalize_depth_to_255(dformer_depth), [0.48], [0.28])
        img_feat = self.dformerv2(img_feat, modal_x)[0]

        # disp2depth volume mapping
        _, disp_c, disp_h, disp_w = disp_volume.shape
        disp_volume = disp_volume.float()

        assert (
            disp_c == self.max_disp
        ), f"disp_volume should have {self.max_disp} channels, but got {disp_c} channels"

        stereo_volume = self.disp2depth_mlp(
            disp_volume
        )  # [1, 104, 96, 320] -> [1, 112, 96, 320]
        stereo_volume = self.stereo_volume_encoder(
            stereo_volume
        )  # one-hot to distribution
        stereo_volume = self.get_depth_dist(stereo_volume)  # softmax

        stereo_volume = F.interpolate(
            stereo_volume, scale_factor=0.5, mode="bilinear", align_corners=True
        )

        return img_feat.view(B, N, -1, H, W), stereo_volume, stereo_depth
