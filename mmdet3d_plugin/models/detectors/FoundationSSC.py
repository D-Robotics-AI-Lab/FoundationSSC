import torch
from mmcv.runner import BaseModule
from mmdet.models import DETECTORS
from mmdet3d_plugin.models import builder
from mmdet3d_plugin.models.backbones.FoundationStereo.core.foundation_stereo import (
    FoundationStereo,
)
from omegaconf import OmegaConf


@DETECTORS.register_module()
class FoundationSSC(BaseModule):
    def __init__(
        self,
        img_backbone,
        img_neck,
        depth_net,
        img_pre_neck,
        img_view_transformer,
        proposal_layer,
        VoxFormer_head,
        dual_feature_fusion,
        occ_encoder_backbone,
        occ_encoder_neck,
        pts_bbox_head,
        plugin_head=None,
        depth_loss=True,
        use_semantic=True,
        train_cfg=None,
        test_cfg=None,
    ):
        super().__init__()

        self.gru_iters = (
            img_backbone.gru_iters if hasattr(img_backbone, "gru_iters") else 12
        )

        ckpt = torch.load(img_backbone.ckpt_dir, map_location="cpu")
        cfg = OmegaConf.load(img_backbone.cfg_path)
        if "vit_size" not in cfg:
            cfg["vit_size"] = "vitl"
        args = OmegaConf.create(cfg)

        self.img_backbone = FoundationStereo(args)
        self.img_backbone.load_state_dict(ckpt["model"], strict=False)

        for param in self.img_backbone.parameters():
            param.requires_grad = False

        # dionv2-based pre-neck
        self.img_pre_neck = builder.build_neck(img_pre_neck)
        self.img_neck = builder.build_neck(img_neck)
        self.depth_net = builder.build_neck(depth_net)
        self.img_view_transformer = builder.build_neck(img_view_transformer)
        self.proposal_layer = builder.build_head(proposal_layer)
        self.VoxFormer_head = builder.build_head(VoxFormer_head)
        self.dual_feature_fusion = builder.build_neck(dual_feature_fusion)
        self.occ_encoder_backbone = builder.build_backbone(occ_encoder_backbone)
        self.occ_encoder_neck = builder.build_neck(occ_encoder_neck)
        self.pts_bbox_head = builder.build_head(pts_bbox_head)
        if plugin_head is not None:
            self.plugin_head = builder.build_head(plugin_head)
        
        self.depth_loss = depth_loss
        self.use_semantic = use_semantic

    def foundation_encoder(self, img, raw_imgs):
        B, N, C, imH, imW = img.shape

        # foundationstereo backbone
        raw_left_imgs = [
            raw_img.permute(0, 3, 1, 2)
            for i, raw_img in enumerate(raw_imgs)
            if i % 2 == 0
        ]
        raw_right_imgs = [
            raw_img.permute(0, 3, 1, 2)
            for i, raw_img in enumerate(raw_imgs)
            if i % 2 == 1
        ]

        raw_left_imgs = torch.cat(raw_left_imgs, dim=0)
        raw_right_imgs = torch.cat(raw_right_imgs, dim=0)

        N = img.size(1)

        with torch.no_grad():
            disp_volume, dinov2_feat = self.img_backbone(
                raw_left_imgs, raw_right_imgs, test_mode=True, iters=self.gru_iters
            )

        # dionv2_feat is stereo feature, we only use left image feature
        x = [(feat[:N, ...], cls_token[:N, ...]) for feat, cls_token in dinov2_feat]

        x = self.img_pre_neck(x)
        x = self.img_neck(x)
        if type(x) in [list, tuple]:
            x = x[0]

        _, output_dim, ouput_H, output_W = x.shape
        x = x.view(B, N, output_dim, ouput_H, output_W)

        return x, disp_volume

    def extract_img_feat(self, img_inputs, img_metas):
        left_img_inputs = [
            img_inputs[0][:, ::2],
            img_inputs[1][:, ::2],
            img_inputs[2][:, ::2],
            img_inputs[3][:, ::2],
            img_inputs[4][:, ::2],
            img_inputs[5][:, ::2],
            img_inputs[6],
            img_inputs[7][:, ::2],
            # img_inputs[8][:, ::2], img_inputs[9][:, ::2]
        ]

        img_enc_feats, disp_volume = self.foundation_encoder(
            left_img_inputs[0], img_metas["raw_img"]
        )
        mlp_input = self.depth_net.get_mlp_input(*left_img_inputs[1:7])
        context, depth, stereo_depth = self.depth_net(
            [img_enc_feats] + left_img_inputs[1:7] + [mlp_input], img_metas, disp_volume
        )
        coarse_queries = self.img_view_transformer(context, depth, left_img_inputs[1:7])
        proposal = self.proposal_layer(left_img_inputs[1:7], stereo_depth)

        lss_volume = coarse_queries.clone()

        x = self.VoxFormer_head(
            [context],
            proposal,
            cam_params=left_img_inputs[1:7],
            lss_volume=lss_volume,
            img_metas=img_metas,
            mlvl_dpt_dists=[depth.unsqueeze(1)],
        )

        x = self.dual_feature_fusion(coarse_queries, x)

        if len(context.shape) == 5:
            b, n, d, h, w = context.shape
            context = context.view(b * n, d, h, w)
        return x, context, depth

    def occ_encoder(self, x):
        x = self.occ_encoder_backbone(x)
        x = self.occ_encoder_neck(x)

        return x

    def forward_train(self, data_dict):
        img_inputs = data_dict["img_inputs"]
        img_metas = data_dict["img_metas"]
        gt_occ = data_dict["gt_occ"]
        if self.use_semantic:
            gt_sem = data_dict["gt_semantics"]

        img_voxel_feats, context, depth = self.extract_img_feat(img_inputs, img_metas)
        voxel_feats_enc = self.occ_encoder(img_voxel_feats)
        
        if hasattr(self, "plugin_head"):
            segmentation = self.plugin_head(context)

        if len(voxel_feats_enc) > 1:
            voxel_feats_enc = [voxel_feats_enc[0]]

        if type(voxel_feats_enc) is not list:
            voxel_feats_enc = [voxel_feats_enc]

        output = self.pts_bbox_head(
            voxel_feats=voxel_feats_enc,
            img_metas=img_metas,
            img_feats=None,
            gt_occ=gt_occ,
        )

        losses = dict()

        if self.depth_loss and depth is not None:
            losses["loss_depth"] = self.depth_net.get_depth_loss(
                img_metas["gt_depths"][:, 0:1, ...], depth
            )

        if hasattr(self, "plugin_head") and self.use_semantic:
            losses["loss_seg_ce"] = self.plugin_head.loss(
                pred=segmentation,
                target=gt_sem[:, 0:1, ...],
                depth=img_metas["gt_depths"][:, 0:1, ...],
            )

        losses_occupancy = self.pts_bbox_head.loss(
            output_voxels=output["output_voxels"],
            target_voxels=gt_occ,
        )
        losses.update(losses_occupancy)

        pred = output["output_voxels"]
        pred = torch.argmax(pred, dim=1)

        train_output = {"losses": losses, "pred": pred, "gt_occ": gt_occ}

        return train_output

    def forward_test(self, data_dict):
        img_inputs = data_dict["img_inputs"]
        img_metas = data_dict["img_metas"]
        gt_occ = data_dict["gt_occ"]

        img_voxel_feats, context, depth = self.extract_img_feat(img_inputs, img_metas)
        voxel_feats_enc = self.occ_encoder(img_voxel_feats)

        if len(voxel_feats_enc) > 1:
            voxel_feats_enc = [voxel_feats_enc[0]]

        if type(voxel_feats_enc) is not list:
            voxel_feats_enc = [voxel_feats_enc]

        output = self.pts_bbox_head(
            voxel_feats=voxel_feats_enc,
            img_metas=img_metas,
            img_feats=None,
            gt_occ=gt_occ,
        )

        pred = output["output_voxels"]
        pred = torch.argmax(pred, dim=1)

        test_output = {"pred": pred, "gt_occ": gt_occ}

        return test_output

    def forward(self, data_dict):
        if self.training:
            return self.forward_train(data_dict)
        else:
            return self.forward_test(data_dict)
