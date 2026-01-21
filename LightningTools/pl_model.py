import os
import torch
import numpy as np
import pytorch_lightning as pl
from .basemodel import LightningBaseModel
from .metric import SSCMetrics
from mmdet3d_plugin.models import build_model
from .utils import get_inv_map
from mmcv.runner.checkpoint import load_checkpoint


class pl_model(LightningBaseModel):
    def __init__(self, config):
        super(pl_model, self).__init__(config)

        model_config = config["model"]
        self.model = build_model(model_config)
        if "load_from" in config:
            load_checkpoint(self.model, config["load_from"], map_location="cpu")

        self.num_class = config["num_class"]
        self.class_names = config["class_names"]

        self.train_metrics = SSCMetrics(config["num_class"])
        self.val_metrics = SSCMetrics(config["num_class"])
        self.test_metrics = SSCMetrics(config["num_class"])
        self.save_path = config["save_path"]
        self.test_mapping = config["test_mapping"]

    def forward(self, data_dict):
        return self.model(data_dict)

    def training_step(self, batch, batch_idx):
        output_dict = self.forward(batch)
        loss_dict = output_dict["losses"]
        loss = 0.0
        for key, value in loss_dict.items():
            self.log(
                f"train/{key}",
                value.detach(),
                on_step=True,
                on_epoch=True,
                sync_dist=True,
                prog_bar=False,
            )
            loss += value

        self.log(
            "train/loss",
            loss.detach(),
            on_step=True,
            on_epoch=True,
            sync_dist=True,
            prog_bar=True,
        )

        pred = output_dict["pred"].detach().cpu().numpy()
        gt_occ = output_dict["gt_occ"].detach().cpu().numpy()
        self.train_metrics.add_batch(pred, gt_occ)

        return loss

    def on_train_epoch_end(self):
        stats = self.train_metrics.get_stats()
        dev = self.device

        self.log(
            "train/mIoU",
            torch.as_tensor(stats["iou_ssc_mean"], dtype=torch.float32, device=dev),
            sync_dist=True,
        )
        self.log(
            "train/IoU",
            torch.as_tensor(stats["iou"], dtype=torch.float32, device=dev),
            sync_dist=True,
        )
        self.log(
            "train/Precision",
            torch.as_tensor(stats["precision"], dtype=torch.float32, device=dev),
            sync_dist=True,
        )
        self.log(
            "train/Recall",
            torch.as_tensor(stats["recall"], dtype=torch.float32, device=dev),
            sync_dist=True,
        )

        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):

        output_dict = self.forward(batch)

        pred = output_dict["pred"].detach().cpu().numpy()
        gt_occ = output_dict["gt_occ"].detach().cpu().numpy()

        self.val_metrics.add_batch(pred, gt_occ)

    def on_validation_epoch_end(self):
        stats = self.val_metrics.get_stats()
        dev = self.device

        self.log(
            "val/mIoU",
            torch.as_tensor(stats["iou_ssc_mean"], dtype=torch.float32, device=dev),
            sync_dist=True,
        )
        self.log(
            "val/IoU",
            torch.as_tensor(stats["iou"], dtype=torch.float32, device=dev),
            sync_dist=True,
        )
        self.log(
            "val/Precision",
            torch.as_tensor(stats["precision"], dtype=torch.float32, device=dev),
            sync_dist=True,
        )
        self.log(
            "val/Recall",
            torch.as_tensor(stats["recall"], dtype=torch.float32, device=dev),
            sync_dist=True,
        )

        self.val_metrics.reset()

    def test_step(self, batch, batch_idx):
        output_dict = self.forward(batch)

        pred = output_dict["pred"].detach().cpu().numpy()
        gt_occ = output_dict.get("gt_occ", None)
        if gt_occ is not None:
            gt_occ = gt_occ.detach().cpu().numpy()

        if self.save_path is not None:
            if self.test_mapping:
                inv_map = get_inv_map()
                output_voxels = inv_map[pred].astype(np.uint16)
            else:
                output_voxels = pred.astype(np.uint16)
            sequence_id = batch["img_metas"]["sequence"][0]
            frame_id = batch["img_metas"]["frame_id"][0]
            save_folder = f"{self.save_path}/sequences/{sequence_id}/predictions"
            save_file = os.path.join(save_folder, f"{frame_id}.label")
            os.makedirs(save_folder, exist_ok=True)
            with open(save_file, "wb") as f:
                output_voxels.tofile(f)
            print(f"\n save to {save_file}")

        if gt_occ is not None:
            self.test_metrics.add_batch(pred, gt_occ)

    def on_test_epoch_end(self):
        stats = self.test_metrics.get_stats()
        dev = self.device

        if getattr(self, "global_rank", 0) == 0:
            for name, iou in zip(self.class_names, stats["iou_ssc"]):
                print(name + ":", iou)

        self.log(
            "test/mIoU",
            torch.as_tensor(stats["iou_ssc_mean"], dtype=torch.float32, device=dev),
            sync_dist=True,
        )
        self.log(
            "test/IoU",
            torch.as_tensor(stats["iou"], dtype=torch.float32, device=dev),
            sync_dist=True,
        )
        self.log(
            "test/Precision",
            torch.as_tensor(stats["precision"], dtype=torch.float32, device=dev),
            sync_dist=True,
        )
        self.log(
            "test/Recall",
            torch.as_tensor(stats["recall"], dtype=torch.float32, device=dev),
            sync_dist=True,
        )

        self.test_metrics.reset()
